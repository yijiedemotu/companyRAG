"""全链路埋点：零侵入的 span 收集器。

**一句话定位**：一次请求内的所有步骤耗时/输入/输出/异常，用上下文管理器或装饰器
收集起来，最后一次性落成 ``traces`` + ``trace_spans`` 两行数据。

**在链路中的位置**：

    ChatService 起 trace
      -> TraceRecorder.start()
      -> 每个节点 @traced("retrieve") / with rec.span(...)   （业务代码几乎不用改）
      -> persist(recorder, session, ...) -> traces / trace_spans
      -> observability.metrics.collect_stats() -> /obs/stats 与 /metrics

**关键设计取舍**：

1. **零侵入靠 `ContextVar`，不靠"传参"**。把 recorder 一路当参数传下去会污染
   所有函数签名（尤其是 LangGraph 的节点签名是固定的）。`ContextVar` 在
   `asyncio` 任务里自动隔离，同时 `threading.Lock` 保证多线程写 span 不丢数据
   （LangGraph 的节点可能跑在线程池里，contextvar 会随 `run_in_threadpool` 复制，
   但 `list.append` 与计数器仍必须加锁）。
2. **埋点绝不能影响主流程**。序列化失败、日志失败、落库失败——三类失败全部
   吞掉只记日志。理由：可观测是"旁路"，为了记录而让用户请求 500 是本末倒置。
   唯一的例外是**业务异常必须原样抛出**（`span()` 里 `raise` 而不是吞掉）。
3. **`start_offset_ms` 存相对偏移而不是绝对时间戳**：前端画瀑布图只需要
   「起点偏移 + 时长」，不用处理时区、时钟漂移与多机时间不同步，
   存储也从 datetime 降成 int。
4. **payload 用"白名单 + 截断"而不是 `json.dumps(default=str)`**：
   `json.dumps` 遇到 `Session` 会试着序列化整个 ORM 对象（连带 lazy 关系），
   既慢又可能触发额外 SQL；白名单直接把它标成 `_unserializable`。
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from types import TracebackType
from typing import Any, Literal, Self, TypeVar, cast

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.db.models.observability import (
    SPAN_LLM,
    SPAN_NODE,
    SPAN_RETRIEVAL,
    STATUS_ERROR,
    STATUS_OK,
    Trace,
    TraceSpan,
    truncate_payload,
)

logger = get_logger(__name__)

#: span 名/错误字段长度上限，与 `trace_spans.name String(64)` / `error String(512)` 对齐。
#: 不对齐的话，超长名字会在 MySQL 严格模式下直接 INSERT 失败，把整棵树都丢掉。
_MAX_NAME_CHARS = 64
_MAX_ERROR_CHARS = 512

#: 序列化时的最大递归深度与单容器元素数：
#: 防的是"对象互相引用"造成的无限递归，以及一次检索返回上千条候选撑爆 JSON 列。
_MAX_DEPTH = 6
_MAX_ITEMS = 50

#: 单个字符串在埋点阶段的粗截阈值（落库前还会再过一次 `truncate_payload`）。
_PRE_TRUNCATE_CHARS = 2000

#: 命中这些**名字**子串的 span 计入 `rerank_ms`。
#: 为什么按名字而不只按 span_type：召回与重排都是 `retrieval` 类型，
#: 只有名字能区分二者，而 `traces` 表要求把 `retrieval_ms` 与 `rerank_ms` 分开计。
_RERANK_NAME_HINTS = ("rerank", "重排")
_GENERATE_NAME_HINTS = ("generate", "answer", "生成")


@dataclass
class SpanRecord:
    """一个 span 的"可落库形态"。

    刻意**不是** ORM 对象：`TraceRecorder` 可能在没有数据库会话的地方使用
    （脚本、单元测试、离线评测），不应该强迫调用方准备 Session。
    `seq` 单调递增，是前端瀑布图的稳定排序键（同毫秒的 span 不会乱序）。
    """

    name: str
    span_type: str
    start_offset_ms: int
    duration_ms: int
    status: str = STATUS_OK
    input_json: Any | None = None
    output_json: Any | None = None
    error: str | None = None
    seq: int = 0


# --------------------------------------------------------------------------------------
# payload 序列化：白名单 + 截断，永不抛
# --------------------------------------------------------------------------------------
def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    """把任意对象转成"能进 JSON 列"的结构；不认识的对象降级成类型名描述。

    **绝不抛异常**是这个函数的契约：它跑在埋点路径上，抛一次就等于让业务请求失败。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > _PRE_TRUNCATE_CHARS:
            return value[:_PRE_TRUNCATE_CHARS] + f"…(+{len(value) - _PRE_TRUNCATE_CHARS}字)"
        return value
    if isinstance(value, dict):
        if _depth >= _MAX_DEPTH:
            return {"_depth_limit": True}
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:_MAX_ITEMS]:
            out[str(key)] = _jsonable(item, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        if _depth >= _MAX_DEPTH:
            return ["_depth_limit"]
        return [_jsonable(item, _depth=_depth + 1) for item in list(value)[:_MAX_ITEMS]]
    # 到这里就是"不认识的对象"：Session / VectorStore / 自定义 dataclass 等。
    # 只记类型名，不记 repr()：repr 会把 ORM 对象的字段（可能含正文、路径）带进 trace。
    return {"_unserializable": f"{type(value).__module__}.{type(value).__name__}"}


def _dump(payload: Any) -> Any:
    """`_jsonable` + 一次真实 `json.dumps` 试跑。

    为什么要额外 dumps 一遍：SQLAlchemy 的 JSON 列**在 flush 时才序列化**，
    那时才发现不可序列化对象，异常会发生在事务提交阶段（更难定位，
    而且会连带把同事务的业务写入一起回滚）。提前在这里验证，失败就降级。
    """
    if payload is None:
        return None
    try:
        safe = _jsonable(payload)
        json.dumps(safe, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 - 埋点不能影响主流程
        return {"_unserializable": f"{type(payload).__module__}.{type(payload).__name__}"}
    return safe


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class SpanContext:
    """`TraceRecorder.span()` 产出的上下文对象。

    额外提供 `set_output()`：很多 span 的产出"进入之后才知道"
    （检索命中几条、模型返回多少 token），必须在 `with` 内部回填。
    """

    __slots__ = ("_ended", "_recorder", "record")

    def __init__(self, recorder: TraceRecorder, record: SpanRecord) -> None:
        self._recorder = recorder
        self.record = record
        self._ended = False

    # -- 便利访问：`with rec.span("retrieve") as sp: sp.set_output(...)` --
    @property
    def name(self) -> str:
        return self.record.name

    @property
    def duration_ms(self) -> int:
        return self.record.duration_ms

    @property
    def status(self) -> str:
        return self.record.status

    def set_output(self, output: Any) -> None:
        """回填输出（序列化失败自动降级，不抛）。"""
        self.record.output_json = _dump(output)

    def set_input(self, payload: Any) -> None:
        self.record.input_json = _dump(payload)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        # 返回 False = **不吞异常**。这是硬规则：埋点只观察，不改变控制流。
        # 返回 True 会把"检索失败"记成成功，用户拿到空答案却看不到错误。
        # 返回类型写 `Literal[False]`（而不是 `bool`）：mypy strict 会把 `bool`
        # 当成"可能返回 True"，从而认为这个上下文管理器可能吞掉异常，直接报错。
        _ = (exc_type, tb)
        self._recorder.finish(self, exc)
        return False


class TraceRecorder:
    """一次请求的 span 收集器（线程安全）。"""

    def __init__(self, trace_id: str, name: str, *, settings: Settings | None = None) -> None:
        self.trace_id = trace_id
        self.name = name
        self.settings = settings or get_settings()
        self._lock = threading.RLock()
        self._records: list[SpanRecord] = []
        self._start: float | None = None
        self._seq = 0
        # ctx_id -> 起跑时刻。用 id(ctx) 而不是 ctx 本身做键：
        # dataclass 默认按值比较，用它当字典键会在两个字段相同的 span 之间撞车。
        self._inflight: dict[int, float] = {}

    # ----------------------------------------------------------------------------------
    # 时间与生命周期
    # ----------------------------------------------------------------------------------
    def start(self) -> None:
        """记 t0。**幂等**：重复调用不重置起点（否则并发分支的偏移会全部错位）。"""
        with self._lock:
            if self._start is None:
                self._start = time.monotonic()

    def elapsed_ms(self) -> int:
        """从 `start()` 到现在的毫秒数；未 start 过则返回 0。"""
        with self._lock:
            start = self._start
        if start is None:
            return 0
        return round((time.monotonic() - start) * 1000)

    def _offset_now(self, start: float) -> int:
        return max(round((time.monotonic() - start) * 1000), 0)

    # ----------------------------------------------------------------------------------
    # 记录 span
    # ----------------------------------------------------------------------------------
    def add_span(
        self,
        name: str,
        *,
        span_type: str = SPAN_NODE,
        start_offset_ms: int | None = None,
        duration_ms: int = 0,
        status: str = STATUS_OK,
        input: Any = None,
        output: Any = None,
        error: str | None = None,
    ) -> SpanRecord:
        """手动追加一个 span。

        流式响应（SSE）下"一个步骤的结束"与"函数返回"不同步，没法用 `with`，
        只能手工记开始/结束时刻。`start_offset_ms` 因此允许外部传入
        （相对 `start()` 的偏移），不传则取当前时刻。
        """
        self.start()
        payload_in = _dump(input)
        payload_out = _dump(output)
        with self._lock:
            assert self._start is not None  # start() 刚保证过；assert 让 mypy 放心
            offset = (
                start_offset_ms if start_offset_ms is not None else self._offset_now(self._start)
            )
            self._seq += 1
            record = SpanRecord(
                name=_short(name, _MAX_NAME_CHARS),
                span_type=span_type,
                start_offset_ms=max(int(offset), 0),
                duration_ms=max(int(duration_ms), 0),
                status=status,
                input_json=payload_in,
                output_json=payload_out,
                error=_short(error, _MAX_ERROR_CHARS) if error else None,
                seq=self._seq,
            )
            self._records.append(record)
            return record

    @contextmanager
    def span(
        self,
        name: str,
        *,
        span_type: str = SPAN_NODE,
        input: Any = None,
        output: Any = None,
    ) -> Iterator[SpanContext]:
        """计时上下文管理器；**异常记录后照原样抛出**。"""
        self.start()
        with self._lock:
            assert self._start is not None
            offset = self._offset_now(self._start)
        record = self.add_span(
            name,
            span_type=span_type,
            start_offset_ms=offset,
            input=input,
            output=output,
        )
        ctx = SpanContext(self, record)
        with self._lock:
            self._inflight[id(ctx)] = time.monotonic()
        try:
            yield ctx
        except BaseException as exc:  # 记录后必须原样抛出（BaseException：Ctrl-C 也要留痕）
            self.finish(ctx, exc)
            raise
        else:
            self.finish(ctx, None)

    def finish(self, ctx: SpanContext, exc: BaseException | None) -> None:
        """收口：算耗时（只算一次）、记异常。**不吞异常**（由调用方决定）。"""
        with self._lock:
            if ctx._ended:
                return  # 双保险：`with` 与手动 finish 混用时不重复计时
            ctx._ended = True
            started = self._inflight.pop(id(ctx), None)
        record = ctx.record
        if started is not None:
            record.duration_ms = max(round((time.monotonic() - started) * 1000), 0)
        if exc is not None:
            record.status = STATUS_ERROR
            record.error = _short(f"{type(exc).__name__}: {exc}", _MAX_ERROR_CHARS)

    def spans(self) -> list[SpanRecord]:
        """返回 span 快照（新 list，按 seq 升序），调用方可安全遍历。"""
        with self._lock:
            return sorted(self._records, key=lambda item: item.seq)

    # ----------------------------------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """给 `/obs` 与日志用的汇总。

        口径说明：`retrieval_ms` / `rerank_ms` / `generate_ms` 是**同类 span 耗时之和**，
        不是墙钟时间。多个检索并行时三者之和可能大于 `elapsed_ms`——这是有意的：
        我们要回答"花在检索上的时间是多少"，而不是"第几秒在做检索"。
        """
        records = self.spans()
        by_type: dict[str, dict[str, int]] = {}
        status_counts: dict[str, int] = {STATUS_OK: 0, STATUS_ERROR: 0}

        retrieval_ms = 0
        rerank_ms = 0
        generate_ms = 0
        llm_calls = 0

        for record in records:
            bucket = by_type.setdefault(
                record.span_type, {"count": 0, "duration_ms": 0, "errors": 0}
            )
            bucket["count"] += 1
            bucket["duration_ms"] += record.duration_ms
            if record.status == STATUS_ERROR:
                bucket["errors"] += 1

            status_counts[record.status] = status_counts.get(record.status, 0) + 1

            lowered = record.name.lower()
            if any(hint in lowered for hint in _RERANK_NAME_HINTS):
                rerank_ms += record.duration_ms
            elif record.span_type == SPAN_RETRIEVAL:
                retrieval_ms += record.duration_ms
            elif any(hint in lowered for hint in _GENERATE_NAME_HINTS):
                generate_ms += record.duration_ms

            if record.span_type == SPAN_LLM:
                llm_calls += 1

        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "elapsed_ms": self.elapsed_ms(),
            "span_count": len(records),
            "by_type": by_type,
            "by_status": status_counts,
            "llm_calls": llm_calls,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generate_ms": generate_ms,
            "spans": [
                {
                    "seq": item.seq,
                    "name": item.name,
                    "span_type": item.span_type,
                    "start_offset_ms": item.start_offset_ms,
                    "duration_ms": item.duration_ms,
                    "status": item.status,
                }
                for item in records
            ],
        }


# --------------------------------------------------------------------------------------
# 上下文变量与装饰器
# --------------------------------------------------------------------------------------
#: 当前请求的 recorder。默认 None —— 脚本/单元测试里直接用被装饰的函数**也不会报错**，
#: 只是"没有记录"而已。这个默认值是"埋点可缺席"这条设计能成立的前提。
current_recorder: ContextVar[TraceRecorder | None] = ContextVar("current_recorder", default=None)


def get_recorder() -> TraceRecorder | None:
    """取当前上下文的 recorder，没有就返回 None（不抛异常）。"""
    return current_recorder.get()


@contextmanager
def use_recorder(rec: TraceRecorder) -> Iterator[TraceRecorder]:
    """把 recorder 绑定到当前上下文，退出时**恢复原值**（不是清空）。

    用 `reset(token)` 而不是 `set(None)`：后台任务里可能嵌套绑定
    （外层请求 recorder + 内层子任务 recorder），简单清空会把外层的也弄丢。
    """
    token = current_recorder.set(rec)
    rec.start()
    try:
        yield rec
    finally:
        current_recorder.reset(token)


F = TypeVar("F", bound=Callable[..., Any])


def traced(
    name: str,
    *,
    span_type: str = SPAN_NODE,
    capture: bool = True,
) -> Callable[[F], F]:
    """装饰器：自动给函数加一个 span。**没有 recorder 时直接执行，零开销**。

    `capture=True` 会记录入参与返回值，但会：
    - 过滤不可 JSON 序列化的对象（`Session` / `VectorStore` / 自定义类实例）；
    - 截断超长字符串与超长容器；
    - 对 `self` / `cls` 只记类型名——把 service 对象整个序列化会连带它的
      `Session`、配置甚至 API Key 字段，这是**信息泄露**风险。

    用 `functools.wraps` 保留签名元信息：否则 FastAPI 的依赖注入与
    LangGraph 的节点 introspection 会认不出被装饰的函数。
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rec = current_recorder.get()
            if rec is None:
                # 无 recorder：只多一次 contextvar 读（纳秒级），不做任何序列化
                return func(*args, **kwargs)

            payload: Any = None
            if capture:
                payload = {"args": _dump(args), "kwargs": _dump(kwargs)}
            with rec.span(name, span_type=span_type, input=payload) as span_ctx:
                result = func(*args, **kwargs)
                if capture:
                    span_ctx.set_output(result)
                return result

        return cast(F, wrapper)  # cast：wrapper 通过 wraps 保住了签名，mypy 看不出来

    return decorator


# --------------------------------------------------------------------------------------
# 落库
# --------------------------------------------------------------------------------------
def _estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int, settings: Settings
) -> Decimal:
    """薄封装：让 `persist` 不必在模块顶层 import pricing（少一跳 import 链）。"""
    from knowflow.observability.pricing import estimate_cost

    return estimate_cost(model, prompt_tokens, completion_tokens, settings)


def _safe_log_error(event: str, trace_id: str | None, exc: BaseException) -> None:
    """日志本身也要包住。

    structlog 渲染某些类型时可能出错（而且此刻我们已经在异常路径上），
    再抛一次会彻底掩盖真实原因——排障时最怕的就是"错误被错误覆盖"。
    """
    try:
        logger.error(event, trace_id=trace_id, error=f"{type(exc).__name__}: {exc}")
    except Exception:  # noqa: BLE001, S110 - 这里"静默"是刻意的：连日志都不可用时无处可写
        pass


def _safe_rollback(session: Session) -> None:
    """回滚，但**不让回滚本身再抛**。

    persist 失败后会话可能处于 "PendingRollbackError" 状态，
    此时业务侧任何 `commit()` 都会连带失败；rollback 把它拉回可用状态。
    连 rollback 都失败（连接已断）时只能记日志。
    """
    try:
        session.rollback()
    except Exception as exc:  # noqa: BLE001
        _safe_log_error("observability.persist.rollback_failed", None, exc)


def persist(
    recorder: TraceRecorder,
    session: Session,
    *,
    name: str = "chat",
    mode: str | None = None,
    user_id: int | None = None,
    conversation_id: int | None = None,
    kb_id: int | None = None,
    request_id: str | None = None,
    status: str = STATUS_OK,
    usage: dict[str, Any] | None = None,
    latency_ms: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """把 recorder 的 span 落到 `trace_spans`，并 upsert 一条 `traces` 行。

    返回 `trace_id`；**失败时同样返回 trace_id，但不抛异常**。

    ★ **硬规则**：可观测失败绝不能影响业务。这里任何异常（表不存在、
    连接断开、字段超长）都只记 error 日志 + rollback，让调用方的请求继续走完。

    为什么 upsert 而不是 insert：流式场景会"边跑边落"，同一次请求可能被
    persist 两次（首帧落一次、结束时补全一次）。按 `trace_id` 查到就更新，
    保证始终只有一行 trace。

    为什么 `trace_spans` 要"先删后插"：同理，第二次 persist 若直接 append
    会让 span 翻倍，瀑布图出现"重影"——而且**不会报错**，只是图看起来怪。
    删的时候用 Core `delete()` + `flush()`：显式让 DELETE 先落到连接上，
    避免 unit of work 把 INSERT 排在 DELETE 前面（MySQL 上没有唯一约束兜底，
    顺序错了就是静默重复）。
    """
    trace_id = recorder.trace_id
    try:
        settings = recorder.settings
        summary = recorder.summary()
        usage = usage or {}

        prompt_tokens = max(int(usage.get("prompt_tokens") or 0), 0)
        completion_tokens = max(int(usage.get("completion_tokens") or 0), 0)
        model = str(usage.get("model") or settings.openai_model)
        llm_calls = int(usage.get("llm_calls") or summary.get("llm_calls") or 0)
        cost_usd = _estimate_cost(model, prompt_tokens, completion_tokens, settings)
        total_latency = int(latency_ms if latency_ms is not None else summary["elapsed_ms"])

        row = session.execute(select(Trace).where(Trace.trace_id == trace_id)).scalar_one_or_none()
        if row is None:
            row = Trace(trace_id=trace_id)
            session.add(row)

        row.request_id = request_id
        row.user_id = user_id
        row.conversation_id = conversation_id
        row.kb_id = kb_id
        row.name = _short(name, _MAX_NAME_CHARS)
        row.mode = mode
        row.status = status
        row.latency_ms = total_latency
        row.retrieval_ms = int(summary.get("retrieval_ms") or 0)
        row.rerank_ms = int(summary.get("rerank_ms") or 0)
        row.generate_ms = int(summary.get("generate_ms") or 0)
        row.llm_calls = llm_calls
        row.prompt_tokens = prompt_tokens
        row.completion_tokens = completion_tokens
        row.cost_usd = cost_usd

        extra = extra or {}
        row.retrieval_rounds = int(extra.get("retrieval_rounds") or 0)
        row.source_count = int(extra.get("source_count") or 0)
        row.refusal = bool(extra.get("refusal") or False)
        error_text = extra.get("error")
        row.error = _short(str(error_text), _MAX_ERROR_CHARS) if error_text else None

        # 覆盖式写入 span：见 docstring（避免重复 persist 造成重影）
        session.execute(delete(TraceSpan).where(TraceSpan.trace_id == trace_id))
        session.flush()

        for record in recorder.spans():
            session.add(
                TraceSpan(
                    trace_id=trace_id,
                    seq=record.seq,
                    name=_short(record.name, _MAX_NAME_CHARS),
                    span_type=record.span_type,
                    start_offset_ms=record.start_offset_ms,
                    duration_ms=record.duration_ms,
                    status=record.status,
                    # ★ 必须过 truncate_payload：span 是排障用的，不是存全文的。
                    # 一次问答的上下文可能 20KB，几万次请求后 JSON 列会把表撑爆。
                    input_json=truncate_payload(record.input_json),
                    output_json=truncate_payload(record.output_json),
                    error=_short(record.error, _MAX_ERROR_CHARS) if record.error else None,
                )
            )
        session.flush()
    except Exception as exc:  # noqa: BLE001 - ★硬规则：可观测失败不能让业务请求失败
        _safe_log_error("observability.persist.failed", trace_id, exc)
        _safe_rollback(session)
    return trace_id


__all__ = [
    "SpanContext",
    "SpanRecord",
    "TraceRecorder",
    "current_recorder",
    "get_recorder",
    "persist",
    "traced",
    "use_recorder",
]
