"""可观测路由（契约 5.9）。

**一句话定位**：把 `traces` / `trace_spans` 变成面板上的数字与瀑布图数据。

**在链路中的位置**：前端 `/obs` 与 `/obs/traces/:traceId` -> **本模块** ->
`observability.metrics`（聚合）+ `traces` / `trace_spans` / `messages` 表（明细）。

**关键设计取舍**：

1. **本文件里有 `select()`，这是刻意的例外**（见 `api/__init__.py`）。
   这里的查询是**纯只读聚合**：没有业务规则、不改状态、被复用的可能性为零
   （只有面板会用"最近 N 小时的 trace 列表"）。为它建一个 service，
   只会多一层什么都不做的转发。`observability/metrics.py` 已经把重活都做完了，
   本文件只负责"过滤条件 + 分页 + 串起来"。
2. **`/obs/stats` 必须把 `collect_stats` 的扁平结果改写成 `ObsStatsOut` 的嵌套形状**。
   两者字段名不同（`latency_ms` vs `latency`、`series` vs `timeline`），
   前端已经按嵌套形状写好归一化（`frontend/src/api/obs.ts`）。
   **不要"顺手"把接口改成服务层形状**：那会让前端所有 ECharts 图变成空白。
3. **`/obs/traces/{trace_id}` 的 `messages` 回裸 dict 列表**（不是 `MessageOut`）。
   这些消息是**排障用的旁证**，需要的字段（角色、内容摘要、token、cost）与
   会话回放页的 `MessageOut` 不完全一致；前端按 `MessageOut` 形状容错读取。
   回裸 dict 让本接口可以在不触发会话模块序列化开销的前提下给出足够信息。
4. **时间窗口默认 24 小时**，与契约一致；`hours` 由用户传，服务端会夹到上限
   （见 `observability.metrics._MAX_SERIES_HOURS`），避免一句 `hours=100000`
   把整张表拉进内存。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import ColumnElement, func, select

from knowflow.api.deps import Services, SettingsDep
from knowflow.core.exceptions import TraceNotFoundError
from knowflow.db.models.chat import Message
from knowflow.db.models.observability import Trace, TraceSpan
from knowflow.db.types import utcnow
from knowflow.observability.metrics import collect_quality, collect_stats
from knowflow.schemas.common import Page, PageParams
from knowflow.schemas.observability import (
    ByModelOut,
    ByModeOut,
    LatencyOut,
    ObsStatsOut,
    QualityOut,
    TimelinePointOut,
    TokenStatsOut,
    TraceDetailOut,
    TraceOut,
    TraceSpanOut,
)

__all__ = ["router"]

router = APIRouter(prefix="/obs", tags=["可观测"])

PageDep = Annotated[PageParams, Depends()]

HoursQuery = Annotated[int, Query(ge=1, le=24 * 31, description="统计窗口（小时），默认 24")]

#: trace 明细里每条消息最多回多少条：排障只需要"这次请求带了哪些上下文"，
#: 一个长会话可能有上千条消息，全量返回会把详情页拖死。
_MAX_DETAIL_MESSAGES = 50

#: 内容摘要长度（字符）。排障要的是"模型大概看到了什么"，不是全文。
_SNIPPET_CHARS = 300


@router.get("/stats", response_model=ObsStatsOut, summary="观测统计")
def obs_stats(services: Services, settings: SettingsDep, hours: HoursQuery = 24) -> ObsStatsOut:
    """窗口内的请求量 / token / 成本 / 延迟分位 / 按模式与模型分组 / 时间序列。"""
    raw = collect_stats(services.session, hours=hours, settings=settings)
    return _to_stats_out(raw)


@router.get("/quality", response_model=QualityOut, summary="质量指标")
def obs_quality(services: Services, hours: HoursQuery = 24) -> QualityOut:
    """拒答率 / 引用数分布 / 零引用率 / 反思通过率。

    `reflect_pass_rate` 是**近似值**（见 `observability/metrics.collect_quality` 的说明），
    它由"该 trace 的 reflect span 没有报错"近似，会略高估。
    `collect_quality` 返回的 `reflect_pass_rate_approx=True` 不进契约模型，
    但前端读的是这个接口的字段——**不要把近似说成精确**。
    """
    raw = collect_quality(services.session, hours=hours)
    return QualityOut.model_validate(
        {
            "hours": raw["hours"],
            "total": raw["requests"],
            "refusal_rate": raw["refusal_rate"],
            "avg_source_count": raw["avg_source_count"],
            "zero_source_rate": raw["zero_source_rate"],
            "avg_retrieval_rounds": raw["avg_retrieval_rounds"],
            "reflect_pass_rate": raw["reflect_pass_rate"],
        }
    )


@router.get("/traces", response_model=Page[TraceOut], summary="trace 列表")
def list_traces(
    params: PageDep,
    services: Services,
    status_filter: Annotated[str | None, Query(alias="status", description="ok / error")] = None,
    mode: Annotated[str | None, Query(description="按模式过滤，如 agent / rag")] = None,
    hours: Annotated[int | None, Query(ge=1, le=24 * 31, description="只看最近 N 小时")] = None,
) -> Page[TraceOut]:
    conditions: list[ColumnElement[bool]] = []
    if status_filter:
        conditions.append(Trace.status == status_filter)
    if mode:
        conditions.append(Trace.mode == mode)
    if hours is not None:
        conditions.append(Trace.created_at >= utcnow() - timedelta(hours=hours))

    total = int(
        services.session.execute(select(func.count(Trace.id)).where(*conditions)).scalar() or 0
    )
    rows = (
        list(
            services.session.execute(
                select(Trace)
                .where(*conditions)
                # 最新在前：排障看的永远是"刚刚那次"
                .order_by(Trace.created_at.desc(), Trace.id.desc())
                .offset(params.offset)
                .limit(params.limit)
            ).scalars()
        )
        if total
        else []
    )
    return Page[TraceOut].create(
        items=[TraceOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.get("/traces/{trace_id}", response_model=TraceDetailOut, summary="trace 详情（瀑布图）")
def get_trace(trace_id: str, services: Services) -> TraceDetailOut:
    """`{trace, spans, messages}` 一次返回。

    `spans` 按 `seq` 升序：`start_offset_ms` 是相对 trace 起点的偏移，
    加上 `duration_ms` 就能画出瀑布图（前端不需要处理时区与时钟漂移）。
    """
    trace = services.session.execute(
        select(Trace).where(Trace.trace_id == trace_id)
    ).scalar_one_or_none()
    if trace is None:
        raise TraceNotFoundError(f"trace {trace_id} 不存在")

    spans = list(
        services.session.execute(
            select(TraceSpan)
            .where(TraceSpan.trace_id == trace_id)
            .order_by(TraceSpan.seq, TraceSpan.id)
        ).scalars()
    )
    messages = list(
        services.session.execute(
            select(Message)
            .where(Message.trace_id == trace_id)
            .order_by(Message.id)
            .limit(_MAX_DETAIL_MESSAGES)
        ).scalars()
    )

    return TraceDetailOut(
        trace=TraceOut.model_validate(trace),
        spans=[TraceSpanOut.model_validate(row) for row in spans],
        messages=[_message_summary(row) for row in messages],
    )


def _message_summary(message: Message) -> dict[str, Any]:
    """消息摘要素描（裸 dict，见模块 docstring 第 3 条）。

    字段刻意与 `MessageOut` 对齐（前端按它读取），但 `content` 截断到
    `_SNIPPET_CHARS`：详情页只是要看清"这次请求的上下文是什么"，
    全文可以从 `/conversations/{id}/messages` 拿。
    `citations` 回空列表而不是省略：前端会做 `.map()`，缺字段会直接崩。
    """
    content = message.content or ""
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": content[:_SNIPPET_CHARS] + ("…" if len(content) > _SNIPPET_CHARS else ""),
        "mode": message.mode,
        "model": message.model,
        "prompt_tokens": message.prompt_tokens,
        "completion_tokens": message.completion_tokens,
        "cost_usd": float(message.cost_usd or 0.0),
        "latency_ms": message.latency_ms,
        "refusal": bool(message.refusal),
        "retrieval_rounds": message.retrieval_rounds,
        "trace_id": message.trace_id,
        "citations": [],
        "created_at": message.created_at,
    }


def _to_stats_out(raw: dict[str, Any]) -> ObsStatsOut:
    """`collect_stats` 的扁平结果 -> 契约 `ObsStatsOut` 的嵌套形状。

    三个必须做的改名（前端已按目标形状写好，改错就是一堆空图表）：

    - `latency_ms` -> `latency`（且只保留 p50/p95/p99/max/avg，丢掉 count）
    - `series[].hour` -> `timeline[].bucket`（另有 `errors` 字段，契约里没有位置）
    - `by_mode` 是 `{mode: {...}}` 字典 -> `by_mode[]` 数组，并补上
      `refusal_rate`（`collect_stats` 没按模式算拒答，用请求数为 0 时给 0 兜底）
    """
    latency_raw = raw.get("latency_ms") or {}
    tokens_prompt = int(raw.get("prompt_tokens") or 0)
    tokens_completion = int(raw.get("completion_tokens") or 0)

    by_mode: list[ByModeOut] = []
    for mode, payload in sorted((raw.get("by_mode") or {}).items()):
        by_mode.append(
            ByModeOut(
                mode=str(mode),
                requests=int(payload.get("requests") or 0),
                tokens=int(payload.get("tokens") or 0),
                cost_usd=float(payload.get("cost_usd") or 0.0),
                avg_latency_ms=float(payload.get("avg_latency_ms") or 0.0),
                refusal_rate=float(payload.get("refusal_rate") or 0.0),
            )
        )

    by_model: list[ByModelOut] = []
    for payload in raw.get("by_model") or []:
        # `collect_stats` 的 by_model 只有 messages 口径的 token/cost（traces 表没有 model 列），
        # 所以 `requests` 用 messages 条数填——前端的标签就叫"消息数"，不会误读。
        by_model.append(
            ByModelOut(
                model=str(payload.get("model") or "unknown"),
                requests=int(payload.get("messages") or 0),
                prompt_tokens=int(payload.get("prompt_tokens") or 0),
                completion_tokens=int(payload.get("completion_tokens") or 0),
                cost_usd=float(payload.get("cost_usd") or 0.0),
                avg_latency_ms=0.0,
            )
        )

    timeline: list[TimelinePointOut] = []
    for point in raw.get("series") or []:
        timeline.append(
            TimelinePointOut(
                bucket=point.get("hour"),
                requests=int(point.get("requests") or 0),
                tokens=int(point.get("tokens") or 0),
                # 这两个字段 `collect_stats` 没有按小时算（只按小时给了 requests/errors/tokens）。
                # **给 0 而不是编一个值**：面板上会显示 0，看的人知道"这个粒度没采"，
                # 编一个数才是真正有害的（会让人以为某小时成本是 0 而放松警惕）。
                cost_usd=0.0,
                avg_latency_ms=0.0,
                refusal_rate=0.0,
            )
        )

    return ObsStatsOut(
        hours=int(raw.get("hours") or 24),
        requests=int(raw.get("requests") or 0),
        tokens=TokenStatsOut(
            prompt=tokens_prompt,
            completion=tokens_completion,
            total=tokens_prompt + tokens_completion,
        ),
        cost_usd=float(raw.get("cost_usd") or 0.0),
        cost_cny=float(raw.get("cost_cny") or 0.0),
        latency=LatencyOut(
            p50=float(latency_raw.get("p50") or 0.0),
            p95=float(latency_raw.get("p95") or 0.0),
            p99=float(latency_raw.get("p99") or 0.0),
            max=float(latency_raw.get("max") or 0.0),
            avg=float(latency_raw.get("avg") or 0.0),
        ),
        by_mode=by_mode,
        by_model=by_model,
        timeline=timeline,
    )
