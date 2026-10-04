"""可观测模块的桥接层。

**为什么需要这一层**：`observability/` 是横切关注点，业务代码不应该因为它
「没被装载 / 版本不匹配 / 导入时报错」就跑不起来。所以这里做一次
防御性导入，对外暴露稳定的名字：

- 正常情况：转发到真实实现；
- 任何异常：退化成 no-op（埋点丢失，但业务照常）。

这条规则和"降级可以，静默不行"并不冲突 —— 这里的降级会在首次失败时
打一条 warning 日志，而不是完全静默。区别在于：**丢埋点不影响用户拿到答案**，
而丢向量化会让答案质量崩掉，两者不该同样处理。
"""

from __future__ import annotations

from contextlib import nullcontext
from decimal import Decimal
from typing import Any

from knowflow.core.logging import get_logger

logger = get_logger(__name__)

_bridge_error: str | None = None

try:  # pragma: no cover - 依赖加载路径，正常环境恒为真
    from knowflow.observability.pricing import estimate_cost, to_cny
    from knowflow.observability.tracing import (
        TraceRecorder,
        get_recorder,
        persist,
        use_recorder,
    )

    OBSERVABILITY_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    _bridge_error = f"{type(exc).__name__}: {exc}"
    OBSERVABILITY_AVAILABLE = False
    TraceRecorder = None  # type: ignore[assignment,misc]

    def get_recorder() -> Any:  # type: ignore[misc]
        return None

    def use_recorder(recorder: Any) -> Any:  # type: ignore[misc]
        return nullcontext(recorder)

    def persist(*_args: Any, **_kwargs: Any) -> str | None:  # type: ignore[misc]
        return None

    def estimate_cost(*_args: Any, **_kwargs: Any) -> Decimal:  # type: ignore[misc]
        return Decimal("0")

    def to_cny(*_args: Any, **_kwargs: Any) -> Decimal:  # type: ignore[misc]
        return Decimal("0")


def build_recorder(name: str, *, trace_id: str, settings: Any | None = None) -> Any:
    """为一次请求创建 recorder。

    **`trace_id` 必须由调用方传入**：一次请求的 trace_id 同时也是
    LangGraph 的 `thread_id`、`messages.trace_id` 与 `/chat` 响应里的 `trace_id`。
    如果让 recorder 自己生成，就会出现"响应里的 trace_id 和库里存的不是同一个"，
    这种不一致几乎无法排查。所以 trace_id 是**请求级唯一标识**，在最外层生成一次。
    """
    if not OBSERVABILITY_AVAILABLE or TraceRecorder is None:
        if _bridge_error:
            logger.debug("observability.unavailable", error=_bridge_error)
        return None
    try:
        return TraceRecorder(trace_id, name, settings=settings)  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        logger.warning("observability.recorder_failed", error=f"{type(exc).__name__}: {exc}")
        return None


def recorder_context(recorder: Any) -> Any:
    """把 recorder 装进上下文；`recorder` 为 None 时退化成 nullcontext。

    **不能直接把 None 传给 `use_recorder`**：真实实现会在进入时调 `recorder.start()`，
    None 会抛 `AttributeError`。可观测性拿不到 recorder 是完全正常的（脚本、单测、
    埋点模块加载失败），业务必须照常跑。
    """
    if recorder is None:
        return nullcontext(None)
    try:
        return use_recorder(recorder)
    except Exception as exc:  # noqa: BLE001
        logger.warning("observability.use_recorder_failed", error=f"{type(exc).__name__}: {exc}")
        return nullcontext(None)


def safe_persist(
    recorder: Any,
    session: Any,
    *,
    name: str,
    mode: str | None = None,
    user_id: int | None = None,
    conversation_id: int | None = None,
    kb_id: int | None = None,
    request_id: str | None = None,
    status: str = "ok",
    usage: dict[str, Any] | None = None,
    latency_ms: int = 0,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """落库 trace。**任何异常都吞掉**：可观测失败不该让一次成功的问答变成 500。

    **这里负责 commit**（真实实现的 `persist` 只 flush，事务边界交给调用方）。
    为什么把 commit 放在这一层而不是每个调用点：调用点有 4 处（非流式成功/失败、
    流式成功/落库失败），任何一处忘了 commit 都会变成
    "接口一切正常、但 /obs 里一条 trace 都没有" —— 一个静默丢失埋点的 bug。
    放在唯一的出口上，这类错误在结构上就不可能发生。
    """
    if recorder is None or not OBSERVABILITY_AVAILABLE:
        return None
    try:
        trace_id = persist(
            recorder,
            session,
            name=name,
            mode=mode,
            user_id=user_id,
            conversation_id=conversation_id,
            kb_id=kb_id,
            request_id=request_id,
            status=status,
            usage=usage,
            latency_ms=latency_ms,
            extra=extra,
        )
        session.commit()
        return trace_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("observability.persist_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        try:
            session.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


__all__ = [
    "OBSERVABILITY_AVAILABLE",
    "build_recorder",
    "estimate_cost",
    "get_recorder",
    "recorder_context",
    "safe_persist",
    "to_cny",
    "use_recorder",
]
