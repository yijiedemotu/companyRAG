"""向量化实现的工厂 + 自动降级包装。

**一句话定位**：按配置造出具体的 `Embedder`，并保证「主实现挂了也不会让链路断」——
统一降级到 `HashEmbedder`，同时把降级事实如实记在 `status` 上供 `/health` 上报。

**在整条链路中的位置**：`container` / `main` 启动时调一次 `build_embedder`，
之后 ingest / retrieve / 评测都共用这个实例（必须是同一个实例：
同一句话在入库侧和检索侧要落在同一个语义空间）。
`get_embedder_status` 是 `/health` 里 `embedding_mode` 的唯一数据来源。

**关键设计取舍**：

1. **降级放在包装器里，不放在各实现里**：local 加载失败、api 401、api 超时
   都要走同一条路。写在一处才不会出现「本地实现会降级、api 实现直接 502」
   这种不一致；也让三个实现本身保持纯粹（只管好自己的事）。
2. **降级是「懒」的，不在 `build_embedder` 里加载模型**：local 是惰性加载，
   启动时加载 2GB 权重会拖慢所有接口，离线 CI 更不该加载。
   想提前暴露问题就在启动流程里调 `warmup_embedder(embedder)`。
3. **降级只发生一次且不可逆**：一旦确认主实现不可用就永久切到兜底，
   避免每次请求都先失败再降级（那会把延迟和日志都毁掉）。
4. **降级后 `provider` 变成 `hash`**：`provider` 表达「真正在用的实现」，
   而不是「配置里写的那个」——否则写进 KB 快照的 provider 就是假的。
   配置里写的那个在 `status.requested_provider` 里，两个都在，不丢信息。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import structlog

from knowflow.core.config import Settings, get_settings
from knowflow.embeddings.api_embedder import ApiEmbedder
from knowflow.embeddings.base import Embedder
from knowflow.embeddings.hash_embedder import HashEmbedder
from knowflow.embeddings.local_bge import LocalBgeEmbedder

logger = structlog.get_logger(__name__)

_T = TypeVar("_T")

# 降级原因用「短标识」而不是整段异常文本：
# 它要拼进 /health 的 embedding_mode（形如 hash(fallback:local_load_failed)），
# 完整异常文本进日志（degrade_detail 字段与 warning 日志里都有）。
_REASON_INIT_FAILED = "provider_init_failed"
_REASON_LOAD_FAILED = "load_failed"
_REASON_RUNTIME_FAILED = "runtime_failed"


@dataclass(slots=True)
class EmbedderStatus:
    """向量化的**真实**运行状态（`/health` 如实上报的依据）。"""

    requested_provider: str
    """配置里写的 provider（`EMBEDDING_PROVIDER`）。"""

    active_provider: str
    """真正在用的 provider；发生降级时是 `hash`。"""

    model: str
    dim: int
    display_name: str
    degraded: bool
    degrade_reason: str | None
    degrade_detail: str | None = None
    """降级时的原始异常文本（排查用；`degrade_reason` 只是给机器看的短标识）。"""

    @property
    def mode(self) -> str:
        """给 `/health` 的 `embedding_mode` 用，格式见接口契约。

        例：正常是 `local`，降级后是 `hash(fallback:load_failed)`。
        """
        if self.degraded and self.degrade_reason:
            return f"{self.active_provider}(fallback:{self.degrade_reason})"
        return self.active_provider


class _DegradingEmbedder(Embedder):
    """把任意 `Embedder` 包成「主实现失败就永久切 hash 兜底」的包装器。

    协议里的 `provider` / `model` / `dim` / `display_name` 用**普通属性**保存并在
    降级时刷新（不用 property 委托）：协议把她们声明成可写属性，
    用只读 property 实现会在 mypy 严格模式下被判为不兼容。
    """

    def __init__(
        self,
        settings: Settings,
        primary: Embedder,
        *,
        init_error: str | None = None,
    ) -> None:
        self._settings = settings
        self._primary = primary
        self._fallback: Embedder | None = None
        self._lock = threading.Lock()

        self.requested_provider = settings.embedding_provider
        self.provider = primary.provider
        self.model = primary.model
        self.dim = primary.dim
        self.display_name = primary.display_name
        self.status = EmbedderStatus(
            requested_provider=self.requested_provider,
            active_provider=primary.provider,
            model=primary.model,
            dim=primary.dim,
            display_name=primary.display_name,
            degraded=False,
            degrade_reason=None,
        )
        if init_error is not None:
            # 构造期就失败（例如 api 没有 Key）：此时 primary 已经是兜底实现，
            # 只需要如实把状态标成降级，不用再造一个 fallback。
            self._mark_degraded(active=primary, reason=_REASON_INIT_FAILED, detail=init_error)

    # ------------------------------------------------------------ 状态查询
    @property
    def is_loaded(self) -> bool:
        """当前生效实现是否已就绪（hash 恒为 True，local 未加载时为 False）。"""
        return bool(getattr(self._active(), "is_loaded", True))

    def _active(self) -> Embedder:
        return self._fallback if self._fallback is not None else self._primary

    # ---------------------------------------------------------------- 预热
    def warmup(self) -> None:
        """预热当前实现；失败则自动降级（不再抛出）。"""
        active = self._active()
        warmup = getattr(active, "warmup", None)
        if not callable(warmup):
            return
        try:
            warmup()
        except Exception as exc:  # noqa: BLE001 - 降级是这里唯一正确的处理方式
            self._degrade(reason=_REASON_LOAD_FAILED, detail=_describe(exc), failed=active)

    # ---------------------------------------------------------------- 编码
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码；主实现失败时降级并对同一批输入重试一次。"""
        return self._with_fallback(lambda e: e.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """编码查询；主实现失败时降级并重试一次。"""
        return self._with_fallback(lambda e: e.embed_query(text))

    def _with_fallback(self, operation: Callable[[Embedder], _T]) -> _T:
        active = self._active()
        try:
            return operation(active)
        except Exception as exc:
            if self._fallback is not None:
                # 兜底实现本身也失败了：真的没辙了，原样抛出，
                # 让上层转成 UPSTREAM_ERROR（而不是假装成功）。
                raise
            self._degrade(reason=_REASON_RUNTIME_FAILED, detail=_describe(exc), failed=active)
            return operation(self._active())

    # ------------------------------------------------------------ 降级实现
    def _degrade(self, *, reason: str, detail: str, failed: Embedder) -> None:
        with self._lock:
            if self._fallback is None:
                self._fallback = HashEmbedder(self._settings)
            active = self._fallback
            self._mark_degraded(active=active, reason=reason, detail=detail)
        logger.warning(
            "向量化实现失败，已自动降级到 hash 兜底（检索质量会明显下降，"
            "但链路不中断；请在 /health 里确认 embedding_mode）",
            requested_provider=self.requested_provider,
            failed_provider=failed.provider,
            degrade_reason=reason,
            error=detail,
        )

    def _mark_degraded(self, *, active: Embedder, reason: str, detail: str) -> None:
        """刷新对外属性与状态；调用方负责加锁规则（此处只做赋值）。"""
        # 兜底实现与主实现的 dim 都取自 EMBEDDING_DIM，所以降级不会让
        # 向量库维度错位；这里仍然从 active 取，保持「看到的即真实的」。
        self.provider = active.provider
        self.model = active.model
        self.dim = active.dim
        self.display_name = active.display_name
        self.status = EmbedderStatus(
            requested_provider=self.requested_provider,
            active_provider=active.provider,
            model=active.model,
            dim=active.dim,
            display_name=active.display_name,
            degraded=True,
            degrade_reason=reason,
            degrade_detail=detail,
        )


def _describe(exc: BaseException) -> str:
    """异常转一行文本（写日志与 `degrade_detail` 用）。"""
    return f"{type(exc).__name__}: {exc}"


def _create_embedder(settings: Settings) -> Embedder:
    """按 `EMBEDDING_PROVIDER` 造主实现（不含降级包装）。"""
    provider = settings.embedding_provider
    if provider == "local":
        return LocalBgeEmbedder(settings)
    if provider == "api":
        return ApiEmbedder(settings)
    return HashEmbedder(settings)


def build_embedder(settings: Settings | None = None) -> Embedder:
    """造出本项目使用的向量化实现（已带自动降级）。

    返回的对象除了协议要求的成员，还额外暴露：
    - `status: EmbedderStatus`（`/health` 读它）、
    - `warmup()` / `is_loaded`（预热与就绪判断）。

    注意：这里**不会**加载模型（local 是惰性加载）。想提前知道向量化是否可用，
    启动流程里调 `warmup_embedder(build_embedder(settings))`，然后读 status。
    """
    resolved = settings if settings is not None else get_settings()
    try:
        primary = _create_embedder(resolved)
    except Exception as exc:  # noqa: BLE001 - 初始化失败也要让链路能起来
        detail = _describe(exc)
        logger.warning(
            "向量化实现初始化失败，直接使用 hash 兜底",
            requested_provider=resolved.embedding_provider,
            error=detail,
        )
        return _DegradingEmbedder(resolved, HashEmbedder(resolved), init_error=detail)
    return _DegradingEmbedder(resolved, primary)


def get_embedder_status(embedder: Embedder) -> EmbedderStatus:
    """统一取状态：降级包装器有 `status`，裸实现没有，就按协议属性现造一个。

    `/health` 只调这一个函数，所以「embedding_mode 是否如实」只需要在这里保证。
    """
    status = getattr(embedder, "status", None)
    if isinstance(status, EmbedderStatus):
        return status
    provider = str(getattr(embedder, "provider", "unknown"))
    return EmbedderStatus(
        requested_provider=provider,
        active_provider=provider,
        model=str(getattr(embedder, "model", "unknown")),
        dim=int(getattr(embedder, "dim", 0)),
        display_name=str(getattr(embedder, "display_name", provider)),
        degraded=False,
        degrade_reason=None,
    )


__all__ = ["EmbedderStatus", "build_embedder", "get_embedder_status"]
