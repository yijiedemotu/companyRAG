"""向量库工厂：按配置选实现，Chroma 起不来就降级到内存。

**一句话定位**：`VECTOR_BACKEND=chroma|memory` 的唯一切换点。

**在整条链路中的位置**：`container` / `main` 启动时调一次 `build_vector_store`，
之后 ingest / retrieve / 一致性校验共用它。`/health` 的 `vector_backend` 与
`vector_count` 都来自这个对象。

**关键设计取舍**：

- **降级到内存必须吵**：Chroma 初始化失败（目录权限、磁盘满、依赖缺失）时
  如果直接让进程起不来，本地开发体验很差；但如果静默降级，用户会以为
  「数据存住了」，重启后全部丢失。所以这里降级 + `warning` 日志 +
  `health()` 里的 `volatile: True`，三处一起让它可见。
- 选 `memory` 是**显式配置**（单测、CI、演示），不是失败，所以不打 warning。
"""

from __future__ import annotations

import structlog

from knowflow.core.config import Settings, get_settings
from knowflow.vectorstore.base import VectorStore
from knowflow.vectorstore.chroma_store import ChromaVectorStore
from knowflow.vectorstore.memory_store import InMemoryVectorStore

logger = structlog.get_logger(__name__)


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    """按 `VECTOR_BACKEND` 造向量库；Chroma 初始化失败时降级到内存实现。

    注意：`chroma` 与 `memory` 的**行为**是一致的（同 score 约定、同 delete 语义），
    差别只在持久化 —— 这正是能让单测用内存跑、生产用 Chroma 的前提。
    """
    resolved = settings if settings is not None else get_settings()
    if resolved.vector_backend == "memory":
        logger.info("使用内存向量库（无持久化，进程退出即丢）", dim=resolved.embedding_dim)
        return InMemoryVectorStore(resolved)

    try:
        store = ChromaVectorStore(resolved)
    except Exception as exc:  # noqa: BLE001 - 起不来也要让服务能跑
        logger.warning(
            "Chroma 向量库初始化失败，降级到内存实现（重启后向量全部丢失）",
            path=str(resolved.chroma_path),
            error=f"{type(exc).__name__}: {exc}",
        )
        return InMemoryVectorStore(resolved)
    return store


__all__ = ["build_vector_store"]
