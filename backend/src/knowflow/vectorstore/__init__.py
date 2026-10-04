"""KnowFlow 向量库层：`VectorStore` 协议 + Chroma 实现 + 内存实现。

对外只需一个入口 `build_vector_store(settings)`：按 `VECTOR_BACKEND` 选实现，
Chroma 起不来时降级到内存实现（并打 warning，`/health` 里能看见 `volatile`）。

**两个实现的差别只有持久化**：同样的 `score` 约定（越大越相似）、同样的
delete 语义、同样的 metadata 规范化。这是「单测用 memory、生产用 chroma」
能够成立的前提。
"""

from __future__ import annotations

from knowflow.vectorstore.base import (
    KB_COLLECTION_PREFIX,
    OPTIONAL_METADATA_KEYS,
    REQUIRED_METADATA_KEYS,
    VectorHit,
    VectorItem,
    VectorStore,
    VectorStoreError,
    collection_name,
    matches_delete_filter,
    metadata_matches,
    sanitize_metadata,
    validate_where,
)
from knowflow.vectorstore.chroma_store import ChromaVectorStore
from knowflow.vectorstore.factory import build_vector_store
from knowflow.vectorstore.memory_store import InMemoryVectorStore

__all__ = [
    "KB_COLLECTION_PREFIX",
    "OPTIONAL_METADATA_KEYS",
    "REQUIRED_METADATA_KEYS",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "VectorHit",
    "VectorItem",
    "VectorStore",
    "VectorStoreError",
    "build_vector_store",
    "collection_name",
    "matches_delete_filter",
    "metadata_matches",
    "sanitize_metadata",
    "validate_where",
]
