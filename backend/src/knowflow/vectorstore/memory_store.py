"""纯内存向量库（numpy 矩阵点积），行为与 Chroma 实现严格对齐。

**一句话定位**：用 `dict[kb_id, dict[vector_id, VectorItem]]` 装数据、
用 numpy 点积做余弦检索的向量库。

**在整条链路中的位置**：两个用途 ——

1. **单测**：不落盘、不依赖 Chroma 的存储层，跑得快；「换后端不换结果」
   是检索层测试的地基，所以本文件与 `chroma_store.py` 必须给出同样的 top-1；
2. **`VECTOR_BACKEND=memory`**：无磁盘模式（CI、演示沙箱），同时也是
   Chroma 初始化失败时 `factory` 的自动降级目标（重启即丢，所以要打 warning）。

**关键设计取舍**：

1. **点积即余弦**：写入时统一 `l2_normalize`（即使调用方忘了归一化），
   查询向量也归一化，于是「矩阵点积 == 余弦相似度」——这正是 Chroma
   `cosine` 空间的行为，两个后端的 score 才对得上（浮点误差约 1e-6）。
   代价：矩阵在每次查询时重建（O(N)），十万级以下完全够用，
   真上量就该换 Milvus/pgvector 而不是优化这里。
2. **metadata 走同一个 `sanitize_metadata`**：Chroma 会静默丢弃值为 None 的键，
   内存实现必须跟着丢，否则「两个后端行为一致」就是一句空话。
3. **用 `RLock`**：FastAPI 把同步检索丢进线程池（`run_in_threadpool`），
   多个请求会并发读写同一份结构；用 RLock 而不是 Lock，是因为
   `count()` / `delete()` 内部还会调 `_bucket()`，普通 Lock 会把同一线程锁死。
4. **排序确定**：候选先按 id 排序，再 `np.argsort(kind="stable")`，
   分数并列时结果稳定可复现。注意 Chroma 的 HNSW 在分数并列时顺序不可预测，
   所以跨后端的测试**不要**断言并列项的先后。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from knowflow.core.config import Settings, get_settings
from knowflow.embeddings.base import l2_normalize
from knowflow.vectorstore.base import (
    VectorHit,
    VectorItem,
    VectorStoreError,
    matches_delete_filter,
    metadata_matches,
    sanitize_metadata,
    validate_where,
)

logger = structlog.get_logger(__name__)


class InMemoryVectorStore:
    """进程内向量库（`VECTOR_BACKEND=memory`）。"""

    backend = "memory"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self.dim = self._settings.embedding_dim
        # kb_id -> (vector_id -> VectorItem)；向量已归一化
        self._data: dict[int, dict[str, VectorItem]] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- 写入
    def upsert(self, *, kb_id: int, items: Sequence[VectorItem]) -> int:
        """幂等写入（同 id 覆盖）；返回本批处理的条数（与 Chroma 实现一致）。"""
        if not items:
            return 0
        prepared: list[VectorItem] = []
        for item in items:
            vector = list(item.vector)
            if len(vector) != self.dim:
                raise VectorStoreError(
                    f"向量维度不符（id={item.id}）：收到 {len(vector)} 维，"
                    f"集合是 {self.dim} 维。换 embedding 模型必须重建向量库。"
                )
            prepared.append(
                VectorItem(
                    id=item.id,
                    vector=l2_normalize(vector),
                    content=item.content,
                    metadata=sanitize_metadata(kb_id, item.metadata),
                )
            )
        with self._lock:
            bucket = self._data.setdefault(kb_id, {})
            for item in prepared:
                bucket[item.id] = item
        return len(prepared)

    # ---------------------------------------------------------------- 检索
    def query(
        self,
        *,
        kb_id: int,
        vector: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """矩阵点积取 top_k；`score` 是余弦相似度（越大越相似）。"""
        if top_k <= 0:
            raise VectorStoreError(f"top_k 必须大于 0，收到 {top_k}")
        query_vector = list(vector)
        if len(query_vector) != self.dim:
            raise VectorStoreError(
                f"查询向量维度不符：收到 {len(query_vector)} 维，集合是 {self.dim} 维"
            )
        validate_where(where)

        with self._lock:
            bucket = self._data.get(kb_id)
            if not bucket:
                return []
            candidates = sorted(
                (item for item in bucket.values() if metadata_matches(item.metadata, where)),
                key=lambda item: item.id,
            )
            if not candidates:
                return []
            matrix = np.asarray([item.vector for item in candidates], dtype=np.float32)
            query_array = np.asarray(l2_normalize(query_vector), dtype=np.float32)
            # 两侧都已归一化 → 点积就是余弦相似度，与 Chroma 的 1 - distance 等价
            scores = matrix @ query_array
            order = np.argsort(-scores, kind="stable")[:top_k]
            return [
                VectorHit(
                    id=candidates[index].id,
                    score=float(scores[index]),
                    content=candidates[index].content,
                    metadata=dict(candidates[index].metadata),
                )
                for index in order
            ]

    # ------------------------------------------------------------ 删除/统计
    def delete(
        self,
        *,
        kb_id: int,
        doc_id: int | None = None,
        vector_ids: Sequence[str] | None = None,
    ) -> int:
        """删除向量；三种组合的语义与 Chroma 实现共用 `matches_delete_filter`。"""
        with self._lock:
            bucket = self._data.get(kb_id)
            if not bucket:
                return 0
            if doc_id is None and vector_ids is None:
                # 清整个 KB：内存实现直接丢掉这个桶
                return len(self._data.pop(kb_id, {}))
            wanted = set(vector_ids) if vector_ids is not None else None
            targets = [
                item_id
                for item_id, item in bucket.items()
                if matches_delete_filter(
                    item_id=item_id,
                    metadata=item.metadata,
                    doc_id=doc_id,
                    vector_ids=wanted,
                )
            ]
            for item_id in targets:
                del bucket[item_id]
            return len(targets)

    def count(self, *, kb_id: int | None = None) -> int:
        """统计向量条数；`kb_id=None` 表示全部 KB。"""
        with self._lock:
            if kb_id is not None:
                return len(self._data.get(kb_id, {}))
            return sum(len(bucket) for bucket in self._data.values())

    def list_vector_ids(self, *, kb_id: int, doc_id: int | None = None) -> list[str]:
        """只取 id；返回**按 id 排序**的列表（Chroma 不保证顺序，测试别依赖它）。"""
        with self._lock:
            bucket = self._data.get(kb_id)
            if not bucket:
                return []
            if doc_id is None:
                return sorted(bucket)
            return sorted(
                item_id
                for item_id, item in bucket.items()
                if matches_delete_filter(
                    item_id=item_id,
                    metadata=item.metadata,
                    doc_id=doc_id,
                    vector_ids=None,
                )
            )

    def reset(self, *, kb_id: int | None = None) -> None:
        """清空指定 KB 或全部（`kb_id=None`）。"""
        with self._lock:
            if kb_id is None:
                self._data.clear()
            else:
                self._data.pop(kb_id, None)

    # ---------------------------------------------------------------- 健康
    def health(self) -> dict[str, Any]:
        """`/health` 用的自述信息。`volatile=True` 表示重启即丢，必须让运维看见。"""
        with self._lock:
            return {
                "backend": self.backend,
                "kb_collections": len(self._data),
                "total_vectors": sum(len(bucket) for bucket in self._data.values()),
                "dim": self.dim,
                "volatile": True,
            }


__all__ = ["InMemoryVectorStore"]
