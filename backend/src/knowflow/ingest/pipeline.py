"""入库编排（纯计算部分，不碰数据库）。

**为什么把"纯计算"和"写库"分开**：

`prepare_chunks` / `embed_chunks` 是**纯函数**（输入文档对象，输出切片与向量），
可以脱离 MySQL、向量库、文件系统单测 —— 这是测试能在几秒内跑完的原因。
而"写三处（向量库 / BM25 / MySQL）并保证一致性"是有副作用的编排，
放在 `services/document.py`。

**写入顺序是刻意设计的**：向量库 → BM25 → 注册表（MySQL）。
中途失败最坏只是向量库里多了"孤儿 chunk"（`delete_document` 能清掉），
而不是"列表显示上传成功、实际检索不到"—— 后者才是最难的排查场景，
因为所有接口都返回 200，只有用户觉得"答不准"。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.embeddings.base import Embedder
from knowflow.ingest.chunkers import Chunk, split_document
from knowflow.ingest.loaders import ParsedDocument

logger = get_logger(__name__)


@dataclass(slots=True)
class PreparedChunks:
    chunks: list[Chunk]
    total_chars: int
    total_tokens: int
    split_ms: int

    @property
    def count(self) -> int:
        return len(self.chunks)


def prepare_chunks(document: ParsedDocument, *, settings: Settings | None = None) -> PreparedChunks:
    """解析结果 → 切片（含父块与出处元信息）。"""
    cfg = settings or get_settings()
    started = time.perf_counter()
    chunks = split_document(
        document,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        min_chars=cfg.chunk_min_chars,
        parent_chunk_size=cfg.parent_chunk_size,
    )
    elapsed = int((time.perf_counter() - started) * 1000)
    return PreparedChunks(
        chunks=chunks,
        total_chars=sum(c.char_count for c in chunks),
        total_tokens=sum(c.token_count for c in chunks),
        split_ms=elapsed,
    )


def embed_chunks(
    chunks: Sequence[Chunk], *, embedder: Embedder, settings: Settings | None = None
) -> tuple[list[list[float]], int]:
    """批量向量化。返回 `(向量列表, 耗时ms)`。

    空输入直接返回空，**不要调用 embedder** —— 有些实现会在空输入时报错，
    而"空文档"是合法输入（虽然会被上层拦掉）。
    """
    if not chunks:
        return [], 0
    started = time.perf_counter()
    vectors = embedder.embed_documents([c.content for c in chunks])
    elapsed = int((time.perf_counter() - started) * 1000)

    if len(vectors) != len(chunks):
        raise ValueError(
            f"向量化返回条数({len(vectors)})与切片数({len(chunks)})不一致，"
            "这会导致向量与正文错位，必须视为致命错误"
        )
    return vectors, elapsed


def build_vector_metadata(
    *,
    kb_id: int,
    doc_id: int,
    doc_name: str,
    ext: str,
    chunk: Chunk,
) -> dict[str, Any]:
    """向量库里的 metadata。

    **不放正文的父块**（`parent_content` 可能有 1800 字）：向量库是索引，
    不该承担存储职责。检索命中后由 `ChunkEnricher` 回 MySQL 取权威正文。

    `chunk_id` 也放进来，但只作为"提示"——真正的 `chunk_id` 以 MySQL 为准
    （插入顺序决定，向量库写入时可能还没有 id）。
    """
    return {
        "kb_id": int(kb_id),
        "doc_id": int(doc_id),
        "doc_name": doc_name,
        "chunk_index": int(chunk.index),
        "page_no": chunk.page_no,
        "section_path": chunk.section_path,
        "ext": ext,
        "char_count": chunk.char_count,
    }


def to_bm25_payload(
    *,
    kb_id: int,
    doc_id: int,
    doc_name: str,
    ext: str,
    chunk: Chunk,
    vector_id: str,
    chunk_id: int | None = None,
) -> dict[str, Any]:
    """BM25 索引里的 payload（含父块，因为它不落盘、只占内存）。

    **`vector_id` 是必须的**：`BM25Registry.add_chunks` 用它做倒排索引的 doc_key，
    检索命中后也靠它回 MySQL 查权威正文。漏了它会在入库时报 `KeyError: 'vector_id'`——
    而这恰好是本项目"写入顺序"设计的价值体现：向量库已写、MySQL 事务回滚，
    结果是一批**孤儿向量**（`retrieval` 的 `dropped_orphans` 会如实报出来），
    而不是"列表显示成功但搜不到"。
    """
    return {
        "vector_id": vector_id,
        "kb_id": int(kb_id),
        "doc_id": int(doc_id),
        "doc_name": doc_name,
        "chunk_id": chunk_id,
        "chunk_index": int(chunk.index),
        "content": chunk.content,
        "parent_content": chunk.parent_content,
        "page_no": chunk.page_no,
        "section_path": chunk.section_path,
        "ext": ext,
    }


__all__ = [
    "PreparedChunks",
    "build_vector_metadata",
    "embed_chunks",
    "prepare_chunks",
    "to_bm25_payload",
]
