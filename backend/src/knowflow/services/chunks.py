"""切片访问助手 + `ChunkEnricher` 的 SQL 实现。

**为什么需要 `ChunkEnricher` 这一层**：检索层（`retrieval/`）不该 import
SQLAlchemy —— 一旦 import，它的单测就必须起数据库，跑得慢且脆。
所以检索层定义一个 Protocol，由这里实现并注入。这是**依赖倒置**的一个实际收益，
不是为了架构图好看。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from knowflow.db.models.knowledge import Chunk, Document
from knowflow.retrieval.types import ChunkInfo


def fetch_chunk_infos(session: Session, vector_ids: Sequence[str]) -> dict[str, ChunkInfo]:
    """按 `vector_id` 批量取 chunk 的权威信息（含父块与出处）。

    一次 `IN` 查询而不是 N 次单查：检索一次可能命中 20 条，
    N+1 查询会让检索多花几十毫秒（都是数据库往返）。
    """
    if not vector_ids:
        return {}

    stmt = (
        select(Chunk, Document.filename, Document.ext)
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.vector_id.in_(list(vector_ids)))
        .where(Document.deleted_at.is_(None))
    )

    infos: dict[str, ChunkInfo] = {}
    for chunk, filename, ext in session.execute(stmt).all():
        infos[chunk.vector_id] = ChunkInfo(
            vector_id=chunk.vector_id,
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            doc_name=filename,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            context_text=chunk.parent_content or chunk.content,
            page_no=chunk.page_no,
            section_path=chunk.section_path,
            ext=ext,
        )
    return infos


class SqlChunkEnricher:
    """实现 `retrieval.types.ChunkEnricher`。

    注意它**持有一个 session**：只在一次请求的生命周期内使用，
    session 由 FastAPI 依赖管理（`get_db`）。
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def fetch(self, vector_ids: Sequence[str]) -> dict[str, ChunkInfo]:
        return fetch_chunk_infos(self.session, vector_ids)


def iter_kb_chunks(
    session: Session, *, kb_id: int, batch_size: int = 2000
) -> Iterator[list[dict[str, object]]]:
    """按批流式读取某个 KB 的全部切片（用于重建 BM25 索引）。

    **为什么分批**：一个 KB 可能有几十万 chunk，一次性 `all()` 会把
    几 GB 对象堆在内存里。分批 + 生成器让内存占用与库大小无关。
    用 `yield_per` 让 SQLAlchemy 走服务端游标（MySQL 的 SSCursor）。
    """
    stmt = (
        select(Chunk, Document.filename, Document.ext)
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.kb_id == kb_id)
        .where(Document.deleted_at.is_(None))
        .order_by(Chunk.doc_id, Chunk.chunk_index)
    )

    batch: list[dict[str, object]] = []
    for chunk, filename, ext in session.execute(stmt).yield_per(batch_size):
        batch.append(
            {
                "vector_id": chunk.vector_id,
                "doc_id": chunk.doc_id,
                "doc_name": filename,
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "parent_content": chunk.parent_content or chunk.content,
                "page_no": chunk.page_no,
                "section_path": chunk.section_path,
                "ext": ext,
            }
        )
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_chunks(session: Session, *, kb_id: int) -> int:
    stmt = select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id)
    return int(session.execute(stmt).scalar() or 0)


def count_chunks_by_doc(session: Session, *, kb_id: int) -> dict[int, int]:
    """`{doc_id: chunk 数}`，用于知识库列表页一次性把计数都拿到（避免 N+1）。"""
    stmt = (
        select(Chunk.doc_id, func.count(Chunk.id))
        .where(Chunk.kb_id == kb_id)
        .group_by(Chunk.doc_id)
    )
    return {int(doc_id): int(count) for doc_id, count in session.execute(stmt).all()}


__all__ = [
    "SqlChunkEnricher",
    "count_chunks",
    "count_chunks_by_doc",
    "fetch_chunk_infos",
    "iter_kb_chunks",
]
