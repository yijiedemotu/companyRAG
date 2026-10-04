"""知识库服务：CRUD、统计、BM25 索引重建。

`rbuild_bm25_index()` 是这个文件里最重要的方法 —— 它是"静默降级"的唯一解药。

BM25 索引在内存里，进程一退出就没了。如果不重建：
服务照常启动、所有接口返回 200、向量检索照常工作，
**只是"PAYLOAD_TOO_LARGE 这种精确关键词"查不到了**。
用户会觉得"时好时坏"，而日志里什么都没有 —— 这是最难排查的一类 bug。

所以约定：**服务启动时必须全量重建一次**，`/health` 里必须报出
`bm25_doc_count` 与 `chunk_count` 是否一致（`consistent` 字段）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import (
    ForbiddenError,
    KBNameConflictError,
    KBNotFoundError,
    ValidationError,
)
from knowflow.core.logging import get_logger
from knowflow.db.models.knowledge import (
    DOC_STATUS_READY,
    Chunk,
    Document,
    KnowledgeBase,
)
from knowflow.embeddings.base import Embedder
from knowflow.retrieval.bm25 import BM25Registry
from knowflow.services.chunks import iter_kb_chunks
from knowflow.vectorstore.base import VectorStore

logger = get_logger(__name__)


class KBService:
    def __init__(
        self,
        *,
        session: Session,
        vector_store: VectorStore,
        bm25: BM25Registry,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ 读
    def list_kbs(
        self,
        *,
        page: int = 1,
        size: int = 20,
        keyword: str | None = None,
        owner_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """返回 `(KB 列表(含计数), 总数)`。

        计数用**两个分组聚合**一次取回，而不是对每个 KB 各查一次：
        列表页 20 个 KB，N+1 查询就是 40 次往返。
        """
        conditions: list[ColumnElement[bool]] = [KnowledgeBase.deleted_at.is_(None)]
        if keyword:
            like = f"%{keyword.strip()}%"
            conditions.append(
                or_(KnowledgeBase.name.like(like), KnowledgeBase.description.like(like))
            )
        if owner_id is not None:
            conditions.append(KnowledgeBase.owner_id == owner_id)

        total = int(
            self.session.execute(select(func.count(KnowledgeBase.id)).where(*conditions)).scalar()
            or 0
        )
        if total == 0:
            return [], 0

        rows = list(
            self.session.execute(
                select(KnowledgeBase)
                .where(*conditions)
                .order_by(KnowledgeBase.id.desc())
                .offset((page - 1) * size)
                .limit(size)
            ).scalars()
        )
        if not rows:
            return [], total

        kb_ids = [kb.id for kb in rows]
        doc_counts = self._doc_counts(kb_ids)
        chunk_counts = self._chunk_counts(kb_ids)

        items: list[dict[str, Any]] = []
        for kb in rows:
            item = self._to_dict(kb)
            item["doc_count"] = doc_counts.get(kb.id, 0)
            item["chunk_count"] = chunk_counts.get(kb.id, 0)
            items.append(item)
        return items, total

    def _doc_counts(self, kb_ids: Sequence[int]) -> dict[int, int]:
        stmt = (
            select(Document.kb_id, func.count(Document.id))
            .where(Document.kb_id.in_(list(kb_ids)))
            .where(Document.deleted_at.is_(None))
            .group_by(Document.kb_id)
        )
        return {int(kb): int(count) for kb, count in self.session.execute(stmt).all()}

    def _chunk_counts(self, kb_ids: Sequence[int]) -> dict[int, int]:
        stmt = (
            select(Chunk.kb_id, func.count(Chunk.id))
            .where(Chunk.kb_id.in_(list(kb_ids)))
            .group_by(Chunk.kb_id)
        )
        return {int(kb): int(count) for kb, count in self.session.execute(stmt).all()}

    def get(self, kb_id: int) -> KnowledgeBase:
        kb = self.session.get(KnowledgeBase, kb_id)
        if kb is None or kb.deleted_at is not None:
            raise KBNotFoundError(f"知识库 {kb_id} 不存在")
        return kb

    def get_with_counts(self, kb_id: int) -> dict[str, Any]:
        kb = self.get(kb_id)
        item = self._to_dict(kb)
        item["doc_count"] = self._doc_counts([kb_id]).get(kb_id, 0)
        item["chunk_count"] = self._chunk_counts([kb_id]).get(kb_id, 0)
        return item

    def active_kb_ids(self) -> list[int]:
        stmt = (
            select(KnowledgeBase.id)
            .where(KnowledgeBase.deleted_at.is_(None))
            .where(KnowledgeBase.is_active.is_(True))
            .order_by(KnowledgeBase.id)
        )
        return [int(row) for row in self.session.execute(stmt).scalars()]

    def stats(self, kb_id: int) -> dict[str, Any]:
        """`/kbs/{id}/stats`：**核心是 `consistent`**。

        `chunk_count != vector_count` 意味着向量库与 MySQL 不一致
        （写入中断、手动删过库、索引重建失败）。把这个布尔量暴露到前端，
        是为了让"静默不一致"变成"界面上一个红点"。
        """
        kb = self.get(kb_id)
        doc_rows = self.session.execute(
            select(Document.status, func.count(Document.id))
            .where(Document.kb_id == kb_id)
            .where(Document.deleted_at.is_(None))
            .group_by(Document.status)
        ).all()
        status_counts = {str(status): int(count) for status, count in doc_rows}

        chunk_count = self._chunk_counts([kb_id]).get(kb_id, 0)
        totals = self.session.execute(
            select(
                func.coalesce(func.sum(Document.char_count), 0),
                func.coalesce(func.sum(Document.token_count), 0),
            )
            .where(Document.kb_id == kb_id)
            .where(Document.deleted_at.is_(None))
        ).one()

        vector_count = self.vector_store.count(kb_id=kb_id)
        bm25_count = self.bm25.doc_count_of(kb_id)

        return {
            "kb_id": kb_id,
            "doc_count": sum(status_counts.values()),
            "ready_doc_count": status_counts.get(DOC_STATUS_READY, 0),
            "status_counts": status_counts,
            "chunk_count": chunk_count,
            "vector_count": vector_count,
            "bm25_doc_count": bm25_count,
            "total_chars": int(totals[0] or 0),
            "total_tokens": int(totals[1] or 0),
            # 三者一致才算健康。任何一个不等都要让运维看见。
            "consistent": chunk_count == vector_count == bm25_count,
            "embedding_provider": kb.embedding_provider,
            "embedding_model": kb.embedding_model,
            "embedding_dim": kb.embedding_dim,
        }

    # ------------------------------------------------------------------ 写
    def create(
        self,
        *,
        name: str,
        description: str | None,
        owner_id: int,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> KnowledgeBase:
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError("知识库名称不能为空")

        size = chunk_size or self.settings.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap
        if overlap >= size:
            raise ValidationError(
                f"chunk_overlap({overlap}) 必须小于 chunk_size({size})",
                detail={"chunk_size": size, "chunk_overlap": overlap},
            )

        existing = self.session.execute(
            select(KnowledgeBase.id)
            .where(KnowledgeBase.name == clean_name)
            .where(KnowledgeBase.deleted_at.is_(None))
        ).scalar()
        if existing is not None:
            raise KBNameConflictError(f"知识库 {clean_name!r} 已存在")

        kb = KnowledgeBase(
            name=clean_name,
            description=description,
            owner_id=owner_id,
            # 快照当前的向量化能力：换模型 = 换维度 = 旧向量全废，必须从源头拦住
            embedding_provider=self.embedder.provider,
            embedding_model=self.embedder.model,
            embedding_dim=self.embedder.dim,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        self.session.add(kb)
        try:
            self.session.flush()
        except IntegrityError as exc:  # 并发建同名 KB 时由唯一索引兜底
            self.session.rollback()
            raise KBNameConflictError(f"知识库 {clean_name!r} 已存在") from exc
        logger.info("kb.created", kb_id=kb.id, name=kb.name, owner_id=owner_id)
        return kb

    def update(self, kb_id: int, **fields: Any) -> KnowledgeBase:
        kb = self.get(kb_id)
        name = fields.get("name")
        if isinstance(name, str):
            clean = name.strip()
            if not clean:
                raise ValidationError("知识库名称不能为空")
            if clean != kb.name:
                conflict = self.session.execute(
                    select(KnowledgeBase.id)
                    .where(KnowledgeBase.name == clean)
                    .where(KnowledgeBase.id != kb_id)
                    .where(KnowledgeBase.deleted_at.is_(None))
                ).scalar()
                if conflict is not None:
                    raise KBNameConflictError(f"知识库 {clean!r} 已存在")
                kb.name = clean

        size = fields.get("chunk_size", kb.chunk_size)
        overlap = fields.get("chunk_overlap", kb.chunk_overlap)
        if overlap >= size:
            raise ValidationError(
                f"chunk_overlap({overlap}) 必须小于 chunk_size({size})",
                detail={"chunk_size": size, "chunk_overlap": overlap},
            )

        for key in ("description", "is_active", "chunk_size", "chunk_overlap"):
            if key in fields and fields[key] is not None:
                setattr(kb, key, fields[key])
        self.session.flush()
        logger.info("kb.updated", kb_id=kb_id, fields=sorted(fields.keys()))
        return kb

    def delete(self, kb_id: int) -> None:
        """软删 KB + 清向量集合 + 清 BM25 索引。

        **顺序**：先清索引（不可逆的副作用），最后软删 DB 行。
        反过来的话，如果清索引失败了，DB 里已经看不到这个 KB，
        它的向量却还在库里 —— 一堆查不到也删不掉的垃圾。
        """
        kb = self.get(kb_id)
        try:
            self.vector_store.reset(kb_id=kb_id)
        except Exception as exc:  # noqa: BLE001
            # 向量库清不掉也要继续：DB 软删更重要（用户视角"删掉了"），
            # 残留向量由 `KBService.stats() 的 consistent 字段 + scripts/smoke_pipeline.py 的一致性断言` 报告并清理。
            logger.warning("kb.vector_reset_failed", kb_id=kb_id, error=str(exc)[:200])
        self.bm25.drop_kb(kb_id)
        kb.mark_deleted()
        kb.is_active = False
        self.session.flush()
        logger.info("kb.deleted", kb_id=kb_id, name=kb.name)

    def ensure_owner(self, kb: KnowledgeBase, user_id: int, *, is_admin: bool) -> None:
        """权限检查：只有所有者或管理员能改动。

        抽成方法而不是散落在路由里，是为了**避免漏检**：
        任何一个写接口忘了调它就是一个越权漏洞。
        """
        if kb.owner_id != user_id and not is_admin:
            raise ForbiddenError("只有知识库创建者或管理员可以执行该操作")

    # ------------------------------------------------------------------ 索引
    def rebuild_bm25_index(self, *, kb_ids: Sequence[int] | None = None) -> int:
        """从 MySQL 全量重建 BM25 索引。返回重建的切片总数。

        **必须在服务启动时调用**，否则内存索引为空 → 关键词检索静默失效。
        分批读取，内存占用与库大小无关。
        """
        targets = list(kb_ids) if kb_ids else self.active_kb_ids()
        total = 0
        for kb_id in targets:
            self.bm25.drop_kb(kb_id)
            for batch in iter_kb_chunks(self.session, kb_id=kb_id):
                total += self.bm25.add_chunks(kb_id, batch)
            logger.info("bm25.rebuilt", kb_id=kb_id, docs=self.bm25.doc_count_of(kb_id))
        if targets:
            logger.info("bm25.rebuild_done", kbs=len(targets), chunks=total)
        return total

    def rebuild_bm25_for_kb(self, kb_id: int) -> int:
        return self.rebuild_bm25_index(kb_ids=[kb_id])

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _to_dict(kb: KnowledgeBase) -> dict[str, Any]:
        return {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "owner_id": kb.owner_id,
            "embedding_provider": kb.embedding_provider,
            "embedding_model": kb.embedding_model,
            "embedding_dim": kb.embedding_dim,
            "chunk_size": kb.chunk_size,
            "chunk_overlap": kb.chunk_overlap,
            "is_active": kb.is_active,
            "created_at": kb.created_at,
            "updated_at": kb.updated_at,
        }


__all__ = ["KBService"]
