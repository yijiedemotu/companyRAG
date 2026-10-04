"""文档服务：上传入库、列表、删除、切片查看。

**入库分两个事务，这是刻意的**：

    事务 1：保存原文件 + 建 documents 行（status=UPLOADED）→ 立即 commit
    事务 2：解析 → 切分 → 向量化 → 写向量库 → 写 chunks → 写 BM25 → 置 READY

为什么要拆？因为 `vector_id = "{doc_id}:{index}"` 需要先有 `doc_id`。
更重要的是**失败可见性**：解析失败时，事务 1 已经落地，
用户可以刷新列表看到那条 `FAILED` 记录和具体错误原因；
如果全在一个事务里回滚，用户只会看到一个"上传失败"的 alert，
既不知道为什么，也看不到自己传过什么。

**事务 2 内部的写入顺序也是刻意的**（见 `ingest/pipeline.py` 的说明）：
向量库 → chunks → BM25 → 注册表置 READY。中途失败最坏是孤儿向量（可清理），
而不是"显示成功但搜不到"。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy import ColumnElement, delete, func, select
from sqlalchemy.orm import Session

from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import (
    DocumentDuplicateError,
    DocumentNotFoundError,
    FileTooLargeError,
    KnowFlowError,
    UnsupportedFileTypeError,
    ValidationError,
)
from knowflow.core.logging import get_logger
from knowflow.db.models.knowledge import (
    DOC_STATUS_CHUNKING,
    DOC_STATUS_EMBEDDING,
    DOC_STATUS_FAILED,
    DOC_STATUS_PARSING,
    DOC_STATUS_READY,
    DOC_STATUS_UPLOADED,
    Chunk,
    Document,
    make_vector_id,
)
from knowflow.embeddings.base import Embedder
from knowflow.ingest.loaders import detect_ext, parse_bytes
from knowflow.ingest.pipeline import (
    build_vector_metadata,
    embed_chunks,
    prepare_chunks,
    to_bm25_payload,
)
from knowflow.retrieval.bm25 import BM25Registry
from knowflow.services.knowledge_base import KBService
from knowflow.vectorstore.base import VectorItem, VectorStore

logger = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        *,
        session: Session,
        kb_service: KBService,
        vector_store: VectorStore,
        bm25: BM25Registry,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.kb_service = kb_service
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ 校验
    def validate_upload(self, *, filename: str, data: bytes) -> str:
        """返回规范化后的扩展名。校验顺序：白名单 → 体积 → 空文件。"""
        ext = detect_ext(filename)
        allowed = self.settings.allowed_extensions
        if ext not in allowed:
            raise UnsupportedFileTypeError(
                f"不支持的文件类型 .{ext}，允许的类型：{', '.join(allowed)}",
                detail={"ext": ext, "allowed": allowed},
            )
        size = len(data)
        if size > self.settings.max_upload_bytes:
            raise FileTooLargeError(
                f"文件大小 {size / 1024 / 1024:.1f}MB 超过上限 {self.settings.max_upload_mb}MB",
                detail={"size_bytes": size, "max_bytes": self.settings.max_upload_bytes},
            )
        if size == 0:
            raise ValidationError("上传的文件为空")
        return ext

    def _check_duplicate(self, *, kb_id: int, sha256: str) -> None:
        existing = self.session.execute(
            select(Document.id)
            .where(Document.kb_id == kb_id)
            .where(Document.sha256 == sha256)
            .where(Document.deleted_at.is_(None))
        ).scalar()
        if existing is not None:
            raise DocumentDuplicateError(
                "该文档内容已存在于这个知识库中（sha256 相同）",
                detail={"doc_id": int(existing), "sha256": sha256},
            )

    def _store_file(self, *, data: bytes, sha256: str, ext: str) -> Path:
        """落盘到 `uploads/<sha256 前 2 位>/<sha256>.<ext>`。

        用内容哈希命名 + 二级目录：内容相同天然去重，且不会出现
        "一个目录里躺着 10 万个文件"（Windows 上这会显著拖慢目录遍历）。
        """
        sub = self.settings.upload_dir / sha256[:2]
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{sha256}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        return path

    # ------------------------------------------------------------------ 入库
    def ingest_upload(self, *, kb_id: int, filename: str, data: bytes) -> Document:
        """上传 → 入库。返回处于 READY（或 FAILED）状态的 Document。"""
        # 只做存在性校验（不存在则 404），返回值这里用不到
        self.kb_service.get(kb_id)
        ext = self.validate_upload(filename=filename, data=data)
        sha256 = hashlib.sha256(data).hexdigest()
        self._check_duplicate(kb_id=kb_id, sha256=sha256)

        # ---- 事务 1：原文件 + 文档行（让失败可见）----
        path = self._store_file(data=data, sha256=sha256, ext=ext)
        doc = Document(
            kb_id=kb_id,
            filename=filename,
            ext=ext,
            size_bytes=len(data),
            sha256=sha256,
            storage_path=str(path.relative_to(self.settings.data_dir)).replace("\\", "/"),
            status=DOC_STATUS_UPLOADED,
        )
        self.session.add(doc)
        self.session.commit()
        doc_id = doc.id
        logger.info("document.uploaded", doc_id=doc_id, kb_id=kb_id, ext=ext, bytes=len(data))

        # ---- 事务 2：解析 → 切分 → 向量化 → 写三处 ----
        try:
            self._process(doc=doc, data=data, kb_id=kb_id, filename=filename, ext=ext)
        except KnowFlowError as exc:
            self._mark_failed(doc_id=doc_id, code=exc.code, message=exc.message)
            raise
        except Exception as exc:  # noqa: BLE001
            self._mark_failed(
                doc_id=doc_id, code="INTERNAL_ERROR", message=f"{type(exc).__name__}: {exc}"
            )
            raise

        self.session.refresh(doc)
        logger.info(
            "document.ready",
            doc_id=doc_id,
            chunks=doc.chunk_count,
            tokens=doc.token_count,
            ingest_ms=doc.ingest_ms,
        )
        return doc

    def _process(self, *, doc: Document, data: bytes, kb_id: int, filename: str, ext: str) -> None:
        import time

        started = time.perf_counter()

        # ---- 解析 ----
        self._set_status(doc, DOC_STATUS_PARSING)
        parsed = parse_bytes(data, filename, allowed_extensions=self.settings.allowed_extensions)
        doc.parser = parsed.parser
        doc.page_count = parsed.page_count
        doc.char_count = parsed.char_count
        self.session.flush()

        # ---- 切分 ----
        self._set_status(doc, DOC_STATUS_CHUNKING)
        prepared = prepare_chunks(parsed, settings=self.settings)
        if prepared.count == 0:
            raise ValidationError(
                "文档切分后没有产生任何切片（内容可能全是空白或符号）",
                detail={"char_count": parsed.char_count},
            )
        self.session.flush()

        # ---- 向量化 ----
        self._set_status(doc, DOC_STATUS_EMBEDDING)
        vectors, embed_ms = embed_chunks(
            prepared.chunks, embedder=self.embedder, settings=self.settings
        )

        # ---- 写 MySQL 的 chunk 行（先拿到 chunk_id，引用要用）----
        chunk_rows: list[Chunk] = []
        for position, chunk in enumerate(prepared.chunks):
            row = Chunk(
                kb_id=kb_id,
                doc_id=doc.id,
                chunk_index=chunk.index,
                content=chunk.content,
                parent_content=chunk.parent_content,
                char_count=chunk.char_count,
                token_count=chunk.token_count,
                page_no=chunk.page_no,
                section_path=chunk.section_path,
                vector_id=make_vector_id(doc.id, chunk.index),
            )
            chunk_rows.append(row)
            self.session.add(row)
        self.session.flush()
        logger.debug(
            "document.chunks_written",
            doc_id=doc.id,
            chunks=len(chunk_rows),
            embed_ms=embed_ms,
            split_ms=prepared.split_ms,
        )

        # ---- 写向量库 ----
        items = [
            VectorItem(
                id=row.vector_id,
                vector=vectors[position],
                content=row.content,
                metadata={
                    **build_vector_metadata(
                        kb_id=kb_id,
                        doc_id=doc.id,
                        doc_name=filename,
                        ext=ext,
                        chunk=prepared.chunks[position],
                    ),
                    "chunk_id": row.id,
                },
            )
            for position, row in enumerate(chunk_rows)
        ]
        written = self.vector_store.upsert(kb_id=kb_id, items=items)
        if written != len(items):
            # 不静默放过：向量少写了会让"检索不到"变成悬案
            logger.warning(
                "document.vector_upsert_mismatch",
                doc_id=doc.id,
                expected=len(items),
                written=written,
            )

        # ---- 写 BM25 ----
        self.bm25.add_chunks(
            kb_id,
            [
                to_bm25_payload(
                    kb_id=kb_id,
                    doc_id=doc.id,
                    doc_name=filename,
                    ext=ext,
                    chunk=prepared.chunks[position],
                    vector_id=row.vector_id,
                    chunk_id=row.id,
                )
                for position, row in enumerate(chunk_rows)
            ],
        )

        # ---- 置 READY ----
        doc.chunk_count = len(chunk_rows)
        doc.token_count = prepared.total_tokens
        doc.char_count = prepared.total_chars
        doc.ingest_ms = int((time.perf_counter() - started) * 1000)
        doc.error_code = None
        doc.error_message = None
        self._set_status(doc, DOC_STATUS_READY)
        self.session.commit()

    def _set_status(self, doc: Document, status: str) -> None:
        doc.status = status
        self.session.flush()

    def _mark_failed(self, *, doc_id: int, code: str, message: str) -> None:
        """把失败写进库并提交。**必须用新的会话状态**：
        上层事务已经因为异常处于不可用状态，直接改当前 session 上的对象不会落库。
        """
        try:
            self.session.rollback()
            doc = self.session.get(Document, doc_id)
            if doc is None:
                return
            doc.status = DOC_STATUS_FAILED
            doc.error_code = code
            doc.error_message = message[:1000]
            self.session.commit()
        except Exception as exc:  # noqa: BLE001 - 记录失败本身失败时只能记日志
            logger.error("document.mark_failed_error", doc_id=doc_id, error=str(exc)[:200])
            self.session.rollback()

    # ------------------------------------------------------------------ 读
    def list_documents(
        self,
        *,
        kb_id: int,
        page: int = 1,
        size: int = 20,
        status: str | None = None,
    ) -> tuple[list[Document], int]:
        conditions: list[ColumnElement[bool]] = [
            Document.kb_id == kb_id,
            Document.deleted_at.is_(None),
        ]
        if status:
            conditions.append(Document.status == status)

        total = int(
            self.session.execute(select(func.count(Document.id)).where(*conditions)).scalar() or 0
        )
        if total == 0:
            return [], 0

        rows = list(
            self.session.execute(
                select(Document)
                .where(*conditions)
                .order_by(Document.id.desc())
                .offset((page - 1) * size)
                .limit(size)
            ).scalars()
        )
        return rows, total

    def get_document(self, doc_id: int) -> Document:
        doc = self.session.get(Document, doc_id)
        if doc is None or doc.deleted_at is not None:
            raise DocumentNotFoundError(f"文档 {doc_id} 不存在")
        return doc

    def list_chunks(self, *, doc_id: int, page: int = 1, size: int = 20) -> tuple[list[Chunk], int]:
        self.get_document(doc_id)
        total = int(
            self.session.execute(
                select(func.count(Chunk.id)).where(Chunk.doc_id == doc_id)
            ).scalar()
            or 0
        )
        if total == 0:
            return [], 0
        rows = list(
            self.session.execute(
                select(Chunk)
                .where(Chunk.doc_id == doc_id)
                .order_by(Chunk.chunk_index)
                .offset((page - 1) * size)
                .limit(size)
            ).scalars()
        )
        return rows, total

    def list_documents_for_tool(
        self, *, kb_id: int, keyword: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """给 Agent 的 `list_documents` 工具用（轻量，不带全文）。"""
        conditions: list[ColumnElement[bool]] = [
            Document.kb_id == kb_id,
            Document.deleted_at.is_(None),
        ]
        if keyword:
            conditions.append(Document.filename.like(f"%{keyword.strip()}%"))
        rows = list(
            self.session.execute(
                select(Document).where(*conditions).order_by(Document.id.desc()).limit(limit)
            ).scalars()
        )
        return [
            {
                "doc_id": doc.id,
                "filename": doc.filename,
                "status": doc.status,
                "chunk_count": doc.chunk_count,
            }
            for doc in rows
        ]

    # ------------------------------------------------------------------ 删
    def delete_document(self, doc_id: int) -> dict[str, Any]:
        """软删文档 + 物理清向量 + 物理清 chunks + 重建该 KB 的 BM25。

        **顺序**：先清外部索引（向量、BM25），再软删行。
        反过来的话，清索引失败就会留下"删不掉的孤儿向量"。

        BM25 直接整库重建而不是按 doc_id 精确删：
        内存索引的删除要靠遍历 payload（没有反向映射表），
        而重建是 O(n) 且行为绝对可靠 —— 删除是低频操作，选可靠的那个。
        """
        doc = self.get_document(doc_id)
        kb_id = doc.kb_id

        removed_vectors = 0
        try:
            removed_vectors = self.vector_store.delete(kb_id=kb_id, doc_id=doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("document.vector_delete_failed", doc_id=doc_id, error=str(exc)[:200])

        removed_chunks = int(
            self.session.execute(
                select(func.count(Chunk.id)).where(Chunk.doc_id == doc_id)
            ).scalar()
            or 0
        )
        # chunk 行物理删：它们已经没有引用价值，留着只会让 verify_consistency 一直报差异
        self.session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))

        doc.mark_deleted()
        doc.chunk_count = 0
        self.session.commit()

        self.kb_service.rebuild_bm25_for_kb(kb_id)

        logger.info(
            "document.deleted",
            doc_id=doc_id,
            kb_id=kb_id,
            vectors=removed_vectors,
            chunks=removed_chunks,
        )
        return {
            "doc_id": doc_id,
            "removed_vectors": removed_vectors,
            "removed_chunks": removed_chunks,
        }

    def storage_path_of(self, doc: Document) -> Path:
        return self.settings.data_dir / doc.storage_path


__all__ = ["DocumentService"]
