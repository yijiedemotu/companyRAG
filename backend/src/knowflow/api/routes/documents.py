"""文档路由（契约 5.3）：上传、列表、详情、切片预览、删除。

**一句话定位**：把"一份文件"变成"可检索的切片"的入口。

**在链路中的位置**：前端文档管理页 -> **本模块** -> `DocumentService` -> 解析/切分/向量化/BM25。

**关键设计取舍**：

1. **上传路由是同步 `def`**。入库是同步阻塞的重活（解析 + 本地模型推理 + 向量库写入），
   写成 `async def` 会**卡死事件循环**：一个 20MB 的 PDF 能把整个服务卡住几十秒。
   同步 `def` 会被 Starlette 丢进线程池，慢的是那一个请求，不是所有人。
2. **先校验扩展名与体积，再读文件**。如果先 `read()` 再判断大小，
   50MB 已经进内存了；这里先用扩展名白名单挡一道、再用 `UploadFile.size` 预判，
   读完后仍以真实长度为准（`UploadFile.size` 来自 multipart 声明，不可全信）。
   两道闸的顺序有意为之：**白名单在前**（零成本），体积判断在后。
3. **契约要求同步返回（201 `DocumentOut`）**，所以不做后台任务：
   用户点上传后要立刻看到"这份文件现在是什么状态"。解析失败也是一条
   `FAILED` 记录（service 的两段式事务保证），前端刷新就能看到原因。
4. **删除是 204**；`DocumentService.delete_document` 返回的统计（清了多少向量/chunk）
   只写日志，不进响应体——契约没给它留位置，硬塞进去会让前端类型对不上。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, UploadFile, status

from knowflow.api.deps import CurrentUser, Services
from knowflow.core.exceptions import (
    FileTooLargeError,
    UnsupportedFileTypeError,
    ValidationError,
)
from knowflow.core.logging import get_logger
from knowflow.db.models.knowledge import Document
from knowflow.ingest.loaders import detect_ext
from knowflow.schemas.common import Page, PageParams
from knowflow.schemas.document import ChunkOut, DocumentOut, DocumentStatus

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(tags=["文档"])

# 同 `knowledge_bases.py`：必须是 `Depends()`（不带参数），否则 page/size 会被当成 body。
PageDep = Annotated[PageParams, Depends()]

#: 状态过滤参数的取值直接复用 `schemas.document.DocumentStatus` 字面量
#: （它已用 import 期断言与 ORM 状态常量钉死，不会漂移）。
StatusQuery = Annotated[
    DocumentStatus | None,
    Query(description="按入库状态过滤：UPLOADED/PARSING/CHUNKING/EMBEDDING/READY/FAILED"),
]


def _assert_upload_doc_owner(
    doc: Document, services: Services, user_id: int, is_admin: bool
) -> None:
    """文档权限 = 所属 KB 的权限。**不复制一套权限逻辑**：
    直接复用 `ensure_owner`，避免"KB 能改、文档不能改"这类不一致。"""
    kb = services.kb_service.get(doc.kb_id)
    services.kb_service.ensure_owner(kb, user_id, is_admin=is_admin)


@router.post(
    "/kbs/{kb_id}/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档（multipart）",
)
def upload_document(
    kb_id: int,
    services: Services,
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="要上传的文件，字段名固定为 file")],
) -> DocumentOut:
    """上传 -> 解析 -> 切分 -> 向量化 -> READY（同步返回最终状态）。

    失败时给的是带错误码的响应 + 库里一条 `FAILED` 记录
    （`error_code` / `error_message` 让用户知道为什么失败），而不是一片模糊的 500。
    """
    kb = services.kb_service.get(kb_id)
    services.kb_service.ensure_owner(kb, user.id, is_admin=user.is_admin)

    filename = (file.filename or "").strip()
    if not filename:
        raise ValidationError("上传的文件缺少 filename（multipart 的 filename 字段为空）")

    settings = services.document_service.settings
    # ---- 第一道闸：扩展名（不读文件，几乎零成本）----
    ext = detect_ext(filename)
    allowed = settings.allowed_extensions
    if ext not in allowed:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型 .{ext}，允许的类型：{', '.join(allowed)}",
            detail={"ext": ext, "allowed": allowed},
        )

    # ---- 第二道闸：体积预判（`UploadFile.size` 由 Starlette 从 multipart 头填，可能为 None）----
    declared = getattr(file, "size", None)
    if declared is not None and declared > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"文件大小 {declared / 1024 / 1024:.1f}MB 超过上限 {settings.max_upload_mb}MB",
            detail={"size_bytes": int(declared), "max_bytes": settings.max_upload_bytes},
        )

    data = file.file.read()
    # ---- 第三道闸：真实长度（声明值不可信，以实际读到的字节数为准）----
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"文件大小 {len(data) / 1024 / 1024:.1f}MB 超过上限 {settings.max_upload_mb}MB",
            detail={"size_bytes": len(data), "max_bytes": settings.max_upload_bytes},
        )
    if not data:
        raise ValidationError("上传的文件为空")

    doc = services.document_service.ingest_upload(kb_id=kb_id, filename=filename, data=data)
    return DocumentOut.model_validate(doc)


@router.get(
    "/kbs/{kb_id}/documents",
    response_model=Page[DocumentOut],
    summary="文档列表",
)
def list_documents(
    kb_id: int, params: PageDep, services: Services, status_filter: StatusQuery = None
) -> Page[DocumentOut]:
    services.kb_service.get(kb_id)  # KB 不存在 -> 404（而不是"空列表"这种误导性结果）
    rows, total = services.document_service.list_documents(
        kb_id=kb_id, page=params.page, size=params.size, status=status_filter
    )
    return Page[DocumentOut].create(
        items=[DocumentOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.get("/documents/{doc_id}", response_model=DocumentOut, summary="文档详情")
def get_document(doc_id: int, services: Services) -> DocumentOut:
    return DocumentOut.model_validate(services.document_service.get_document(doc_id))


@router.get(
    "/documents/{doc_id}/chunks",
    response_model=Page[ChunkOut],
    summary="切片预览",
)
def list_chunks(doc_id: int, params: PageDep, services: Services) -> Page[ChunkOut]:
    rows, total = services.document_service.list_chunks(
        doc_id=doc_id, page=params.page, size=params.size
    )
    return Page[ChunkOut].create(
        items=[ChunkOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除文档")
def delete_document(doc_id: int, services: Services, user: CurrentUser) -> None:
    """软删文档 + 清向量 + 清 chunks + 重建该 KB 的 BM25。"""
    doc = services.document_service.get_document(doc_id)
    _assert_upload_doc_owner(doc, services, user.id, user.is_admin)
    result = services.document_service.delete_document(doc_id)
    logger.info(
        "api.document_deleted",
        doc_id=doc_id,
        removed_vectors=result.get("removed_vectors"),
        removed_chunks=result.get("removed_chunks"),
        user_id=user.id,
    )
