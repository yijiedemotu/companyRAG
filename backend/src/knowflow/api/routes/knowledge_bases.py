"""知识库路由（契约 5.2）。

**一句话定位**：知识库的 CRUD 与一致性自检接口。

**在链路中的位置**：前端 `/kbs` 页面 -> **本模块** -> `KBService` -> `knowledge_bases` 表 +
向量库集合 + BM25 索引。

**关键设计取舍**：

1. **读接口不校验归属，写接口必须校验**。这是刻意的：
   "能看见哪些 KB"与"能改哪些 KB"是两个问题。本项目是内部工具形态，
   列表里能看到别人的 KB（协作场景需要），但改名/删除/上传只允许所有者或管理员
   （`KBService.ensure_owner`）——越权写才是真正会造成损失的。
2. **`DELETE` 回 204 而不是 200**（契约 5.2）。204 不能有响应体，
   所以这里返回 `None` 且不设 `response_model`；失败信息走统一错误信封。
3. **创建/更新后手动 `commit()`**：`KBService` 只 `flush()`（它不知道自己的调用方
   是 HTTP 请求还是脚本）。事务边界由调用方决定，路由就是那个调用方。
4. **`KBOut` 的 `doc_count` / `chunk_count` 由 service 聚合**，路由不做二次查询：
   列表页 20 个 KB 各查一次就是 40 次往返（N+1）。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status

from knowflow.api.deps import CurrentUser, Services
from knowflow.db.models.identity import User
from knowflow.db.models.knowledge import KnowledgeBase
from knowflow.schemas.common import Page, PageParams
from knowflow.schemas.knowledge_base import (
    KBCreateRequest,
    KBOut,
    KBStatsOut,
    KBUpdateRequest,
)

__all__ = ["router"]

router = APIRouter(prefix="/kbs", tags=["知识库"])

# **必须写 `Annotated[PageParams, Depends()]`**：写成 `Depends(PageParams)`
# FastAPI 会把 page/size 推导成一个 JSON body 参数（实测确认，见 schemas/common.py）。
# `Depends()` 不带参数才对——它让 FastAPI 去调用 `PageParams.__init__` 并解析其 Query 默认值。
PageDep = Annotated[PageParams, Depends()]

KeywordQuery = Annotated[str | None, Query(max_length=128, description="按名称/描述模糊搜索")]


def _require_owner(kb_id: int, services: Services, user: User) -> KnowledgeBase:
    """取 KB 并校验"当前用户能否改动它"。不存在 -> 404，非本人且非管理员 -> 403。"""
    kb = services.kb_service.get(kb_id)
    services.kb_service.ensure_owner(kb, user.id, is_admin=user.is_admin)
    return kb


@router.get("", response_model=Page[KBOut], summary="知识库列表")
def list_kbs(params: PageDep, services: Services, keyword: KeywordQuery = None) -> Page[KBOut]:
    items, total = services.kb_service.list_kbs(page=params.page, size=params.size, keyword=keyword)
    return Page[KBOut].create(
        items=[KBOut.model_validate(item) for item in items],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED, summary="创建知识库")
def create_kb(payload: KBCreateRequest, services: Services, user: CurrentUser) -> KBOut:
    """创建 KB。**向量化信息在这里被快照下来**：换模型 = 换维度 = 旧向量全废。"""
    kb = services.kb_service.create(
        name=payload.name,
        description=payload.description,
        owner_id=user.id,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    services.session.commit()
    return KBOut.model_validate(services.kb_service.get_with_counts(kb.id))


@router.get("/{kb_id}", response_model=KBOut, summary="知识库详情")
def get_kb(kb_id: int, services: Services) -> KBOut:
    return KBOut.model_validate(services.kb_service.get_with_counts(kb_id))


@router.patch("/{kb_id}", response_model=KBOut, summary="更新知识库")
def update_kb(kb_id: int, payload: KBUpdateRequest, services: Services, user: CurrentUser) -> KBOut:
    """PATCH 语义：只改传了的字段。

    `exclude_unset=True` 是**必须**的（见 `KBUpdateRequest` 的说明）：
    不排除未设置字段的话，前端只改名字会把 `chunk_size` 一起重置成默认值，
    用户看到的是"名称改了，切片参数也变了"这种静默数据损坏。
    """
    _require_owner(kb_id, services, user)
    fields: dict[str, Any] = payload.model_dump(exclude_unset=True)
    services.kb_service.update(kb_id, **fields)
    services.session.commit()
    return KBOut.model_validate(services.kb_service.get_with_counts(kb_id))


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除知识库")
def delete_kb(kb_id: int, services: Services, user: CurrentUser) -> None:
    """软删 KB + 清向量集合 + 清 BM25（顺序由 service 保证：先清外部索引再软删）。"""
    _require_owner(kb_id, services, user)
    services.kb_service.delete(kb_id)
    services.session.commit()


@router.get("/{kb_id}/stats", response_model=KBStatsOut, summary="知识库一致性自检")
def kb_stats(kb_id: int, services: Services) -> KBStatsOut:
    """`consistent=False` 说明 MySQL 的 chunk 数与向量库/BM25 不一致。

    前端要为它画红点：**把静默降级变可见**，否则"检索莫名少召回"会变成悬案。
    """
    stats = services.kb_service.stats(kb_id)
    # service 额外带了 status_counts / embedding_* 等字段（给 Agent 工具与 /health 用），
    # 这里用 `model_validate` 只取契约 5.2 声明的字段，多余字段自动丢弃。
    return KBStatsOut.model_validate(stats)
