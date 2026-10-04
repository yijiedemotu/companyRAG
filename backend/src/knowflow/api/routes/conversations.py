"""会话与消息路由（契约 5.7）。

**一句话定位**：会话列表/详情/软删、消息回放（含引用）、反馈提交。

**在链路中的位置**：前端对话页 -> **本模块** -> `ChatService`（会话与消息读写）。

**关键设计取舍**：

1. **`GET /conversations/{id}/messages` 的响应模型是 `MessagePage`，不是
   `Page["MessageOut"]`**。字符串 ForwardRef 会被 Pydantic 用**定义 `Page` 的那个模块**
   （`schemas/common.py`，那里也有一个 `MessageOut` 别名 = 提示文案）的命名空间解析，
   于是静默变成 `Page[SimpleMessageOut]` —— 接口能跑，返回的 items 却是提示文案。
   这个坑已经被踩过一次并修好，`MessagePage` 里的 import 期断言会拦住它再复发。
2. **`citations` 缺 `doc_name` / `page_no` / `section_path`，在这里 JOIN 补齐**。
   这三列不在 `message_citations` 表上（见 `CitationOut` 的说明），
   补它们需要 `chunks` + `documents` 两次 JOIN。**放在路由而不是 service 的理由**：
   它不改变任何业务状态、只影响"响应长什么样"，属于**表示层聚合**；
   而且它必须发生在 `MessageOut` 序列化**之前**，放进 service 反而要在
   service 里返回"待补全的 dict"，把 ORM 对象和半成品混在一起。
   **一次 `IN` 查询取全部 chunk**，不做 N+1。
3. **别人的会话一律 404 而不是 403**（service 层保证）：
   返回 403 等于告诉调用方"这个 id 存在，只是不属于你"，那是一个资源枚举器。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select

from knowflow.agent.memory import create_conversation, touch_conversation_title
from knowflow.api.deps import CurrentUser, Services
from knowflow.core.exceptions import ConversationNotFoundError
from knowflow.db.models.chat import Message
from knowflow.db.models.knowledge import Chunk, Document
from knowflow.schemas.common import Page, PageParams
from knowflow.schemas.conversation import (
    CitationOut,
    ConversationCreateRequest,
    ConversationOut,
    FeedbackOut,
    FeedbackRequest,
    MessageOut,
    MessagePage,
)

__all__ = ["router"]

router = APIRouter(tags=["会话"])

PageDep = Annotated[PageParams, Depends()]


def _citation_context(
    services: Services, messages: Sequence[Message]
) -> dict[int, tuple[str | None, int | None, str | None]]:
    """一次查出所有引用切片的三元信息：`{chunk_id: (doc_name, page_no, section_path)}`。

    **为什么在这里写 SQL**（本文件是全项目两处例外之一，见 `api/__init__.py`）：
    这是纯粹的"响应形状补齐"，没有业务规则、不写库。放进 service 会迫使它返回
    `list[dict]` 这种半成品，反而破坏 service 层的类型完整性。

    为什么要 JOIN `documents` 而不是信 `message_citations.snippet`：
    文件名要以**当前**的 `documents.filename` 为准（文档改名后历史引用也该跟着改），
    而 snippet 只是展示用的截断文本，不是出处元信息。
    """
    chunk_ids: set[int] = set()
    for message in messages:
        for citation in message.citations:
            if citation.chunk_id is not None:
                chunk_ids.add(int(citation.chunk_id))
    if not chunk_ids:
        return {}

    stmt = (
        select(Chunk.id, Chunk.page_no, Chunk.section_path, Document.filename)
        .join(Document, Document.id == Chunk.doc_id)
        .where(Chunk.id.in_(sorted(chunk_ids)))
    )
    context: dict[int, tuple[str | None, int | None, str | None]] = {}
    for chunk_id, page_no, section_path, filename in services.session.execute(stmt).all():
        context[int(chunk_id)] = (filename, page_no, section_path)
    return context


def _message_out(
    message: Message, context: dict[int, tuple[str | None, int | None, str | None]]
) -> MessageOut:
    """ORM `Message` + 引用补全 -> `MessageOut`。

    逐条构造 `CitationOut` 而不是 `model_validate` 整个 ORM 对象：
    ORM 的 `MessageCitation` 上没有那三个字段，直接 validate 会得到 `None`，
    必须显式把 JOIN 结果塞进去。
    """
    citations: list[CitationOut] = []
    for citation in sorted(message.citations, key=lambda item: item.rank):
        doc_name, page_no, section_path = context.get(
            int(citation.chunk_id or -1), (None, None, None)
        )
        citations.append(
            CitationOut(
                id=citation.id,
                message_id=citation.message_id,
                chunk_id=citation.chunk_id,
                doc_id=citation.doc_id,
                rank=citation.rank,
                score=float(citation.score or 0.0),
                snippet=citation.snippet or "",
                doc_name=doc_name,
                page_no=page_no,
                section_path=section_path,
            )
        )
    return MessageOut(
        id=message.id,
        conversation_id=message.conversation_id,
        role=message.role,  # type: ignore[arg-type]  # 取值由 ORM 常量与 Literal 断言钉死
        content=message.content,
        mode=message.mode,
        model=message.model,
        prompt_tokens=message.prompt_tokens,
        completion_tokens=message.completion_tokens,
        cost_usd=float(message.cost_usd or 0.0),
        latency_ms=message.latency_ms,
        refusal=bool(message.refusal),
        retrieval_rounds=message.retrieval_rounds,
        trace_id=message.trace_id,
        citations=citations,
        created_at=message.created_at,
    )


@router.get("/conversations", response_model=Page[ConversationOut], summary="会话列表")
def list_conversations(
    params: PageDep,
    services: Services,
    user: CurrentUser,
    kb_id: Annotated[int | None, Query(ge=1, description="按知识库过滤")] = None,
) -> Page[ConversationOut]:
    rows, total = services.chat_service.list_conversations(
        user_id=user.id, page=params.page, size=params.size, kb_id=kb_id
    )
    return Page[ConversationOut].create(
        items=[ConversationOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
    summary="新建会话",
)
def create_conversation_route(
    payload: ConversationCreateRequest, services: Services, user: CurrentUser
) -> ConversationOut:
    """手动建会话（用于"新建对话"按钮，不发第一条消息）。

    `mode` 固定 `agent`：手动建的会话还没有消息，模式由第一次问答决定
    （`ChatService.resolve_conversation` 在传入 `conversation_id` 时沿用会话上的 kb_id）。
    """
    if payload.kb_id is not None:
        services.kb_service.get(payload.kb_id)  # 不存在 -> 404，而不是建一个坏会话
    conversation = create_conversation(services.session, user_id=user.id, kb_id=payload.kb_id)
    if payload.title:
        conversation.title = payload.title
    else:
        touch_conversation_title(conversation, question="新会话")
    services.session.commit()
    return ConversationOut.model_validate(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationOut, summary="会话详情")
def get_conversation(
    conversation_id: int, services: Services, user: CurrentUser
) -> ConversationOut:
    conversation = services.chat_service.get_conversation(
        conversation_id=conversation_id, user_id=user.id
    )
    return ConversationOut.model_validate(conversation)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessagePage,
    summary="消息回放（含引用）",
)
def list_messages(
    conversation_id: int, params: PageDep, services: Services, user: CurrentUser
) -> MessagePage:
    """按 id 升序回放消息，`citations` 里的 `doc_name` / `page_no` / `section_path` 已补齐。"""
    rows, total = services.chat_service.list_messages(
        conversation_id=conversation_id,
        user_id=user.id,
        page=params.page,
        size=params.size,
    )
    context = _citation_context(services, rows)
    return MessagePage.create(
        items=[_message_out(row, context) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话（软删）",
)
def delete_conversation(conversation_id: int, services: Services, user: CurrentUser) -> None:
    services.chat_service.delete_conversation(conversation_id=conversation_id, user_id=user.id)


@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackOut,
    summary="提交反馈",
)
def submit_feedback(
    message_id: int, payload: FeedbackRequest, services: Services, user: CurrentUser
) -> FeedbackOut:
    """Upsert 反馈（同一人对同一条消息只留最新评价，由唯一索引兜底）。

    `rating=-1` 时 comment 必填 —— 这条规则在 `FeedbackRequest` 里，
    路由不再重复判断（校验只有一处）。
    """
    feedback = services.chat_service.add_feedback(
        message_id=message_id,
        user_id=user.id,
        rating=payload.rating,
        comment=payload.comment,
    )
    if feedback is None:  # pragma: no cover - service 保证不返回 None
        raise ConversationNotFoundError(f"消息 {message_id} 的反馈写入失败")
    return FeedbackOut.model_validate(feedback)
