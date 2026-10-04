"""对话路由（契约 5.5 / 5.6）：`POST /chat` 与 `POST /chat/stream`。

**一句话定位**：非流式与流式两种问答出口，共用同一个请求体与同一套落库逻辑。

**在链路中的位置**：前端对话页 -> **本模块** -> `ChatService.answer` /
`ChatService.answer_stream` -> Agent 图（LangGraph）-> 检索 + 生成 + 落库 + trace。

**关键设计取舍**（这一层是全项目踩坑最密集的地方）：

1. **不用 `sse-starlette` 的 `EventSourceResponse`**：它会额外插入自己的心跳帧与字段，
   而契约 5.6 把帧格式钉死成 `event: <name>\\ndata: <json>\\n\\n`。
   手写产帧只有 3 行代码，换来的是"前端拿到的每一帧都完全可控"。
2. **必须带 `X-Accel-Buffering: no`**：这是给 Nginx 的。不加的话生产环境上
   Nginx 会把整个响应缓冲到结束才吐给浏览器——**流式接口看起来"不流式"了**，
   而本地直连 uvicorn 时完全正常，属于最难复现的那类问题。
   `Cache-Control: no-cache` / `Connection: keep-alive` 同理，缺一个都可能在中间层被改写。
3. **路由函数写同步 `def`**。`ChatService.answer_stream` 是**同步生成器**，
   Starlette 对同步生成器会自动用 `iterate_in_threadpool` 逐帧取；
   写成 `async def` 再直接 `for frame in gen` 会**阻塞事件循环**——
   一个流式问答期间整个服务不响应任何请求。
4. **生成器内部整体 try/except**。响应头在第一帧之前就已经发出去了，
   那时再抛异常**不可能**再走 `main.py` 的异常处理器：客户端只会看到一个断掉的连接。
   所以任何异常都要在生成器里转成 `error` 帧 + `end` 帧（契约 5.6 的顺序保证）。
5. **`meta` 帧的 `embedding_mode` 由本层补**：`answer_stream` 只回 `model` / `offline`，
   而契约 5.6 要求 `meta` 里有 `embedding_mode`（向量化降级时前端要显著提示）。
   本层读容器状态补上，比让 service 依赖容器更干净。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from knowflow.api.deps import ContainerDep, CurrentUser, Services
from knowflow.core.exceptions import KnowFlowError
from knowflow.core.logging import get_logger
from knowflow.schemas.chat import ChatRequest, ChatResponse
from knowflow.services.chat import ChatAnswer

__all__ = ["SSE_HEADERS", "encode_frame", "router"]

logger = get_logger(__name__)

router = APIRouter(tags=["对话"])

#: SSE 响应头。三个都不能省，原因见模块 docstring 第 2 条。
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def encode_frame(payload: dict[str, Any]) -> str:
    """把 `{"event": ..., ...}` 编码成一帧 SSE 文本。

    `data` 用 `ensure_ascii=False`：中文答案如果被转义成 `\\u4e00` 这样的形式，
    帧体积会翻好几倍，抓包排障时也完全没法读。
    Pydantic 对象直接 `model_dump`，避免调用方还要记着"这一帧是 dict 还是模型"。
    """
    frame = dict(payload)
    event = str(frame.pop("event", "message"))
    body = _jsonable(frame)
    return f"event: {event}\ndata: {json.dumps(body, ensure_ascii=False, default=str)}\n\n"


def _jsonable(value: Any) -> Any:
    """递归把 Pydantic 模型 / Decimal 等转成可 JSON 序列化的结构。"""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _response_of(answer: ChatAnswer) -> ChatResponse:
    """`ChatAnswer.to_response()` 已经是契约 5.5 的字段形状，直接用。
    不在这里重新拼一份 dict：两份字段定义迟早漂移（改了一处忘了另一处）。"""
    return ChatResponse.model_validate(answer.to_response())


@router.post("/chat", response_model=ChatResponse, summary="问答（非流式）")
def chat(
    payload: ChatRequest, request: Request, services: Services, user: CurrentUser
) -> ChatResponse:
    """一次问答跑完整条链路（检索 -> 判定 -> 生成 -> 落库 -> trace）后一次性返回。"""
    request_id = getattr(request.state, "request_id", None)
    answer = services.chat_service.answer(
        user_id=user.id,
        question=payload.question,
        kb_id=payload.kb_id,
        conversation_id=payload.conversation_id,
        mode=payload.mode,
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        use_memory=payload.use_memory,
        request_id=request_id,
    )
    return _response_of(answer)


@router.post("/chat/stream", summary="问答（SSE 流式）")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    services: Services,
    user: CurrentUser,
    container: ContainerDep,
) -> StreamingResponse:
    """SSE 流式问答。

    帧序保证（契约 5.6）：`meta` → (`trace`|`tool`|`reflect`)* → `sources` → `token`*
    → `done` → `end`。任何阶段出错：`error` 帧 + `end` 帧。
    """
    request_id = getattr(request.state, "request_id", None)
    # 向量化真实模式在这里读一次：`meta` 帧要如实上报降级状态（"不骗人"硬规则）。
    embedding_mode = container.embedding_mode()

    frames = services.chat_service.answer_stream(
        user_id=user.id,
        question=payload.question,
        kb_id=payload.kb_id,
        conversation_id=payload.conversation_id,
        mode=payload.mode,
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        use_memory=payload.use_memory,
        request_id=request_id,
    )

    return StreamingResponse(
        _sse_frames(frames, embedding_mode=embedding_mode, request_id=request_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _fallback_meta(embedding_mode: str) -> dict[str, Any]:
    """兜底 `meta` 帧：只在"一帧都没发出去就抛异常"时使用。

    `conversation_id=0` / `trace_id=""` 是显式的"这一轮没建立起来"，
    比缺字段更安全：前端拿 `undefined` 去建气泡会直接崩在渲染里。
    """
    return {
        "event": "meta",
        "conversation_id": 0,
        "mode": "agent",
        "trace_id": "",
        "model": "",
        "offline": False,
        "embedding_mode": embedding_mode,
    }


def _sse_frames(
    frames: Iterator[dict[str, Any]],
    *,
    embedding_mode: str,
    request_id: str | None,
) -> Iterator[str]:
    """把 service 的事件字典流翻译成 SSE 文本流。

    两件事必须做对：

    1. **`meta` 一定是第一帧**。`answer_stream` 在解析会话（可能抛
       `ConversationNotFoundError`）之前就 yield `meta`，正常路径天然满足；
       但万一将来它变了，前端会拿到一个"没有会话 id 的流"。
       所以这里记住"发过 meta 没有"，没发过就补一帧。
    2. **异常一律转成 `error` + `end`**。响应头已经发出去了，
       异常冒泡只会得到一个断掉的连接（用户看到答案打到一半没了，且没有任何提示）。
    """
    sent_meta = False
    try:
        for payload in frames:
            if payload.get("event") == "meta":
                sent_meta = True
                # 契约 5.6 要求 meta 带 embedding_mode（降级要显著提示），
                # 而 service 只回 model/offline —— 由本层补，见模块 docstring 第 5 条。
                if not payload.get("embedding_mode"):
                    payload = {**payload, "embedding_mode": embedding_mode}
            yield encode_frame(payload)
    except KnowFlowError as exc:
        if not sent_meta:
            yield encode_frame(_fallback_meta(embedding_mode))
        yield encode_frame(
            {"event": "error", "code": exc.code, "message": exc.message, "request_id": request_id}
        )
    except Exception as exc:  # noqa: BLE001 - SSE 边界：任何异常都必须变成帧
        logger.error(
            "chat.stream_failed",
            error=f"{type(exc).__name__}: {exc}"[:300],
            exc_info=True,
        )
        if not sent_meta:
            yield encode_frame(_fallback_meta(embedding_mode))
        yield encode_frame(
            {
                "event": "error",
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误，请稍后重试",
                "request_id": request_id,
            }
        )
    finally:
        # `end` 帧**始终**是最后一帧（无论成功、出错还是客户端断开）。
        # 前端收到它就收尾；没有它前端只能等超时，用户会觉得"卡住了"。
        yield encode_frame({"event": "end"})
