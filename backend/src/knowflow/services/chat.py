"""对话服务：把"一次问答"完整跑完并落库。

**这个类是一次问答的事务边界**，它负责：

    1. 解析/创建会话（多轮记忆的载体）
    2. 读历史（双重截断）
    3. 跑 LangGraph（agent 全图 或 rag 快路径）
    4. 落 user message / assistant message / 引用
    5. 落 trace（span 树 + token + 成本）
    6. 返回结构化的 ChatAnswer

**非流式与流式共用同一套落库逻辑**（`_persist_turn`），
这是刻意的：如果两套各写一遍，迟早会出现"流式问答的引用没存下来"
或者"两种接口的 token 统计口径不同"这类难查的不一致。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from knowflow.agent.memory import (
    append_message,
    create_conversation,
    load_history,
    touch_conversation_title,
)
from knowflow.agent.runner import AgentOutcome, AgentRunner
from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import ConversationNotFoundError
from knowflow.core.logging import get_logger
from knowflow.db.models.chat import MODE_AGENT, MODE_RAG, Conversation, Message, MessageCitation
from knowflow.llm.base import ChatModel
from knowflow.retrieval.engine import HybridRetriever
from knowflow.retrieval.types import RetrievalResult
from knowflow.services import obs_bridge
from knowflow.services.document import DocumentService
from knowflow.services.knowledge_base import KBService

logger = get_logger(__name__)

VALID_MODES = (MODE_AGENT, MODE_RAG)


@dataclass(slots=True)
class ChatAnswer:
    """一次非流式问答的结果。"""

    conversation_id: int
    message_id: int
    outcome: AgentOutcome
    trace_id: str
    cost_usd: Decimal = Decimal("0")
    cost_cny: Decimal = Decimal("0")
    latency_ms: int = 0
    model: str = ""
    offline: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_response(self) -> dict[str, Any]:
        """拼成 `ChatResponse` 需要的字段（schemas 层直接 model_validate 它）。"""
        usage = self.outcome.usage
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "trace_id": self.trace_id,
            "answer": self.outcome.answer,
            "sources": self.outcome.sources,
            "refusal": self.outcome.refusal,
            "retrieval_rounds": self.outcome.retrieval_rounds,
            "reflect_passed": self.outcome.reflect_passed,
            "grading": self.outcome.grading,
            "rewritten_queries": self.outcome.rewritten_queries,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            },
            "cost_usd": float(self.cost_usd),
            "latency_ms": self.latency_ms,
            "model": self.model,
            "offline": self.offline,
        }


class ChatService:
    def __init__(
        self,
        *,
        session: Session,
        retriever: HybridRetriever,
        runner: AgentRunner,
        kb_service: KBService,
        document_service: DocumentService,
        chat_model: ChatModel,
        settings: Settings | None = None,
        recorder_box: Any | None = None,
    ) -> None:
        self.session = session
        self.retriever = retriever
        self.runner = runner
        self.kb_service = kb_service
        self.document_service = document_service
        self.chat_model = chat_model
        self.settings = settings or get_settings()
        # 本次请求的 trace 持有者：写进去之后，图里的节点就能记 span。
        # **不用 contextvar**（流式生成器会被 Starlette 跨线程分次驱动，token 无法复位）。
        self.recorder_box = recorder_box

    # ------------------------------------------------------------------ 检索
    def search(
        self,
        *,
        kb_id: int,
        query: str,
        mode: str | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
        fusion: str | None = None,
        alpha: float | None = None,
        use_rerank: bool | None = None,
        use_autocut: bool | None = None,
        vector_threshold: float | None = None,
        keyword_threshold: float | None = None,
    ) -> RetrievalResult:
        """`/search` 检索调试接口：**不调用大模型，不花钱**。"""
        self.kb_service.get(kb_id)  # 校验存在性，不存在则 404
        return self.retriever.search(
            query=query,
            kb_id=kb_id,
            mode=mode,
            top_k=top_k,
            fetch_k=fetch_k,
            fusion=fusion,
            alpha=alpha,
            use_rerank=use_rerank,
            use_autocut=use_autocut,
            vector_threshold=vector_threshold,
            keyword_threshold=keyword_threshold,
        )

    # ------------------------------------------------------------------ 会话
    def resolve_conversation(
        self,
        *,
        user_id: int,
        conversation_id: int | None,
        kb_id: int | None,
        mode: str,
        question: str,
    ) -> Conversation:
        """取会话或新建。校验归属，防止越权读写别人的会话。"""
        if conversation_id is None:
            if kb_id is not None:
                self.kb_service.get(kb_id)  # 不存在直接 404
            created = create_conversation(self.session, user_id=user_id, kb_id=kb_id, mode=mode)
            touch_conversation_title(created, question=question)
            self.session.commit()
            return created

        # 变量名与上面分开写，避免 mypy 把类型推成 Conversation 后再赋 None
        existing = self.session.get(Conversation, conversation_id)
        if existing is None or existing.deleted_at is not None:
            raise ConversationNotFoundError(f"会话 {conversation_id} 不存在")
        if existing.user_id != user_id:
            # 不返回 403 而是 404：不泄露"这个 id 存在但不属于你"
            raise ConversationNotFoundError(f"会话 {conversation_id} 不存在")
        return existing

    def _effective_kb_id(self, conversation: Conversation, kb_id: int | None) -> int | None:
        """会话已绑定 KB 时以会话为准，避免"同一个会话里答案的检索范围变来变去"。"""
        return conversation.kb_id if conversation.kb_id is not None else kb_id

    def _resolve_kb_ids(self, kb_id: int | None) -> list[int]:
        if kb_id is not None:
            self.kb_service.get(kb_id)
            return [kb_id]
        return self.kb_service.active_kb_ids()

    # ------------------------------------------------------------------ 落库
    def _persist_turn(
        self,
        *,
        conversation: Conversation,
        question: str,
        outcome: AgentOutcome,
        mode: str,
        cost_usd: Decimal,
        latency_ms: int,
        trace_id: str,
        user_message_id: int | None = None,
    ) -> Message:
        """写 assistant 消息 + 引用。返回落库后的 Message。

        引用只存**真正展示给用户的那几条**（`outcome.sources`），
        不是全部候选：候选可能有 20 条，用户只看到 5 条，
        存 20 条会让"引用精度"这类指标失真。
        """
        message = append_message(
            self.session,
            conversation_id=conversation.id,
            role="assistant",
            content=outcome.answer,
            mode=mode,
            model=outcome.model or self.chat_model.model,
            prompt_tokens=int(outcome.usage.get("prompt_tokens", 0)),
            completion_tokens=int(outcome.usage.get("completion_tokens", 0)),
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            refusal=outcome.refusal,
            retrieval_rounds=outcome.retrieval_rounds,
            trace_id=trace_id,
        )
        for source in outcome.sources:
            self.session.add(
                MessageCitation(
                    message_id=message.id,
                    chunk_id=source.get("chunk_id"),
                    doc_id=source.get("doc_id"),
                    rank=int(source.get("rank", 0) or 0),
                    score=float(source.get("score", 0.0) or 0.0),
                    snippet=str(source.get("snippet", ""))[:500],
                )
            )
        self.session.commit()
        return message

    # ------------------------------------------------------------------ 非流式
    def answer(
        self,
        *,
        user_id: int,
        question: str,
        kb_id: int | None = None,
        conversation_id: int | None = None,
        mode: str = MODE_AGENT,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        use_memory: bool = True,
        request_id: str | None = None,
    ) -> ChatAnswer:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        resolved_mode = mode if mode in VALID_MODES else MODE_AGENT

        conversation = self.resolve_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            kb_id=kb_id,
            mode=resolved_mode,
            question=question,
        )
        target_kb = self._effective_kb_id(conversation, kb_id)
        kb_ids = self._resolve_kb_ids(target_kb)

        # 历史要在写入本轮 user 消息**之前**读，否则会把当前问题当成上文
        history = (
            load_history(self.session, conversation.id, settings=self.settings)
            if use_memory
            else []
        )

        append_message(
            self.session,
            conversation_id=conversation.id,
            role="user",
            content=question,
        )
        self.session.commit()

        recorder = obs_bridge.build_recorder("chat", trace_id=trace_id, settings=self.settings)
        if self.recorder_box is not None:
            self.recorder_box.value = recorder
        try:
            outcome = self.runner.run(
                question=question,
                kb_id=target_kb,
                kb_ids=kb_ids,
                top_k=top_k,
                use_rerank=use_rerank,
                history=history,
                mode=resolved_mode,
                thread_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001
            self.session.rollback()
            logger.error("chat.failed", error=f"{type(exc).__name__}: {exc}"[:300])
            latency_ms = int((time.perf_counter() - started) * 1000)
            obs_bridge.safe_persist(
                recorder,
                self.session,
                name="chat",
                mode=resolved_mode,
                user_id=user_id,
                conversation_id=conversation.id,
                kb_id=target_kb,
                request_id=request_id,
                status="error",
                latency_ms=latency_ms,
                extra={"error": f"{type(exc).__name__}: {exc}"[:300]},
            )
            raise
        finally:
            if self.recorder_box is not None:
                self.recorder_box.clear()

        latency_ms = int((time.perf_counter() - started) * 1000)
        cost_usd = obs_bridge.estimate_cost(
            outcome.model or self.chat_model.model,
            int(outcome.usage.get("prompt_tokens", 0)),
            int(outcome.usage.get("completion_tokens", 0)),
            self.settings,
        )
        message = self._persist_turn(
            conversation=conversation,
            question=question,
            outcome=outcome,
            mode=resolved_mode,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )
        obs_bridge.safe_persist(
            recorder,
            self.session,
            name="chat",
            mode=resolved_mode,
            user_id=user_id,
            conversation_id=conversation.id,
            kb_id=target_kb,
            request_id=request_id,
            status="ok",
            usage={
                "prompt_tokens": outcome.usage.get("prompt_tokens", 0),
                "completion_tokens": outcome.usage.get("completion_tokens", 0),
                "model": outcome.model,
                "llm_calls": outcome.usage.get("llm_calls", 0),
            },
            latency_ms=latency_ms,
            extra={
                "retrieval_rounds": outcome.retrieval_rounds,
                "source_count": len(outcome.sources),
                "refusal": outcome.refusal,
                "reflect_passed": outcome.reflect_passed,
                "vector_error": outcome.vector_error,
            },
        )

        return ChatAnswer(
            conversation_id=conversation.id,
            message_id=message.id,
            outcome=outcome,
            trace_id=trace_id,
            cost_usd=cost_usd,
            cost_cny=obs_bridge.to_cny(cost_usd, self.settings),
            latency_ms=latency_ms,
            model=outcome.model or self.chat_model.model,
            offline=self.chat_model.offline,
        )

    # ------------------------------------------------------------------ 流式
    def answer_stream(
        self,
        *,
        user_id: int,
        question: str,
        kb_id: int | None = None,
        conversation_id: int | None = None,
        mode: str = MODE_AGENT,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        use_memory: bool = True,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """流式问答。产出的每个 dict 都已经是"一帧 SSE 的内容"。

        帧序保证：`meta` → (`trace`|`tool`|`reflect`)* → `sources` → `token`* → `done` → `end`。
        `meta` 必须在**任何耗时操作之前**发出 —— 用户端 200ms 内看到响应，
        而不是盯着转圈等 5 秒（这是 SSE 体验的关键）。
        """
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        resolved_mode = mode if mode in VALID_MODES else MODE_AGENT

        conversation = self.resolve_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            kb_id=kb_id,
            mode=resolved_mode,
            question=question,
        )
        target_kb = self._effective_kb_id(conversation, kb_id)

        yield {
            "event": "meta",
            "conversation_id": conversation.id,
            "mode": resolved_mode,
            "trace_id": trace_id,
            "model": self.chat_model.model,
            "offline": self.chat_model.offline,
        }

        kb_ids = self._resolve_kb_ids(target_kb)
        history = (
            load_history(self.session, conversation.id, settings=self.settings)
            if use_memory
            else []
        )
        append_message(self.session, conversation_id=conversation.id, role="user", content=question)
        self.session.commit()

        recorder = obs_bridge.build_recorder("chat", trace_id=trace_id, settings=self.settings)
        accumulated: dict[str, Any] = {}
        status = "ok"

        # 把 recorder 写进本次请求的持有者（**不用 contextvar**：流式生成器会被
        # Starlette 用 iterate_in_threadpool 在不同线程上分次驱动，token 无法跨 context 复位）
        if self.recorder_box is not None:
            self.recorder_box.value = recorder
        try:
            for payload in self.runner.stream(
                question=question,
                kb_id=target_kb,
                kb_ids=kb_ids,
                top_k=top_k,
                use_rerank=use_rerank,
                history=history,
                mode=resolved_mode,
                thread_id=trace_id,
            ):
                if payload.get("event") == "done":
                    accumulated = dict(payload)
                    continue
                if payload.get("event") == "end":
                    continue
                if payload.get("event") == "error":
                    status = "error"
                yield payload
        finally:
            if self.recorder_box is not None:
                self.recorder_box.clear()

        latency_ms = int((time.perf_counter() - started) * 1000)
        outcome = self._outcome_from_done(accumulated)
        cost_usd = obs_bridge.estimate_cost(
            outcome.model or self.chat_model.model,
            int(outcome.usage.get("prompt_tokens", 0)),
            int(outcome.usage.get("completion_tokens", 0)),
            self.settings,
        )

        message_id: int | None = None
        try:
            message = self._persist_turn(
                conversation=conversation,
                question=question,
                outcome=outcome,
                mode=resolved_mode,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                trace_id=trace_id,
            )
            message_id = message.id
        except Exception as exc:  # noqa: BLE001
            # 答案已经流给用户了，落库失败不能把"已经成功的一次问答"变成错误。
            self.session.rollback()
            status = "error"
            logger.error("chat.persist_failed", error=f"{type(exc).__name__}: {exc}"[:300])

        obs_bridge.safe_persist(
            recorder,
            self.session,
            name="chat",
            mode=resolved_mode,
            user_id=user_id,
            conversation_id=conversation.id,
            kb_id=target_kb,
            request_id=request_id,
            status=status,
            usage={
                "prompt_tokens": outcome.usage.get("prompt_tokens", 0),
                "completion_tokens": outcome.usage.get("completion_tokens", 0),
                "model": outcome.model,
                "llm_calls": outcome.usage.get("llm_calls", 0),
            },
            latency_ms=latency_ms,
            extra={
                "retrieval_rounds": outcome.retrieval_rounds,
                "source_count": len(outcome.sources),
                "refusal": outcome.refusal,
                "reflect_passed": outcome.reflect_passed,
                "vector_error": outcome.vector_error,
            },
        )

        done = dict(accumulated)
        done["event"] = "done"
        done.update(
            {
                "conversation_id": conversation.id,
                "message_id": message_id,
                "trace_id": trace_id,
                "latency_ms": latency_ms,
                "cost_usd": float(cost_usd),
                "offline": self.chat_model.offline,
            }
        )
        yield done
        yield {"event": "end"}

    # ------------------------------------------------------------------ 辅助
    def _outcome_from_done(self, done: dict[str, Any]) -> AgentOutcome:
        """把 SSE `done` 帧反解成 AgentOutcome（用于落库）。

        为什么不直接让 runner 返回 outcome 对象：流式接口的契约是"事件流"，
        让生成器同时返回对象和事件会让调用方很难处理。
        反解一次的代价是一次字典读取，换来接口的一致与简单。
        """
        usage = done.get("usage") or {}
        return AgentOutcome(
            answer=str(done.get("answer") or ""),
            refusal=bool(done.get("refusal", False)),
            sources=list(done.get("sources") or []),
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "llm_calls": int(usage.get("llm_calls", 0) or 0),
            },
            retrieval_rounds=int(done.get("retrieval_rounds") or 0),
            reflect_passed=bool(done.get("reflect_passed", True)),
            grading=list(done.get("grading") or []),
            rewritten_queries=list(done.get("rewritten_queries") or []),
            tool_calls=list(done.get("tool_calls") or []),
            needs_retrieval=bool(done.get("needs_retrieval", True)),
            question_type=str(done.get("question_type") or "unknown"),
            model=str(done.get("model") or self.chat_model.model),
            vector_error=done.get("vector_error"),
            retrieval_ms=int(done.get("retrieval_ms") or 0),
            generate_ms=int(done.get("generate_ms") or 0),
        )

    # ------------------------------------------------------------------ 会话查询
    def list_conversations(
        self, *, user_id: int, page: int = 1, size: int = 20, kb_id: int | None = None
    ) -> tuple[list[Conversation], int]:
        from sqlalchemy import ColumnElement, func, select

        conditions: list[ColumnElement[bool]] = [
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        ]
        if kb_id is not None:
            conditions.append(Conversation.kb_id == kb_id)
        total = int(
            self.session.execute(select(func.count(Conversation.id)).where(*conditions)).scalar()
            or 0
        )
        if total == 0:
            return [], 0
        rows = list(
            self.session.execute(
                select(Conversation)
                .where(*conditions)
                .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                .offset((page - 1) * size)
                .limit(size)
            ).scalars()
        )
        return rows, total

    def get_conversation(self, *, conversation_id: int, user_id: int) -> Conversation:
        conversation = self.session.get(Conversation, conversation_id)
        if (
            conversation is None
            or conversation.deleted_at is not None
            or conversation.user_id != user_id
        ):
            raise ConversationNotFoundError(f"会话 {conversation_id} 不存在")
        return conversation

    def list_messages(
        self, *, conversation_id: int, user_id: int, page: int = 1, size: int = 50
    ) -> tuple[list[Message], int]:
        from sqlalchemy import func, select

        self.get_conversation(conversation_id=conversation_id, user_id=user_id)
        total = int(
            self.session.execute(
                select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
            ).scalar()
            or 0
        )
        if total == 0:
            return [], 0
        rows = list(
            self.session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
                .offset((page - 1) * size)
                .limit(size)
            ).scalars()
        )
        return rows, total

    def delete_conversation(self, *, conversation_id: int, user_id: int) -> None:
        conversation = self.get_conversation(conversation_id=conversation_id, user_id=user_id)
        conversation.mark_deleted()
        self.session.commit()
        logger.info("conversation.deleted", conversation_id=conversation_id, user_id=user_id)

    def add_feedback(
        self, *, message_id: int, user_id: int, rating: int, comment: str | None
    ) -> Any:
        """Upsert 反馈。同一人对同一条消息只保留最新评价（唯一索引兜底）。"""
        from sqlalchemy import select

        from knowflow.db.models.chat import Feedback

        message = self.session.get(Message, message_id)
        if message is None:
            raise ConversationNotFoundError(f"消息 {message_id} 不存在")
        conversation = self.session.get(Conversation, message.conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise ConversationNotFoundError(f"消息 {message_id} 不存在")

        existing = self.session.execute(
            select(Feedback).where(Feedback.message_id == message_id, Feedback.user_id == user_id)
        ).scalar_one_or_none()
        if existing is None:
            existing = Feedback(
                message_id=message_id, user_id=user_id, rating=rating, comment=comment
            )
            self.session.add(existing)
        else:
            existing.rating = rating
            existing.comment = comment
        self.session.commit()
        return existing

    def kb_stats_for_tool(self, kb_id: int | None) -> dict[str, Any]:
        """给 Agent 的 `kb_stats` 工具用。没有 kb_id 时汇总所有活跃 KB。"""
        if kb_id is not None:
            return self.kb_service.stats(kb_id)
        stats = {
            "doc_count": 0,
            "chunk_count": 0,
            "vector_count": 0,
            "bm25_doc_count": 0,
        }
        for kid in self.kb_service.active_kb_ids():
            one = self.kb_service.stats(kid)
            stats["doc_count"] += int(one["doc_count"])
            stats["chunk_count"] += int(one["chunk_count"])
            stats["vector_count"] += int(one["vector_count"])
            stats["bm25_doc_count"] += int(one["bm25_doc_count"])
        return stats

    def list_documents_for_tool(
        self, *, kb_id: int | None, keyword: str | None
    ) -> list[dict[str, Any]]:
        if kb_id is None:
            kb_ids = self.kb_service.active_kb_ids()
            kb_id = kb_ids[0] if kb_ids else None
        if kb_id is None:
            return []
        return self.document_service.list_documents_for_tool(kb_id=kb_id, keyword=keyword)


__all__ = ["VALID_MODES", "ChatAnswer", "ChatService"]
