"""Agent 运行器：把 LangGraph 的图包装成"一次问答"。

**为什么需要这一层**：图的输入输出是"状态字典"，而 API 层需要的是
"答案 + 引用 + 用量 + 反思结果"。这一层做映射与收尾，好处是：

- 图的形状可以被替换（`agent` 全图 / `rag` 快路径），而 API 代码零改动；
- 非流式（`POST /chat`）与流式（`POST /chat/stream`）**共用同一套逻辑**，
  避免"流式和非流式行为不一致"这个经典 bug（用户会发现两种接口答得不一样）。

流式实现的关键取舍：用 `stream_mode=["custom", "updates"]` **同时**收两类事件 ——
`custom` 是节点主动推的（token / sources / tool / reflect），
`updates` 是 LangGraph 报的状态增量。用后者自己累积出最终状态，
这样**即使没有 checkpointer 也能拿到完整结果**（不依赖额外存储）。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from knowflow.agent.memory import truncate_history
from knowflow.agent.nodes import select_contexts
from knowflow.agent.state import AgentState
from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.llm.base import ChatMessage
from knowflow.retrieval.types import Candidate

logger = get_logger(__name__)


@dataclass(slots=True)
class AgentOutcome:
    """一次 Agent 问答的完整产出（API 层据此拼 ChatResponse）。"""

    answer: str = ""
    refusal: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    retrieval_rounds: int = 0
    reflect_passed: bool = True
    reflect_rounds: int = 0
    grading: list[dict[str, Any]] = field(default_factory=list)
    rewritten_queries: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    needs_retrieval: bool = True
    question_type: str = "unknown"
    model: str = ""
    vector_error: str | None = None
    retrieval_ms: int = 0
    generate_ms: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def rewritten_only(self) -> list[str]:
        """去掉与原始问题相同的"改写"，只留下真正改过的查询（给前端展示）。"""
        return [q for q in self.rewritten_queries if q]

    def done_payload(self) -> dict[str, Any]:
        """SSE 的 `done` 帧内容（契约 5.6）。"""
        return {
            "answer": self.answer,
            "refusal": self.refusal,
            "sources": self.sources,
            "retrieval_rounds": self.retrieval_rounds,
            "reflect_passed": self.reflect_passed,
            "grading": self.grading,
            "rewritten_queries": self.rewritten_queries,
            "usage": {
                "prompt_tokens": self.usage.get("prompt_tokens", 0),
                "completion_tokens": self.usage.get("completion_tokens", 0),
                "total_tokens": self.usage.get("prompt_tokens", 0)
                + self.usage.get("completion_tokens", 0),
            },
            "model": self.model,
            "needs_retrieval": self.needs_retrieval,
            "question_type": self.question_type,
            "tool_calls": self.tool_calls,
            "vector_error": self.vector_error,
            "retrieval_ms": self.retrieval_ms,
            "generate_ms": self.generate_ms,
        }


class AgentRunner:
    """图的门面。`agent_graph` 与 `fixed_graph` 由 container 注入。"""

    def __init__(
        self,
        *,
        agent_graph: Any,
        fixed_graph: Any | None = None,
        settings: Settings | None = None,
        model_name: str = "",
    ) -> None:
        self.agent_graph = agent_graph
        self.fixed_graph = fixed_graph
        self.settings = settings or get_settings()
        self.model_name = model_name

    # ------------------------------------------------------------------ 内部
    def _graph_for(self, mode: str) -> Any:
        """`rag` 模式走快路径（不反思、不改写），`agent` 走全图。"""
        if mode == "rag" and self.fixed_graph is not None:
            return self.fixed_graph
        return self.agent_graph

    def _initial_state(
        self,
        *,
        question: str,
        kb_id: int | None,
        kb_ids: Sequence[int] | None,
        top_k: int | None,
        use_rerank: bool | None,
        history: Sequence[ChatMessage] | None,
        mode: str,
    ) -> AgentState:
        resolved_kbs = list(kb_ids) if kb_ids else ([kb_id] if kb_id else [])
        trimmed = truncate_history(
            list(history or []),
            max_turns=self.settings.memory_max_turns,
            max_chars=self.settings.memory_max_chars,
        )
        state: AgentState = {
            "question": question,
            "kb_id": kb_id,
            "kb_ids": resolved_kbs,
            "top_k": top_k or self.settings.top_k,
            "use_rerank": self.settings.rerank_enabled if use_rerank is None else use_rerank,
            "history": list(trimmed),
            "mode": mode,
            "retrieval_rounds": 0,
            "rewrite_round": 0,
            "reflect_round": 0,
            "retrieval_ms": 0,
            "model": self.model_name,
        }
        return state

    @staticmethod
    def _merge_updates(accumulated: dict[str, Any], update: Any) -> None:
        """把 LangGraph 的 `updates` 增量并进一个普通 dict。

        只关心**标量字段**（answer/refusal/usage/…），因为累加字段
        （candidates/all_queries）的真值会由 `graph.invoke` 或单独的 get_state 给出。
        这里用 update 覆盖即可，不做 reducer 语义模拟。
        """
        if not isinstance(update, dict):
            return
        for _node_name, partial in update.items():
            if isinstance(partial, dict):
                accumulated.update(partial)

    def _to_outcome(self, state: dict[str, Any]) -> AgentOutcome:
        candidates = list(state.get("candidates") or [])
        # 复用 generate 节点选上下文的那套逻辑，保证"展示的引用"和"喂给模型的上下文"完全一致
        used = select_contexts(state)  # type: ignore[arg-type]
        sources = [c.to_source_dict(i) for i, c in enumerate(used, start=1)]

        usage_raw = state.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "llm_calls": int(usage_raw.get("llm_calls", 0) or 0),
        }

        return AgentOutcome(
            answer=str(state.get("answer") or ""),
            refusal=bool(state.get("refusal", False)),
            sources=sources,
            candidates=candidates,
            usage=usage,
            retrieval_rounds=int(state.get("retrieval_rounds") or 0),
            reflect_passed=bool(state.get("reflect_passed", True)),
            reflect_rounds=int(state.get("reflect_round") or 0),
            grading=list(state.get("grading") or []),
            rewritten_queries=list(state.get("all_queries") or []),
            tool_calls=list(state.get("tool_calls") or []),
            needs_retrieval=bool(state.get("needs_retrieval", True)),
            question_type=str(state.get("question_type") or "unknown"),
            model=str(state.get("model") or self.model_name),
            vector_error=state.get("vector_error"),
            retrieval_ms=int(state.get("retrieval_ms") or 0),
            generate_ms=int(state.get("generate_ms") or 0),
            errors=list(state.get("errors") or []),
        )

    # ------------------------------------------------------------------ 对外
    def run(
        self,
        *,
        question: str,
        kb_id: int | None = None,
        kb_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        history: Sequence[ChatMessage] | None = None,
        mode: str = "agent",
        thread_id: str | None = None,
    ) -> AgentOutcome:
        """非流式执行（`POST /chat` 用）。"""
        graph = self._graph_for(mode)
        state = self._initial_state(
            question=question,
            kb_id=kb_id,
            kb_ids=kb_ids,
            top_k=top_k,
            use_rerank=use_rerank,
            history=history,
            mode=mode,
        )
        config = _config(thread_id)
        final = graph.invoke(state, config)
        return self._to_outcome(dict(final))

    def stream(
        self,
        *,
        question: str,
        kb_id: int | None = None,
        kb_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
        history: Sequence[ChatMessage] | None = None,
        mode: str = "agent",
        thread_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """流式执行：产出**已经可以直接序列化成 SSE 帧**的事件字典。

        事件顺序保证（契约 5.6）：`meta` → (`trace`|`tool`|`reflect`)* →
        `sources` → `token`* → `done` → `end`。
        `meta`/`done`/`end` 由本方法补，其余由节点用 stream writer 推。
        """
        graph = self._graph_for(mode)
        state = self._initial_state(
            question=question,
            kb_id=kb_id,
            kb_ids=kb_ids,
            top_k=top_k,
            use_rerank=use_rerank,
            history=history,
            mode=mode,
        )
        config = _config(thread_id)

        accumulated: dict[str, Any] = {}
        try:
            for mode_name, payload in graph.stream(
                state, config, stream_mode=["custom", "updates"]
            ):
                if mode_name == "custom":
                    if isinstance(payload, dict):
                        yield payload
                elif mode_name == "updates":
                    self._merge_updates(accumulated, payload)
        except Exception as exc:  # noqa: BLE001 - 流一旦开始就不能再抛给 HTTP 层了
            logger.error("agent.stream_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            yield {
                "event": "error",
                "code": "AGENT_ERROR",
                "message": f"生成过程中出错：{type(exc).__name__}",
            }
            outcome = self._to_outcome(accumulated)
            yield {"event": "done", **outcome.done_payload()}
            yield {"event": "end"}
            return

        outcome = self._to_outcome(accumulated)
        if not outcome.answer and not outcome.refusal:
            # 图跑完却没有答案：说明某个节点静默失败了。**不能返回空字符串**，
            # 那会让前端显示一个空气泡，用户以为界面卡了。
            outcome.answer = "抱歉，本次生成没有产出内容。请重试或换一种问法。"
            outcome.refusal = False
        yield {"event": "done", **outcome.done_payload()}
        yield {"event": "end"}


def _config(thread_id: str | None) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id or "anonymous"}} if thread_id else {}


__all__ = ["AgentOutcome", "AgentRunner"]
