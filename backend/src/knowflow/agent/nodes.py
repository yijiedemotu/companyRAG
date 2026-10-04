"""LangGraph 的各节点实现。

**图的结构由 `graph.py` 定义，节点内部做什么由这里定义。**
把两者分开的好处：改"要不要反思"只动 graph.py，改"怎么判相关性"只动这里。

六个节点的职责与产出：

| 节点 | 干什么 | 写回 state |
| --- | --- | --- |
| `analyze` | 判断要不要检索；生成检索用查询 | needs_retrieval / queries / question_type |
| `retrieve` | 模型驱动工具循环（或降级为直接检索） | candidates / gate_* / tool_calls / retrieval_rounds |
| `grade` | CRAG：逐条判断召回是否相关 | relevant_ids / grading |
| `rewrite` | 召回不足时改写查询（有次数上限） | queries / rewrite_round |
| `generate` | 组装上下文生成答案（流式推 token） | answer / refusal / usage |
| `reflect` | Self-RAG：检查答案是否有引用支撑 | reflect_passed / unsupported / reflect_instruction |

**所有节点都有"模型抽风也要能继续"的降级分支**。这不是防御性编程洁癖：
LLM 输出 JSON 失败是常态（实测约 2%~5%），如果每次失败都中断整个问答，
用户体验会是"每 30 次提问就报一次错"。正确的做法是降级 + 记录 + 继续。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from knowflow.agent.state import AgentState
from knowflow.agent.tools import TOOL_KB_SEARCH, ToolRegistry, ToolResult
from knowflow.core.config import Settings
from knowflow.core.logging import get_logger
from knowflow.db.models.observability import SPAN_LLM, SPAN_NODE, SPAN_RETRIEVAL, SPAN_TOOL
from knowflow.llm.base import ChatMessage, ChatModel, ChatUsage
from knowflow.llm.parsing import extract_json_object
from knowflow.llm.prompts import (
    REFUSAL_ANSWER,
    build_analyze_messages,
    build_direct_messages,
    build_grading_messages,
    build_rag_messages,
    build_reflect_messages,
    build_rewrite_messages,
    detect_refusal,
)
from knowflow.retrieval.engine import HybridRetriever
from knowflow.retrieval.types import Candidate

logger = get_logger(__name__)

# 工具循环的最大轮数：防止模型无限调工具（每轮都是一次 API 调用 = 钱和时间）
MAX_TOOL_ITERATIONS = 3
# 一次工具循环里最多执行多少次 kb_search
MAX_SEARCH_CALLS = 4


# --------------------------------------------------------------------------------------
# 依赖与埋点
# --------------------------------------------------------------------------------------
class RecorderBox:
    """一次请求的 `TraceRecorder` 持有者（显式传递，不用 contextvars）。

    **为什么不用 contextvars**（这是踩出来的，值得写清楚）：

    我们最初用模块级 `ContextVar` 绑定 recorder，非流式路径完全正常。
    但**流式路径必然报错**：

        ValueError: <Token var=<ContextVar name='current_recorder'>> was created
                    in a different Context

    原因是 Starlette 迭代**同步生成器**时会用 `iterate_in_threadpool`，
    **每取一个元素就是一次 `run_in_threadpool(next, ...)`** ——
    也就是每个 `next()` 可能落在不同的线程、拿着**各自的 context 副本**上。
    而 `ContextVar.set()` 返回的 token 只能在**同一个 context** 里 `reset()`。
    生成器第 1 次 yield 时 set，第 N 次 yield 后退出 `with` 时 reset → 直接抛异常。

    流式恰好是本项目的主路径，所以这个问题必须从根上解决：
    **把 recorder 挂在依赖对象上显式传递**，不依赖任何隐式上下文。
    生成器在哪个线程跑都不影响 —— 它读的是同一个对象。
    """

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: Any = None

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, recorder: Any) -> None:
        self._value = recorder

    def clear(self) -> None:
        self._value = None


@dataclass(slots=True)
class AgentDeps:
    """节点需要的一切。用依赖注入而不是全局单例，测试才能塞假实现。"""

    settings: Settings
    chat_model: ChatModel
    judge_model: ChatModel
    retriever: HybridRetriever
    tools: ToolRegistry
    #: 本次请求的 trace 收集器（显式传递；见 `RecorderBox` 的说明）
    recorder_box: RecorderBox = field(default_factory=RecorderBox)


def _get_recorder() -> Any:
    """兜底：从 contextvar 取 recorder（**仅供非流式路径**）。

    这里用 try/except 而不是顶层 import，是为了让 `agent/` 在
    "可观测模块尚未装载"的场景下（脚本、单测）依然可用 ——
    可观测是横切关注点，**不该成为业务代码的硬依赖**。
    """
    try:
        from knowflow.observability.tracing import get_recorder

        return get_recorder()
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def node_span(
    deps: AgentDeps, name: str, *, span_type: str = SPAN_NODE, payload: Any = None
) -> Iterator[None]:
    """给一个节点/一次模型调用记 span。

    优先用 `deps.recorder_box`（显式、线程无关），
    取不到再退回 contextvar（兼容脚本与单测里直接调用节点的场景）。
    """
    recorder = deps.recorder_box.value or _get_recorder()
    if recorder is None:
        with nullcontext():
            yield
        return
    with recorder.span(name, span_type=span_type, input=payload):
        yield


def _writer() -> Any:
    """取 LangGraph 的自定义流写入器；不在流式运行时返回 None。"""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001
        return None


def emit(payload: dict[str, Any]) -> None:
    """推一个 SSE 事件。**任何异常都不能影响主流程**（前端掉线不该让问答失败）。"""
    writer = _writer()
    if writer is None:
        return
    try:
        writer(payload)
    except Exception as exc:  # noqa: BLE001
        logger.debug("agent.emit_failed", error=f"{type(exc).__name__}: {exc}")


def _usage_dict(usage: ChatUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "llm_calls": max(usage.llm_calls, 1),
    }


# --------------------------------------------------------------------------------------
# ① analyze：规划
# --------------------------------------------------------------------------------------
def analyze_node(deps: AgentDeps) -> Any:
    def node(state: AgentState) -> dict[str, Any]:
        question = (state.get("question") or "").strip()
        history = list(state.get("history") or [])

        if not question:
            return {
                "needs_retrieval": False,
                "queries": [],
                "question_type": "empty",
                "analyze_reason": "空问题",
                "answer": "请提出一个具体的问题。",
                "refusal": True,
                "finished": True,
            }

        with node_span(deps, "analyze", payload={"question": question}):
            emit({"event": "trace", "node": "analyze", "status": "start"})
            started = _now()
            messages = build_analyze_messages(
                question=question, history_summary=_history_summary(history)
            )
            decision: dict[str, Any] = {}
            try:
                result = deps.chat_model.complete(messages, temperature=0.0)
                decision = extract_json_object(result.text) or {}
                usage = _usage_dict(result.usage)
            except Exception as exc:  # noqa: BLE001
                logger.warning("agent.analyze_failed", error=f"{type(exc).__name__}: {exc}"[:200])
                decision = {}
                usage = {}

            needs = decision.get("needs_retrieval")
            if not isinstance(needs, bool):
                # 降级：拿不到判断就**默认检索**。理由：多检索一次的成本
                # （几十毫秒 + 一点 token）远小于"该查没查"导致的错误答案。
                needs = True
            queries = _normalize_queries(decision.get("queries"), fallback=question)

            duration = _now() - started
            emit(
                {
                    "event": "trace",
                    "node": "analyze",
                    "status": "ok",
                    "duration_ms": duration,
                    "detail": {
                        "needs_retrieval": needs,
                        "question_type": decision.get("question_type") or "unknown",
                        "queries": queries,
                    },
                }
            )

        return {
            "needs_retrieval": needs,
            "queries": queries,
            "all_queries": queries,
            "question_type": str(decision.get("question_type") or "unknown"),
            "analyze_reason": str(decision.get("reason") or "")[:300],
            "usage": usage,
        }

    return node


# --------------------------------------------------------------------------------------
# ② retrieve：工具循环（检索）
# --------------------------------------------------------------------------------------
def retrieve_node(deps: AgentDeps) -> Any:
    def node(state: AgentState) -> dict[str, Any]:
        queries = _normalize_queries(state.get("queries"), fallback=state.get("question", ""))
        kb_ids: list[int] = [int(k) for k in (state.get("kb_ids") or []) if k is not None]
        if not kb_ids and state.get("kb_id") is not None:
            kb_ids = [int(state["kb_id"])]  # type: ignore[arg-type]
        top_k = int(state.get("top_k") or deps.settings.top_k)
        use_rerank = bool(state.get("use_rerank", deps.settings.rerank_enabled))
        history = list(state.get("history") or [])

        emit({"event": "trace", "node": "retrieve", "status": "start"})
        started = _now()

        all_candidates: list[Candidate] = []
        tool_events: list[dict[str, Any]] = []
        vector_error: str | None = None
        gate_passed = False
        gate_reason = ""
        used_tool_loop = False

        with node_span(deps, "retrieve", span_type=SPAN_RETRIEVAL, payload={"queries": queries}):
            if deps.chat_model.supports_tools():
                used_tool_loop = True
                results = _run_tool_loop(
                    deps,
                    question=state.get("question", ""),
                    queries=queries,
                    kb_ids=kb_ids,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    history=history,
                )
            else:
                # ---------- 优雅降级：模型不支持工具调用 ----------
                # 直接用 analyze 产出的查询逐条检索，把"工具调用"退化成"固定检索"。
                # 能力下降（不能根据中间结果改查询），但功能完整、且行为可预测。
                logger.info("agent.tools_unsupported", model=deps.chat_model.model)
                results = [
                    deps.tools.invoke(
                        TOOL_KB_SEARCH,
                        {"query": query, "top_k": top_k},
                        kb_ids=kb_ids,
                        top_k=top_k,
                        use_rerank=use_rerank,
                    )
                    for query in queries[:MAX_SEARCH_CALLS]
                ]

            for result in results:
                tool_events.append(result.to_event())
                emit({"event": "tool", **result.to_event()})
                if result.candidates:
                    all_candidates.extend(result.candidates)
                payload = result.payload or {}
                if payload.get("gate", {}) and payload["gate"].get("passed"):
                    gate_passed = True
                    gate_reason = str(payload["gate"].get("reason") or "")
                elif not gate_reason and payload.get("gate"):
                    gate_reason = str(payload["gate"].get("reason") or "")
                debug = payload.get("debug") or {}
                if debug.get("vector_error"):
                    vector_error = str(debug["vector_error"])

        merged = _dedupe_candidates(all_candidates)[: max(top_k * 3, top_k)]
        duration = _now() - started

        emit(
            {
                "event": "trace",
                "node": "retrieve",
                "status": "ok",
                "duration_ms": duration,
                "detail": {
                    "rounds": int(state.get("retrieval_rounds") or 0) + 1,
                    "candidates": len(merged),
                    "gate_passed": gate_passed,
                    "tool_loop": used_tool_loop,
                },
            }
        )

        return {
            "candidates": merged,
            "tool_calls": tool_events,
            "retrieval_rounds": int(state.get("retrieval_rounds") or 0) + 1,
            "retrieval_ms": int(state.get("retrieval_ms") or 0) + duration,
            "gate_passed": gate_passed,
            "gate_reason": gate_reason,
            "vector_error": vector_error,
        }

    return node


def _run_tool_loop(
    deps: AgentDeps,
    *,
    question: str,
    queries: Sequence[str],
    kb_ids: Sequence[int],
    top_k: int,
    use_rerank: bool,
    history: Sequence[ChatMessage],
) -> list[ToolResult]:
    """模型驱动的工具循环。

    **与固定链的本质区别**：这里模型看到第一批检索结果后，
    可以决定"信息不够，换个词再查一次"或者"够了，不用再查"。
    循环上限 `MAX_TOOL_ITERATIONS` 是硬约束 —— 没有它，一个爱查的模型
    能把一次提问变成 20 次 API 调用。
    """
    tools = deps.tools
    messages: list[ChatMessage] = [
        ChatMessage(
            role="system",
            content=(
                "你可以调用工具检索企业知识库。请先用最精确的关键词调用 kb_search；"
                "如果返回的结果不足以回答用户问题，可以换关键词再查一次（最多 3 次）。"
                "信息足够时直接给出最终回答，不要再调用工具。"
            ),
        )
    ]
    messages.extend(m for m in history if m.role in ("user", "assistant"))
    messages.append(ChatMessage(role="user", content=question))

    results: list[ToolResult] = []
    search_calls = 0

    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        # 第一轮强制先执行 analyze 给出的查询 —— 保证"至少检索一次"，
        # 也避免模型在第一轮就空手回答（那等于绕过了 RAG）。
        if iteration == 1 and queries:
            for query in queries[:MAX_SEARCH_CALLS]:
                result = _invoke_search(
                    deps, query=query, kb_ids=kb_ids, top_k=top_k, use_rerank=use_rerank
                )
                results.append(result)
                search_calls += 1
                messages.append(
                    ChatMessage(role="assistant", content=f"（调用 kb_search：{query}）")
                )
                messages.append(ChatMessage(role="tool", content=result.to_tool_message()))

        if search_calls >= MAX_SEARCH_CALLS:
            break

        with node_span(
            deps, "tool_decision", span_type=SPAN_TOOL, payload={"iteration": iteration}
        ):
            try:
                decision = deps.chat_model.complete_with_tools(  # type: ignore[attr-defined]
                    messages, tools=tools.schemas(), temperature=0.0
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent.tool_decision_failed", error=f"{type(exc).__name__}: {exc}"[:200]
                )
                break

        if not decision.tool_calls:
            break

        executed = 0
        for call in decision.tool_calls:
            if call.name != TOOL_KB_SEARCH:
                # 非检索类工具（kb_stats / list_documents）执行一次即可
                result = tools.invoke(call.name, call.arguments, kb_ids=kb_ids, top_k=top_k)
                results.append(result)
                messages.append(ChatMessage(role="tool", content=result.to_tool_message()))
                continue
            if search_calls >= MAX_SEARCH_CALLS:
                break
            result = tools.invoke(
                call.name,
                call.arguments,
                kb_ids=kb_ids,
                top_k=top_k,
                use_rerank=use_rerank,
            )
            results.append(result)
            search_calls += 1
            executed += 1
            messages.append(ChatMessage(role="tool", content=result.to_tool_message()))

        if executed == 0:
            break

    return results


def _invoke_search(
    deps: AgentDeps,
    *,
    query: str,
    kb_ids: Sequence[int],
    top_k: int,
    use_rerank: bool,
) -> ToolResult:
    return deps.tools.invoke(
        TOOL_KB_SEARCH,
        {"query": query, "top_k": top_k},
        kb_ids=kb_ids,
        top_k=top_k,
        use_rerank=use_rerank,
    )


# --------------------------------------------------------------------------------------
# ③ grade：CRAG 相关性判定
# --------------------------------------------------------------------------------------
def grade_node(deps: AgentDeps) -> Any:
    def node(state: AgentState) -> dict[str, Any]:
        candidates = list(state.get("candidates") or [])
        question = state.get("question", "")
        cfg = deps.settings

        if not candidates:
            return {"relevant_ids": [], "grading": [], "grading_skipped": False}

        # 候选不多时直接全放行：省一次模型调用。
        # 3 条以内基本都在 top_k 里，逐条判定的收益小于成本。
        if len(candidates) <= 3:
            return {
                "relevant_ids": [c.vector_id for c in candidates],
                "grading": [],
                "grading_skipped": True,
            }

        emit({"event": "trace", "node": "grade", "status": "start"})
        started = _now()
        grading_items: list[dict[str, Any]] = []
        relevant_ids: list[str] = []

        with node_span(deps, "grade", payload={"candidates": len(candidates)}):
            if not cfg.agent_grading_enabled:
                relevant_ids = _heuristic_relevance(candidates, cfg)
                strategy = "heuristic(disabled)"
            else:
                contexts = [(i, c.content) for i, c in enumerate(candidates, start=1)]
                judged = _llm_grade(deps, question=question, contexts=contexts)
                if judged is None:
                    relevant_ids = _heuristic_relevance(candidates, cfg)
                    strategy = "heuristic(llm_failed)"
                else:
                    strategy = "llm"
                    for index, candidate in enumerate(candidates, start=1):
                        item = judged.get(index)
                        is_relevant = bool(item.get("relevant")) if item else False
                        grading_items.append(
                            {
                                "chunk_id": candidate.chunk_id,
                                "vector_id": candidate.vector_id,
                                "relevant": is_relevant,
                                "reason": str((item or {}).get("reason") or "")[:200],
                            }
                        )
                        if is_relevant:
                            relevant_ids.append(candidate.vector_id)

        duration = _now() - started
        emit(
            {
                "event": "trace",
                "node": "grade",
                "status": "ok",
                "duration_ms": duration,
                "detail": {
                    "strategy": strategy,
                    "total": len(candidates),
                    "relevant": len(relevant_ids),
                },
            }
        )

        return {
            "relevant_ids": relevant_ids,
            "grading": grading_items,
            "grading_skipped": False,
        }

    return node


def _llm_grade(
    deps: AgentDeps, *, question: str, contexts: Sequence[tuple[int, str]]
) -> dict[int, dict[str, Any]] | None:
    """让模型逐条判定相关性。返回 `{编号: {relevant, reason}}`，失败返回 None。"""
    messages = build_grading_messages(question=question, contexts=contexts)
    try:
        with node_span(deps, "grade.llm", span_type=SPAN_LLM):
            result = deps.judge_model.complete(messages, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent.grade_llm_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        return None

    obj = extract_json_object(result.text)
    if not obj:
        logger.warning("agent.grade_unparsable", preview=result.text[:120])
        return None
    raw_items = obj.get("items")
    if not isinstance(raw_items, list):
        return None

    judged: dict[int, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if raw_id is None:
            continue
        try:
            index = int(raw_id)
        except (TypeError, ValueError):
            continue
        judged[index] = item
    return judged or None


def _heuristic_relevance(candidates: Sequence[Candidate], cfg: Settings) -> list[str]:
    """零成本相关性判定：向量分或关键词覆盖率任一达标就算相关。"""
    relevant: list[str] = []
    for candidate in candidates:
        vector_ok = (candidate.vector_score or 0.0) >= cfg.vector_min_score
        keyword_ok = candidate.keyword_coverage >= cfg.keyword_min_coverage
        if vector_ok or keyword_ok:
            relevant.append(candidate.vector_id)
    # 都没达标但确实召回了内容时，至少留最高分那条 ——
    # 全丢会让"检索到了一点东西"变成"完全拒答"，通常更糟。
    if not relevant and candidates:
        relevant = [candidates[0].vector_id]
    return relevant


# --------------------------------------------------------------------------------------
# ④ rewrite：查询改写
# --------------------------------------------------------------------------------------
def rewrite_node(deps: AgentDeps) -> Any:
    def node(state: AgentState) -> dict[str, Any]:
        question = state.get("question", "")
        attempt = int(state.get("rewrite_round") or 0) + 1
        history = list(state.get("history") or [])

        emit(
            {"event": "trace", "node": "rewrite", "status": "start", "detail": {"attempt": attempt}}
        )
        started = _now()

        with node_span(deps, "rewrite", payload={"attempt": attempt}):
            messages = build_rewrite_messages(
                question=question,
                history_summary=_history_summary(history),
                attempt=attempt,
            )
            queries = [question]
            reason = ""
            try:
                result = deps.chat_model.complete(messages, temperature=0.0)
                obj = extract_json_object(result.text) or {}
                queries = _normalize_queries(obj.get("queries"), fallback=question)
                reason = str(obj.get("reason") or "")[:200]
                usage = _usage_dict(result.usage)
            except Exception as exc:  # noqa: BLE001
                logger.warning("agent.rewrite_failed", error=f"{type(exc).__name__}: {exc}"[:200])
                usage = {}

        duration = _now() - started
        emit(
            {
                "event": "trace",
                "node": "rewrite",
                "status": "ok",
                "duration_ms": duration,
                "detail": {"attempt": attempt, "queries": queries, "reason": reason},
            }
        )

        return {
            "queries": queries,
            "all_queries": queries,
            "rewrite_round": attempt,
            "rewrite_reason": reason,
            "usage": usage,
        }

    return node


# --------------------------------------------------------------------------------------
# ⑤ generate：生成答案
# --------------------------------------------------------------------------------------
def generate_node(deps: AgentDeps) -> Any:
    def node(state: AgentState) -> dict[str, Any]:
        question = state.get("question", "")
        needs_retrieval = bool(state.get("needs_retrieval", True))
        history = list(state.get("history") or [])
        reflect_instruction = state.get("reflect_instruction")

        emit({"event": "trace", "node": "generate", "status": "start"})
        started = _now()

        # ---------- 分支 1：不需要检索（闲聊 / 询问系统能力） ----------
        if not needs_retrieval:
            messages = build_direct_messages(question=question, history=history)
            answer, usage = _generate_text(deps, messages, emit_tokens=True)
            duration = _now() - started
            emit(
                {
                    "event": "trace",
                    "node": "generate",
                    "status": "ok",
                    "duration_ms": duration,
                    "detail": {"direct": True},
                }
            )
            emit({"event": "sources", "sources": []})
            return {
                "answer": answer,
                "refusal": False,
                "usage": usage,
                "generate_ms": duration,
                "model": deps.chat_model.model,
                "finished": True,
            }

        # ---------- 选出真正要喂给模型的候选 ----------
        contexts = select_contexts(state)

        # ---------- 分支 2：没有可用上下文 → 短路拒答 ----------
        # **这一步是这个项目最重要的"防幻觉"设计**：不给模型没有资料的机会。
        # 同时也省掉一次 token 花费。
        if not contexts:
            duration = _now() - started
            emit(
                {
                    "event": "trace",
                    "node": "generate",
                    "status": "ok",
                    "duration_ms": duration,
                    "detail": {"short_circuit": "no_context"},
                }
            )
            emit({"event": "sources", "sources": []})
            return {
                "answer": REFUSAL_ANSWER,
                "refusal": True,
                "usage": {},
                "generate_ms": duration,
                "model": deps.chat_model.model,
                "finished": True,
            }

        # ---------- 分支 3：正常 RAG 生成 ----------
        emit(
            {
                "event": "sources",
                "sources": [c.to_source_dict(i) for i, c in enumerate(contexts, start=1)],
            }
        )
        messages = build_rag_messages(
            question=question,
            contexts=[(i, c.source_label, c.context_text) for i, c in enumerate(contexts, start=1)],
            history=history,
            extra_instruction=reflect_instruction,
        )
        answer, usage = _generate_text(deps, messages, emit_tokens=True)
        duration = _now() - started
        refusal = detect_refusal(answer)

        emit(
            {
                "event": "trace",
                "node": "generate",
                "status": "ok",
                "duration_ms": duration,
                "detail": {
                    "contexts": len(contexts),
                    "refusal": refusal,
                    "reflect_retry": bool(reflect_instruction),
                },
            }
        )

        return {
            "answer": answer,
            "refusal": refusal,
            "usage": usage,
            "generate_ms": duration,
            "model": deps.chat_model.model,
            # 重试过一次就清掉额外指令，避免它一直被带进后续轮次
            "reflect_instruction": None,
            "finished": True,
        }

    return node


def _generate_text(
    deps: AgentDeps, messages: list[ChatMessage], *, emit_tokens: bool
) -> tuple[str, dict[str, int]]:
    """流式/非流式生成。流式运行时逐 token 推 SSE，同时累积完整文本。"""
    writer = _writer()
    if writer is None:
        with node_span(deps, "generate.llm", span_type=SPAN_LLM):
            result = deps.chat_model.complete(messages, temperature=deps.settings.llm_temperature)
        return result.text, _usage_dict(result.usage)

    pieces: list[str] = []
    usage = ChatUsage(model=deps.chat_model.model)
    try:
        with node_span(deps, "generate.llm", span_type=SPAN_LLM):
            for chunk in deps.chat_model.stream(
                messages, temperature=deps.settings.llm_temperature
            ):
                if chunk.text:
                    pieces.append(chunk.text)
                    if emit_tokens:
                        try:
                            writer({"event": "token", "text": chunk.text})
                        except Exception:  # noqa: BLE001 - 客户端断线不该中断生成
                            emit_tokens = False
                if chunk.usage is not None:
                    usage.merge(chunk.usage)
    except Exception as exc:  # noqa: BLE001
        # 已经流出去一部分了：不要把已经很长的答案丢掉。返回已生成的部分 + 标注。
        logger.warning("agent.generate_stream_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        partial = "".join(pieces)
        if partial:
            return partial + "\n\n（生成中断：模型服务异常，以上为已生成部分）", _usage_dict(usage)
        raise

    if usage.llm_calls == 0:
        usage.llm_calls = 1
    return "".join(pieces), _usage_dict(usage)


def select_contexts(state: AgentState) -> list[Candidate]:
    """按 `relevant_ids` 过滤候选（grade 判定为相关的优先），保持分数序。"""
    candidates = list(state.get("candidates") or [])
    if not candidates:
        return []
    relevant_ids = set(state.get("relevant_ids") or [])
    selected = [c for c in candidates if c.vector_id in relevant_ids] if relevant_ids else []
    if not selected:
        # grade 全判不相关时不留空：退回分数最高的前几条，
        # 让 generate 有机会基于它们回答（并在 reflect 阶段被检查引用）。
        selected = candidates[: max(1, min(3, len(candidates)))]
    selected.sort(key=lambda c: (-c.score, c.vector_id))
    return selected


# --------------------------------------------------------------------------------------
# ⑥ reflect：Self-RAG 引用支撑检查
# --------------------------------------------------------------------------------------
def reflect_node(deps: AgentDeps) -> Any:
    def node(state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        contexts = select_contexts(state)
        round_no = int(state.get("reflect_round") or 0) + 1

        emit(
            {"event": "trace", "node": "reflect", "status": "start", "detail": {"round": round_no}}
        )
        started = _now()

        passed = True
        unsupported: list[dict[str, Any]] = []

        with node_span(deps, "reflect", payload={"round": round_no}):
            messages = build_reflect_messages(
                answer=answer,
                contexts=[(i, c.context_text) for i, c in enumerate(contexts, start=1)],
            )
            try:
                with node_span(deps, "reflect.llm", span_type=SPAN_LLM):
                    result = deps.judge_model.complete(messages, temperature=0.0)
                obj = extract_json_object(result.text)
                if obj is None:
                    # 解析失败时**默认通过**：宁可漏掉一次纠正，
                    # 也不要因为判官输出格式问题让用户多等一轮生成。
                    passed = True
                    unsupported = []
                else:
                    raw_unsupported = obj.get("unsupported")
                    unsupported = (
                        [item for item in raw_unsupported if isinstance(item, dict)][:10]
                        if isinstance(raw_unsupported, list)
                        else []
                    )
                    declared = obj.get("passed")
                    passed = bool(declared) if isinstance(declared, bool) else not unsupported
                    if unsupported:
                        passed = False
                usage = _usage_dict(result.usage)
            except Exception as exc:  # noqa: BLE001
                logger.warning("agent.reflect_failed", error=f"{type(exc).__name__}: {exc}"[:200])
                passed = True
                usage = {}

        duration = _now() - started
        instruction: str | None = None
        if not passed and unsupported:
            # 把"错在哪"明确告诉模型，否则重试等于重新掷骰子。
            sentences = "；".join(str(item.get("sentence", ""))[:80] for item in unsupported[:3])
            instruction = (
                f"上一轮回答中以下句子缺少参考资料支撑：{sentences}。"
                "请重新作答：每一句事实性陈述都必须以 [n] 标注来源；"
                "如果某条信息在参考资料中确实不存在，请直接删除该句，不要改写或猜测。"
            )

        emit(
            {
                "event": "reflect",
                "round": round_no,
                "passed": passed,
                "unsupported": unsupported,
                "action": "regenerate" if instruction else "accept",
            }
        )
        emit(
            {
                "event": "trace",
                "node": "reflect",
                "status": "ok",
                "duration_ms": duration,
                "detail": {"round": round_no, "passed": passed, "unsupported": len(unsupported)},
            }
        )

        return {
            "reflect_round": round_no,
            "reflect_passed": passed,
            "unsupported": unsupported,
            "reflect_instruction": instruction,
            "usage": usage,
        }

    return node


# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------
def _now() -> int:
    import time

    return int(time.perf_counter() * 1000)


def _history_summary(history: Sequence[ChatMessage]) -> str | None:
    from knowflow.agent.memory import summarize_history

    summary = summarize_history(history)
    return summary or None


def _normalize_queries(raw: Any, *, fallback: str) -> list[str]:
    """把模型给的查询规整成去重、非空、最多 3 条的列表。永远不返回空。"""
    queries: list[str] = []
    if isinstance(raw, str):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = [str(item) for item in raw]
    else:
        candidates = []

    seen: set[str] = set()
    for item in candidates:
        text = " ".join(item.split()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        queries.append(text)
        if len(queries) >= 3:
            break

    if not queries:
        cleaned = " ".join(fallback.split()).strip()
        queries = [cleaned] if cleaned else []
    return queries


def _dedupe_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """按 `vector_id` 去重，保留分数更高的一条，按分数降序。"""
    by_id: dict[str, Candidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.vector_id)
        if existing is None or candidate.score > existing.score:
            by_id[candidate.vector_id] = candidate
    return sorted(by_id.values(), key=lambda c: (-c.score, c.vector_id))


__all__ = [
    "MAX_SEARCH_CALLS",
    "MAX_TOOL_ITERATIONS",
    "AgentDeps",
    "RecorderBox",
    "analyze_node",
    "emit",
    "generate_node",
    "grade_node",
    "node_span",
    "reflect_node",
    "retrieve_node",
    "rewrite_node",
    "select_contexts",
]
