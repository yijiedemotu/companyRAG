"""Agent 状态定义（LangGraph 的 State）。

**这份定义就是"图的形状"**，读它比读 `graph.py` 更快看清 Agent 能做什么。

三个关键设计：

1. **reducer 决定"节点返回值是覆盖还是累加"**
   - `queries` 用默认（覆盖）：每轮改写只关心"这一轮用什么查"；
   - `all_queries` 用 `operator.add`（累加）：trace 与调试要看完整的查询历史；
   - `candidates` 用自定义 reducer（按 `vector_id` 去重取高分）：
     多轮检索会重复捞到同一 chunk，累加会让 prompt 里出现重复引用 `[3]` 和 `[7]`
     指向同一段原文 —— 既浪费 token 又让引用编号失去意义。

2. **刻意不做"过滤式"写入**：`grade` 节点**不**返回过滤后的 candidates，
   而是返回 `relevant_ids`，由 `generate` 去筛。
   原因：如果用 reducer 累加，过滤结果会被下一轮的累加又"合并回来"
   （reducer 是 `left + right`，你没法通过返回更少的元素来删东西）。
   这是 LangGraph 最容易踩的坑之一，写在注释里避免后人重犯。

3. **usage 用合并 reducer**：generate 可能被 reflect 打回重试，
   两次调用的 token 都要计入成本，不能覆盖。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from knowflow.llm.base import ChatMessage
from knowflow.retrieval.types import Candidate


def merge_candidates(
    left: list[Candidate] | None, right: list[Candidate] | None
) -> list[Candidate]:
    """按 `vector_id` 去重合并，保留分数更高的那条，按分数降序返回。"""
    by_id: dict[str, Candidate] = {}
    for candidate in list(left or []) + list(right or []):
        existing = by_id.get(candidate.vector_id)
        if existing is None or candidate.score > existing.score:
            by_id[candidate.vector_id] = candidate
    return sorted(by_id.values(), key=lambda c: (-c.score, c.vector_id))


def merge_usage(left: dict[str, int] | None, right: dict[str, int] | None) -> dict[str, int]:
    """token 用量累加（值都是 int，直接相加即可）。"""
    merged: dict[str, int] = dict(left or {})
    for key, value in (right or {}).items():
        if isinstance(value, int):
            merged[key] = merged.get(key, 0) + value
        else:
            merged[key] = value
    return merged


class AgentState(TypedDict, total=False):
    """LangGraph 的状态。`total=False` 表示节点可以只返回自己改动的字段。"""

    # ---------------- 输入（整个流程不变） ----------------
    question: str
    kb_id: int | None
    kb_ids: list[int]
    top_k: int
    use_rerank: bool
    history: list[ChatMessage]
    mode: str

    # ---------------- analyze ----------------
    needs_retrieval: bool
    question_type: str
    analyze_reason: str
    queries: list[str]  # 本轮要检索的查询（覆盖语义）
    all_queries: Annotated[list[str], operator.add]  # 历史所有查询（累加语义）

    # ---------------- retrieve（含模型驱动的工具循环） ----------------
    candidates: Annotated[list[Candidate], merge_candidates]
    retrieval_rounds: int
    retrieval_ms: int
    vector_error: str | None
    gate_passed: bool
    gate_reason: str
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    rewrite_round: int
    rewrite_reason: str

    # ---------------- grade（CRAG） ----------------
    grading: Annotated[list[dict[str, Any]], operator.add]
    relevant_ids: list[str]
    grading_skipped: bool

    # ---------------- generate ----------------
    answer: str
    refusal: bool
    usage: Annotated[dict[str, int], merge_usage]
    generate_ms: int
    reflect_instruction: str | None
    model: str

    # ---------------- reflect（Self-RAG） ----------------
    reflect_round: int
    reflect_passed: bool
    unsupported: list[dict[str, Any]]

    # ---------------- 收尾 ----------------
    errors: Annotated[list[str], operator.add]
    finished: bool


__all__ = ["AgentState", "merge_candidates", "merge_usage"]
