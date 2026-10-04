"""LangGraph 状态图装配。

**这张图就是"控制流"本身**，读它比读任何文档都快：

    START → analyze ──needs_retrieval?──┬─ no ──────────────────────► generate ──┐
                                        └─ yes ─► retrieve ─► grade ──┬─ 不足 ─► rewrite ─┐
                                                                     │                  │
                                                                     └─ 足够 ─► generate ◄┘
                                                                                   │
                                            ┌──────────────────────────────────────┘
                                            ▼
                                         reflect ──不通过且还有重试次数──► generate
                                            │
                                            └─ 通过 ─► END

**每个条件边的判据都写成小函数**（`route_after_*`），而不是内联 lambda：
条件边是"业务规则"，需要被单独测试（"give 一个 gate 没过且还有改写次数的 state，
应该走 rewrite"）。内联 lambda 没法测。

**关于 checkpointer（重要决定）**：
- `thread_id` 用 **trace_id**（一次请求一个 thread），不是 conversation_id。

为什么不是 conversation_id？因为 `AgentState` 里的 `candidates` / `all_queries`
用的是**累加 reducer**。如果同一会话共用一个 thread，第二轮的状态会与第一轮
合并 —— 上一轮的候选片段会混进这一轮的上下文，引用编号也全乱。
而"多轮记忆"这件事本项目已经有更可控的实现：**从 MySQL 读消息 + 双重截断**
（见 `agent/memory.py`），那才是产品语义上的记忆。

那 checkpointer 还留着做什么？**可回溯**：每个 trace 的图状态被持久化，
排障时能看到"那次请求在图里走到了哪一步、每个节点的中间产出是什么"。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.graph import END, START, StateGraph

from knowflow.agent.nodes import (
    AgentDeps,
    analyze_node,
    generate_node,
    grade_node,
    reflect_node,
    retrieve_node,
    rewrite_node,
)
from knowflow.agent.state import AgentState
from knowflow.core.config import Settings
from knowflow.core.logging import get_logger

logger = get_logger(__name__)

# 条件边的目标名（用常量避免拼写错误导致 LangGraph 运行期才报错）
NODE_ANALYZE = "analyze"
NODE_RETRIEVE = "retrieve"
NODE_GRADE = "grade"
NODE_REWRITE = "rewrite"
NODE_GENERATE = "generate"
NODE_REFLECT = "reflect"


def route_after_analyze(state: AgentState, cfg: Settings) -> str:
    """不需要检索就直接生成（闲聊 / 询问系统能力）。"""
    if state.get("finished"):
        return END
    return NODE_RETRIEVE if state.get("needs_retrieval", True) else NODE_GENERATE


def route_after_grade(state: AgentState, cfg: Settings) -> str:
    """召回不足且还有改写次数 → 改写查询再查一次；否则去生成（可能是拒答）。"""
    if state.get("finished"):
        return END
    if not state.get("candidates"):
        return NODE_GENERATE
    round_no = int(state.get("rewrite_round") or 0)
    if round_no >= cfg.agent_max_rewrite:
        return NODE_GENERATE
    if not state.get("gate_passed", False):
        return NODE_REWRITE
    if not state.get("relevant_ids"):
        return NODE_REWRITE
    return NODE_GENERATE


def route_after_generate(state: AgentState, cfg: Settings) -> str:
    """只有"确实检索过 + 有上下文 + 没拒答 + 还有反思次数"才去反思。

    注意几个跳过条件都是有理由的：
    - 直答（没检索）：没有引用可校验；
    - 拒答：本来就没编东西，再校验一次纯浪费；
    - 没有候选：reflect 拿不到参考资料，判定必然不可靠。
    """
    if state.get("finished") and not state.get("answer"):
        return END
    if not state.get("needs_retrieval", True):
        return END
    if state.get("refusal"):
        return END
    if cfg.agent_max_reflect <= 0:
        return END
    if not state.get("candidates"):
        return END
    if int(state.get("reflect_round") or 0) >= cfg.agent_max_reflect:
        return END
    return NODE_REFLECT


def route_after_reflect(state: AgentState, cfg: Settings) -> str:
    """反思不通过就带着"哪里缺引用"再生成一次；次数用完就接受现状。"""
    if state.get("reflect_passed", True):
        return END
    if int(state.get("reflect_round") or 0) > cfg.agent_max_reflect:
        return END
    return NODE_GENERATE


def build_checkpointer(cfg: Settings) -> Any | None:
    """按配置构造 checkpointer。

    `sqlite` 用文件持久化（服务重启后仍能回溯历史 trace 的图状态）；
    `memory` 只在进程内。任何构造失败都返回 None（**不阻断服务启动**：
    可观测性坏了不该让问答不可用）。
    """
    backend = cfg.agent_checkpoint_backend
    try:
        if backend == "sqlite":
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            from langgraph.checkpoint.sqlite import SqliteSaver

            # check_same_thread=False：LangGraph 可能在别的线程里读写 checkpoint
            conn = sqlite3.connect(str(cfg.checkpoint_path), check_same_thread=False)
            saver = SqliteSaver(conn)
            saver.setup()
            logger.info("agent.checkpointer", backend="sqlite", path=str(cfg.checkpoint_path))
            return saver
        from langgraph.checkpoint.memory import InMemorySaver

        logger.info("agent.checkpointer", backend="memory")
        return InMemorySaver()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "agent.checkpointer_failed", backend=backend, error=f"{type(exc).__name__}: {exc}"[:200]
        )
        return None


def build_agent_graph(deps: AgentDeps, *, checkpointer: Any | None = None) -> Any:
    """装配并编译图。返回 LangGraph 的 CompiledStateGraph。"""
    cfg = deps.settings
    builder: StateGraph[AgentState] = StateGraph(AgentState)

    builder.add_node(NODE_ANALYZE, analyze_node(deps))
    builder.add_node(NODE_RETRIEVE, retrieve_node(deps))
    builder.add_node(NODE_GRADE, grade_node(deps))
    builder.add_node(NODE_REWRITE, rewrite_node(deps))
    builder.add_node(NODE_GENERATE, generate_node(deps))
    builder.add_node(NODE_REFLECT, reflect_node(deps))

    builder.add_edge(START, NODE_ANALYZE)

    # 条件边用 lambda 把 cfg 注入判据函数（判据本身是独立可测的纯函数）
    builder.add_conditional_edges(
        NODE_ANALYZE,
        lambda state: route_after_analyze(state, cfg),
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_GENERATE: NODE_GENERATE, END: END},
    )
    builder.add_edge(NODE_RETRIEVE, NODE_GRADE)
    builder.add_conditional_edges(
        NODE_GRADE,
        lambda state: route_after_grade(state, cfg),
        {NODE_REWRITE: NODE_REWRITE, NODE_GENERATE: NODE_GENERATE, END: END},
    )
    builder.add_edge(NODE_REWRITE, NODE_RETRIEVE)
    builder.add_conditional_edges(
        NODE_GENERATE,
        lambda state: route_after_generate(state, cfg),
        {NODE_REFLECT: NODE_REFLECT, END: END},
    )
    builder.add_conditional_edges(
        NODE_REFLECT,
        lambda state: route_after_reflect(state, cfg),
        {NODE_GENERATE: NODE_GENERATE, END: END},
    )

    return builder.compile(checkpointer=checkpointer, name="knowflow_agent")


def build_fixed_chain(deps: AgentDeps) -> Any:
    """ "快路径"：只有 analyze → retrieve → generate，不做 grade/rewrite/reflect。

    存在的意义是**延迟对比**：评测里用 `rag` 模式跑一遍，能得到
    "自我纠正换来了多少准确率、多花了多少毫秒"的真实数字。
    这也是面试里回答"你的 Agent 比固定链好在哪"的依据。
    """
    cfg = deps.settings
    builder: StateGraph[AgentState] = StateGraph(AgentState)
    builder.add_node(NODE_ANALYZE, analyze_node(deps))
    builder.add_node(NODE_RETRIEVE, retrieve_node(deps))
    builder.add_node(NODE_GENERATE, generate_node(deps))
    builder.add_edge(START, NODE_ANALYZE)
    builder.add_conditional_edges(
        NODE_ANALYZE,
        lambda state: route_after_analyze(state, cfg),
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_GENERATE: NODE_GENERATE, END: END},
    )
    builder.add_edge(NODE_RETRIEVE, NODE_GENERATE)
    builder.add_edge(NODE_GENERATE, END)
    return builder.compile(name="knowflow_fixed_chain")


__all__ = [
    "NODE_ANALYZE",
    "NODE_GENERATE",
    "NODE_GRADE",
    "NODE_REFLECT",
    "NODE_RETRIEVE",
    "NODE_REWRITE",
    "build_agent_graph",
    "build_checkpointer",
    "build_fixed_chain",
    "route_after_analyze",
    "route_after_generate",
    "route_after_grade",
    "route_after_reflect",
]
