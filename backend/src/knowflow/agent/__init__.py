"""Agent 层：LangGraph 状态图 + 工具 + 会话记忆。

这也是整个项目最值得在面试里讲的一层，一句话概括：

> **"我没有把控制流交给模型，我把控制流画成了图；模型只在每个节点内部做判断。
> 这样既有自适应的好处，又保证每一步可观测、可兜底、可测试。"**
"""

from knowflow.agent.graph import build_agent_graph, build_checkpointer, build_fixed_chain
from knowflow.agent.memory import load_history, truncate_history
from knowflow.agent.nodes import AgentDeps
from knowflow.agent.runner import AgentOutcome, AgentRunner
from knowflow.agent.state import AgentState
from knowflow.agent.tools import ToolRegistry, ToolResult

__all__ = [
    "AgentDeps",
    "AgentOutcome",
    "AgentRunner",
    "AgentState",
    "ToolRegistry",
    "ToolResult",
    "build_agent_graph",
    "build_checkpointer",
    "build_fixed_chain",
    "load_history",
    "truncate_history",
]
