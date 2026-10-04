"""对话模型的统一抽象。

**为什么是同步优先（sync-first）**：本项目的两个重组件 —— 本地 BGE-M3 推理与
Chroma 查询 —— 都是同步阻塞 API。如果对话模型走 async，就会出现
「async 路由里 await 模型，但检索却同步阻塞事件循环」的混合状态，
反而更容易卡死。所以选择：**内部全同步，在边界处交给线程池**
（Starlette 对同步生成器会自动用 `iterate_in_threadpool`）。

代价（诚实写在这里，面试主动说）：
- 一个流式问答会占住一个线程池线程直到回答结束，长连接多时线程是瓶颈；
- 要换 async 客户端（如 httpx.AsyncClient + async chroma）才能真正并发。

`ChatModel` 只暴露 2 个方法，让上层（RAG 链 / Agent 节点）不关心供应商：

    complete(messages) -> ChatResult          一次性拿完整回答（要用量、要判官）
    stream(messages)   -> Iterator[Chunk]     逐块拿（SSE 流式）

工具调用（tool calling）刻意**不放进这个协议**：Agent 的工具循环由
LangGraph 节点自己编排（见 `agent/`），模型层只负责"把消息变成文本"。
这样换供应商时，工具协议差异被隔离在 `openai_compat.py` 一个文件里。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    """一条对话消息。用简单 dataclass 而不是 langchain 的 BaseMessage，
    是为了让 `llm/` 层不把 langchain 的类型泄漏到业务代码里。"""

    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str | None = None
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: ChatUsage) -> None:
        """把一次调用的用量累加进来（Agent 一轮会调多次模型）。"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_calls += max(other.llm_calls, 1)
        if other.model:
            self.model = other.model

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "llm_calls": self.llm_calls,
        }


@dataclass(slots=True)
class ChatResult:
    text: str
    usage: ChatUsage
    finish_reason: str | None = None
    latency_ms: int = 0
    offline: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """流式增量块。

    `usage` 只在最后一个块里出现（OpenAI 协议里 usage 是收尾才给的），
    所以类型是可选。消费方必须处理"中间块没有 usage"这件事。
    """

    text: str = ""
    usage: ChatUsage | None = None
    finish_reason: str | None = None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolChatResult:
    """带工具调用的响应（`agent/` 用）。"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: ChatUsage = field(default_factory=ChatUsage)
    finish_reason: str | None = None
    latency_ms: int = 0
    offline: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """模型调用失败（网络、限流、鉴权）。上层转成 `UpstreamError`。"""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body[:500]


@runtime_checkable
class ChatModel(Protocol):
    """对话模型协议。`model` / `offline` 会被 `/health` 与指标读到。"""

    model: str
    offline: bool

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> ChatResult: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Iterator[Chunk]: ...

    def supports_tools(self) -> bool: ...


def messages_to_text(messages: Sequence[ChatMessage]) -> str:
    """把消息列表拍平成纯文本。

    用途：离线抽取式回答要在 prompt 里找上下文；日志与 trace 要记录"模型到底看到了什么"。
    注意这只是**展示/分析**用途，真正发给模型的一定是结构化消息列表。
    """
    return "\n\n".join(f"[{m.role}]\n{m.content}" for m in messages)


__all__ = [
    "ChatMessage",
    "ChatModel",
    "ChatResult",
    "ChatUsage",
    "Chunk",
    "LLMError",
    "Role",
    "ToolCall",
    "ToolChatResult",
    "messages_to_text",
]
