"""真实模型接入：LangChain + OpenAI 兼容协议。

**为什么走 OpenAI 兼容协议而不是某个厂商的 SDK**：换供应商只改
`OPENAI_BASE_URL`（DeepSeek / 通义 / 智谱 / vLLM / Ollama 都提供兼容端点），
业务代码与 Prompt 一行不动。代价是拿不到厂商独有参数。

这个类要处理四件麻烦事，每件都有真实背景：

1. **token 用量有两种来源**：新版在 `usage_metadata`，老/第三方网关只在
   `response_metadata["token_usage"]`。只读一个会导致成本统计全是 0。
   本项目**两个都读，优先新版**。
2. **流式默认不返回 usage**：OpenAI 协议里要显式传 `stream_options={"include_usage": true}`。
   LangChain 把它封装成 `stream_usage=True`，不开的话流式请求的成本统计永远为 0。
3. **`content` 可能是字符串也可能是块列表**（多模态/推理模型会返回 list）。
   统一走 `_extract_text`，否则 `answer` 字段会变成一坨 dict。
4. **重试要指数退避且只重试可重试的错误**：429/5xx/超时重试，
   401/400 立刻失败（重试只是浪费用户 10 秒）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.llm.base import (
    ChatMessage,
    ChatResult,
    ChatUsage,
    Chunk,
    LLMError,
    ToolCall,
    ToolChatResult,
)

logger = get_logger(__name__)

# 可重试的 HTTP 状态码：限流与服务端临时故障
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _to_lc_messages(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    mapping = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
        "tool": ToolMessage,
    }
    out: list[BaseMessage] = []
    for msg in messages:
        cls = mapping.get(msg.role)
        if cls is None:
            raise LLMError(f"未知的消息角色: {msg.role!r}")
        if cls is ToolMessage:
            # ToolMessage 必须有 tool_call_id，这里用不到工具消息回灌，直接降级成 HumanMessage
            out.append(HumanMessage(content=msg.content))
        else:
            out.append(cls(content=msg.content))
    return out


def _extract_text(message: Any) -> str:
    """从 LangChain 消息里取纯文本。

    优先 `.text`（langchain-core 1.x 的官方途径，会正确处理块列表）；
    失败则手工拼 content 里 type 为 text 的块；再失败就 str()。
    **绝不能因为拿不到文本就抛异常**——那会让一次成功的模型调用变成 500。
    """
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _usage_from_message(message: Any, model: str) -> ChatUsage:
    """两个来源都读，优先新版 `usage_metadata`。"""
    usage = ChatUsage(model=model, llm_calls=1)

    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict):
        usage.prompt_tokens = int(meta.get("input_tokens") or 0)
        usage.completion_tokens = int(meta.get("output_tokens") or 0)

    if usage.total_tokens == 0:
        response_meta = getattr(message, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage") or response_meta.get("usage") or {}
        if isinstance(token_usage, dict):
            usage.prompt_tokens = int(
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            )
            usage.completion_tokens = int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )
    return usage


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "ratelimit", "apiconnection"))


class OpenAIChatModel:
    """实现了 `ChatModel` 协议（见 `llm/base.py`）。"""

    offline = False

    def __init__(self, settings: Settings | None = None, *, model: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.openai_model
        self._client = ChatOpenAI(
            model=self.model,
            # langchain-openai 要求 SecretStr：直接传 str 在运行期也能工作，
            # 但类型上不合法，而且 SecretStr 能防止 Key 被 repr/日志打印出来。
            api_key=SecretStr(self.settings.openai_api_key),
            base_url=self.settings.openai_base_url,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
            max_retries=0,  # 重试由本类自己控制，便于区分可重试/不可重试
            stream_usage=True,  # ★ 不开这个，流式请求的成本统计永远是 0
        )

    # ------------------------------------------------------------------ 内部
    def _invoke(
        self, messages: Sequence[ChatMessage], *, temperature: float | None, max_tokens: int | None
    ) -> Any:
        payload = _to_lc_messages(messages)
        # `.bind()` 返回的是 RunnableBinding 而不是 ChatOpenAI，所以这里不能标注成 ChatOpenAI
        bound: Any = self._client
        if temperature is not None or max_tokens is not None:
            kwargs: dict[str, Any] = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            bound = self._client.bind(**kwargs)

        attempts = self.settings.llm_max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return bound.invoke(payload)
            except Exception as exc:  # noqa: BLE001 - 统一转成 LLMError
                last_exc = exc
                if not _is_retryable(exc) or attempt == attempts:
                    break
                delay = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "llm.retry",
                    attempt=attempt,
                    max_attempts=attempts,
                    delay_s=delay,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
                time.sleep(delay)
        raise LLMError(
            f"模型调用失败: {type(last_exc).__name__}: {last_exc}",
            status_code=getattr(last_exc, "status_code", None),
            body=str(getattr(last_exc, "response", "") or "")[:500],
        ) from last_exc

    # ------------------------------------------------------------------ 协议
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        message = self._invoke(messages, temperature=temperature, max_tokens=max_tokens)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ChatResult(
            text=_extract_text(message),
            usage=_usage_from_message(message, self.model),
            finish_reason=(getattr(message, "response_metadata", None) or {}).get("finish_reason"),
            latency_ms=elapsed_ms,
            offline=False,
        )

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Iterator[Chunk]:
        payload = _to_lc_messages(messages)
        bound: Any = self._client
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if kwargs:
            bound = self._client.bind(**kwargs)

        try:
            for lc_chunk in bound.stream(payload):
                text = _extract_text(lc_chunk)
                usage = None
                if getattr(lc_chunk, "usage_metadata", None):
                    usage = _usage_from_message(lc_chunk, self.model)
                finish = (getattr(lc_chunk, "response_metadata", None) or {}).get("finish_reason")
                if text or usage or finish:
                    yield Chunk(text=text, usage=usage, finish_reason=finish)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"流式调用失败: {type(exc).__name__}: {exc}") from exc

    def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict[str, Any]],
        temperature: float | None = None,
    ) -> ToolChatResult:
        """让模型自行决定是否调用工具（OpenAI function calling）。

        返回的 `tool_calls` 由 `agent/nodes.py` 执行；执行结果以
        `tool` 角色消息回灌后再次调用本方法，直到模型不再要求调工具。
        """
        payload = _to_lc_messages(messages)
        llm = self._client.bind_tools(list(tools))
        if temperature is not None:
            llm = llm.bind(temperature=temperature)

        started = time.perf_counter()
        try:
            message = llm.invoke(payload)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"工具调用失败: {type(exc).__name__}: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or []:
            args = raw.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call_{len(calls)}"),
                    name=str(raw.get("name") or ""),
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        return ToolChatResult(
            text=_extract_text(message),
            tool_calls=calls,
            usage=_usage_from_message(message, self.model),
            finish_reason=(getattr(message, "response_metadata", None) or {}).get("finish_reason"),
            latency_ms=elapsed_ms,
            offline=False,
        )

    def supports_tools(self) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {
            "offline": False,
            "model": self.model,
            "base_url": self.settings.openai_base_url,
            "supports_tools": True,
            "temperature": self.settings.llm_temperature,
        }


__all__ = ["OpenAIChatModel"]
