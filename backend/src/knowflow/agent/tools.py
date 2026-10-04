"""Agent 工具集。

**为什么 Agent 要有工具，而不是直接写死检索**：
固定链只能"检索一次、用同一批结果回答"。真实问题常常需要
"先查 A 再根据 A 的结果决定查 B"（多跳），或者根本不需要查（闲聊）。
把检索包装成工具后，**控制流由模型在图的结构内决定**。

工具设计的经验（这一条比实现更重要）：
**工具描述（description）决定调用准确率。**
写"搜索知识库"模型会乱调、乱传参；写成
"在企业知识库中做混合检索（语义+关键词）。当问题涉及公司制度、产品文档、
业务数据时使用；参数 query 应使用名词短语而不是整句问句"
——调用准确率会明显提升。所以本文件的 `description` 是**产品代码**，
不是注释，改它要像改 Prompt 一样谨慎。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from knowflow.core.logging import get_logger
from knowflow.retrieval.engine import HybridRetriever
from knowflow.retrieval.types import Candidate

logger = get_logger(__name__)

# 工具名常量：trace、SSE 事件、测试都引用它，不要写字面量
TOOL_KB_SEARCH = "kb_search"
TOOL_KB_STATS = "kb_stats"
TOOL_LIST_DOCUMENTS = "list_documents"


@dataclass(slots=True)
class ToolResult:
    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None

    def to_event(self) -> dict[str, Any]:
        """转成 SSE 的 `tool` 事件（契约 5.6）。**不暴露正文**，只给数量与摘要。"""
        return {
            "name": self.name,
            "args": self.arguments,
            "ok": self.ok,
            "result_count": len(self.candidates) or int(self.payload.get("count", 0) or 0),
            "summary": self.summary,
            "duration_ms": self.duration_ms,
        }

    def to_tool_message(self) -> str:
        """回灌给模型的工具结果文本。**必须有长度上限**：
        一次检索 5 条父块可能上万字，全塞回去会让下一轮 prompt 爆掉。"""
        if not self.ok:
            return f"工具 {self.name} 执行失败：{self.error}"
        if self.candidates:
            lines = [
                f"[{i}] {c.source_label}\n{c.context_text[:1200]}"
                for i, c in enumerate(self.candidates, start=1)
            ]
            body = "\n\n".join(lines)
            return f"工具 {self.name} 返回 {len(self.candidates)} 条结果：\n\n{body}"
        return f"工具 {self.name} 返回：{self.summary}"


class ToolRegistry:
    """工具注册表：负责"告诉模型有哪些工具"和"真正执行"。"""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        kb_stats: Callable[[], dict[str, Any]] | None = None,
        list_documents: Callable[[str | None], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.retriever = retriever
        self._kb_stats = kb_stats
        self._list_documents = list_documents

    # ------------------------------------------------------------------ schema
    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling 格式的工具声明。"""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": TOOL_KB_SEARCH,
                    "description": (
                        "在企业知识库中做混合检索（语义向量 + 关键词 BM25），返回带出处的原文片段。"
                        "当问题涉及公司制度、流程规范、产品文档、业务数据等私有信息时必须使用。"
                        "query 用名词短语效果最好（例如「一线城市 住宿标准」），"
                        "不要直接传整句口语问句。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索用的关键词短语"},
                            "top_k": {
                                "type": "integer",
                                "description": "返回条数，1-10，默认 5",
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]
        if self._kb_stats is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_KB_STATS,
                        "description": (
                            "查看当前知识库的规模（文档数、切片数、向量数）。"
                            "当用户问「知识库里有什么资料/有多少文档」，"
                            "或你需要判断检索不到是不是因为库是空的时候使用。"
                        ),
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )
        if self._list_documents is not None:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_LIST_DOCUMENTS,
                        "description": (
                            "列出知识库中的文档文件名。当用户问「有哪些文档/有没有关于 X 的文档」时使用。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "可选的文件名关键词过滤，不传则列出全部",
                                }
                            },
                        },
                    },
                }
            )
        return tools

    def names(self) -> list[str]:
        return [t["function"]["name"] for t in self.schemas()]

    # ------------------------------------------------------------------ 执行
    def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        kb_ids: Sequence[int],
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> ToolResult:
        started = time.perf_counter()
        try:
            if name == TOOL_KB_SEARCH:
                result = self._kb_search(
                    arguments, kb_ids=kb_ids, top_k=top_k, use_rerank=use_rerank
                )
            elif name == TOOL_KB_STATS:
                result = self._kb_stats_tool()
            elif name == TOOL_LIST_DOCUMENTS:
                result = self._list_documents_tool(arguments)
            else:
                result = ToolResult(
                    name=name,
                    arguments=arguments,
                    ok=False,
                    summary=f"未知工具 {name}",
                    error=f"未知工具: {name}",
                )
        except Exception as exc:  # noqa: BLE001 - 工具失败不能中断整个 Agent
            logger.warning("tool.failed", tool=name, error=f"{type(exc).__name__}: {exc}"[:200])
            result = ToolResult(
                name=name,
                arguments=arguments,
                ok=False,
                summary="工具执行失败",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _kb_search(
        self,
        arguments: dict[str, Any],
        *,
        kb_ids: Sequence[int],
        top_k: int,
        use_rerank: bool,
    ) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(
                name=TOOL_KB_SEARCH,
                arguments=arguments,
                ok=False,
                summary="query 为空",
                error="query 参数不能为空",
            )
        requested = arguments.get("top_k")
        k = int(requested) if isinstance(requested, int) else top_k
        k = max(1, min(10, k))

        search = self.retriever.search(
            query=query,
            kb_ids=list(kb_ids) or None,
            kb_id=kb_ids[0] if len(kb_ids) == 1 else None,
            top_k=k,
            use_rerank=use_rerank,
        )
        passed = search.passed_gate
        candidates = search.candidates if passed else []
        if passed:
            gate_text = "通过"
        elif search.gate is not None:
            gate_text = f"未通过: {search.gate.reason}"
        else:
            gate_text = "未知"
        summary = f"命中 {len(candidates)} 条（闸门 {gate_text}）"
        return ToolResult(
            name=TOOL_KB_SEARCH,
            arguments={"query": query, "top_k": k},
            ok=True,
            summary=summary,
            payload={
                "count": len(candidates),
                "gate": search.gate.to_dict() if search.gate else None,
                "debug": search.debug.to_dict(),
                "latency_ms": search.total_ms,
            },
            candidates=candidates,
        )

    def _kb_stats_tool(self) -> ToolResult:
        if self._kb_stats is None:
            return ToolResult(
                name=TOOL_KB_STATS,
                arguments={},
                ok=False,
                summary="未提供知识库统计能力",
                error="kb_stats callback 未注入",
            )
        stats = self._kb_stats()
        return ToolResult(
            name=TOOL_KB_STATS,
            arguments={},
            ok=True,
            summary=f"{stats.get('doc_count', 0)} 篇文档 / {stats.get('chunk_count', 0)} 个切片",
            payload={"count": int(stats.get("doc_count", 0) or 0), **stats},
        )

    def _list_documents_tool(self, arguments: dict[str, Any]) -> ToolResult:
        if self._list_documents is None:
            return ToolResult(
                name=TOOL_LIST_DOCUMENTS,
                arguments=arguments,
                ok=False,
                summary="未提供文档列表能力",
                error="list_documents callback 未注入",
            )
        keyword = arguments.get("keyword")
        keyword_str = str(keyword).strip() if keyword else None
        docs = self._list_documents(keyword_str)
        names = ", ".join(str(d.get("filename", "?")) for d in docs[:20])
        return ToolResult(
            name=TOOL_LIST_DOCUMENTS,
            arguments=arguments,
            ok=True,
            summary=f"共 {len(docs)} 篇：{names}" if names else "没有文档",
            payload={"count": len(docs), "documents": docs[:50]},
        )


__all__ = [
    "TOOL_KB_SEARCH",
    "TOOL_KB_STATS",
    "TOOL_LIST_DOCUMENTS",
    "ToolRegistry",
    "ToolResult",
]
