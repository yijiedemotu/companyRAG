"""检索层的数据结构。

一个关键概念：**候选（Candidate）在整条检索链上被逐步"加工"**，
每一路信号都保留在自己的字段里，而不是覆盖成一个总分。原因：

- 调试时（`/search` 接口）要能回答"这条为什么排第 3"——只留总分就说不清；
- 评测时要能对比"纯向量 vs 纯 BM25 vs 混合"，需要各路分数分离；
- 重排（rerank）是在融合结果之上再算一次，不能丢掉融合分。

所以字段分三段：**原始召回分** → **融合分** → **最终分**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class ChunkInfo:
    """MySQL 里 chunk 的权威信息。

    **为什么检索结果要回查 MySQL，而不是全信向量库的 metadata**：
    向量库是"索引"，MySQL 是"事实来源"。两者可能不一致（写入中途失败、
    手改过库、重建索引）。引用必须指向真实存在的 chunk，否则前端点开就是 404。
    所以约定：**向量库只用来找"哪些候选"，正文与出处一律以 MySQL 为准。**
    """

    vector_id: str
    chunk_id: int
    doc_id: int
    doc_name: str
    chunk_index: int
    content: str
    context_text: str
    page_no: int | None = None
    section_path: str | None = None
    ext: str | None = None


class ChunkEnricher(Protocol):
    """由 service 层实现（它才知道怎么查库）。

    引擎依赖这个协议而不是直接 import SQLAlchemy —— 保持检索层可单测、
    不依赖数据库（这是测试能跑得又快又稳的关键）。
    """

    def fetch(self, vector_ids: Sequence[str]) -> dict[str, ChunkInfo]: ...


@dataclass(slots=True)
class Candidate:
    """一个候选切片，贯穿召回 → 融合 → 重排 → 闸门 → 组装上下文。"""

    # ---- 身份 ----
    vector_id: str
    doc_id: int
    doc_name: str
    chunk_id: int | None = None
    chunk_index: int = 0

    # ---- 文本 ----
    content: str = ""  # 子块：用于展示、BM25、引用片段
    context_text: str = ""  # 父块：用于喂给模型（为空则退回 content）

    # ---- 元信息（引用可溯源的关键）----
    page_no: int | None = None
    section_path: str | None = None
    ext: str | None = None

    # ---- 各路原始信号 ----
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None

    # ---- 加工后的分数 ----
    fused_score: float = 0.0
    rerank_score: float | None = None
    keyword_coverage: float = 0.0
    score: float = 0.0  # 最终排序分

    def __post_init__(self) -> None:
        if not self.context_text:
            self.context_text = self.content

    @property
    def source_label(self) -> str:
        """给模型看的出处描述，例如 `员工报销制度.md > 差旅报销标准 > 第3页`。

        把三级信息拼成一行塞进 Prompt —— 模型看到小节路径后引用准确率明显更高。
        """
        parts: list[str] = [self.doc_name or f"文档{self.doc_id}"]
        if self.section_path:
            parts.append(self.section_path)
        if self.page_no is not None:
            parts.append(f"第{self.page_no}页")
        return " > ".join(parts)

    @property
    def snippet(self) -> str:
        """引用卡片上显示的短片段（最长 200 字）。"""
        text = " ".join(self.content.split())
        return text[:200] + ("…" if len(text) > 200 else "")

    def to_hit_dict(self, rank: int) -> dict[str, Any]:
        """转成 `/search` 接口的 hit 结构（字段名见契约 5.4）。"""
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_no": self.page_no,
            "section_path": self.section_path,
            "content": self.content,
            "snippet": self.snippet,
            "score": round(self.score, 6),
            "vector_score": _round(self.vector_score),
            "bm25_score": _round(self.bm25_score),
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": _round(self.rerank_score),
            "keyword_coverage": round(self.keyword_coverage, 6),
        }

    def to_source_dict(self, rank: int) -> dict[str, Any]:
        """转成 `ChatResponse.sources` 的结构（契约 5.5）。"""
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_no": self.page_no,
            "section_path": self.section_path,
            "score": round(self.score, 6),
            "snippet": self.snippet,
        }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(slots=True)
class GateResult:
    """相关性闸门判定。

    `passed=False` 时上层**不调大模型**，直接返回拒答 —— 既省钱又防幻觉。
    `reason` 是稳定字符串（不是给人看的句子），前端/测试按它分支。
    """

    passed: bool
    reason: str
    vector_threshold: float
    keyword_threshold: float
    best_vector_score: float
    best_keyword_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "vector_threshold": self.vector_threshold,
            "keyword_threshold": self.keyword_threshold,
            "best_vector_score": round(self.best_vector_score, 6),
            "best_keyword_coverage": round(self.best_keyword_coverage, 6),
        }


@dataclass(slots=True)
class RetrievalDebug:
    """各阶段残留数量，用于回答"为什么没捞到"。"""

    vector_candidates: int = 0
    bm25_candidates: int = 0
    fused: int = 0
    after_autocut: int = 0
    after_rerank: int = 0
    dropped_by_gate: int = 0
    dropped_by_budget: int = 0
    # 孤儿向量：向量库里有、但 MySQL 里已没有对应 chunk 行。
    # 必须丢弃——否则引用会指向不存在的 chunk，前端点开就是 404。
    dropped_orphans: int = 0
    # 向量化失败时的原因（降级成纯 BM25 继续跑，但必须可见）
    vector_error: str | None = None
    # 检索用到的 KB 列表（多库合并检索时前端要能看出来）
    kb_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_candidates": self.vector_candidates,
            "bm25_candidates": self.bm25_candidates,
            "fused": self.fused,
            "after_autocut": self.after_autocut,
            "after_rerank": self.after_rerank,
            "dropped_by_gate": self.dropped_by_gate,
            "dropped_by_budget": self.dropped_by_budget,
            "dropped_orphans": self.dropped_orphans,
            "vector_error": self.vector_error,
            "kb_ids": self.kb_ids,
        }


@dataclass(slots=True)
class RetrievalResult:
    """一次检索的完整产出。"""

    query: str
    mode: str
    candidates: list[Candidate] = field(default_factory=list)
    gate: GateResult | None = None
    debug: RetrievalDebug = field(default_factory=RetrievalDebug)

    # 分段耗时（毫秒）——这条数据是"检索慢在哪"的唯一依据
    embedding_ms: int = 0
    vector_ms: int = 0
    bm25_ms: int = 0
    fusion_ms: int = 0
    rerank_ms: int = 0
    total_ms: int = 0
    rewritten_from: str | None = None

    @property
    def passed_gate(self) -> bool:
        return bool(self.gate and self.gate.passed)

    def sources(self) -> list[dict[str, Any]]:
        return [c.to_source_dict(i) for i, c in enumerate(self.candidates, start=1)]

    def context_pairs(self) -> list[tuple[int, str, str]]:
        """给 `prompts.format_contexts` 用的 `(编号, 出处, 正文)`。"""
        return [(i, c.source_label, c.context_text) for i, c in enumerate(self.candidates, start=1)]

    def hits(self) -> list[dict[str, Any]]:
        return [c.to_hit_dict(i) for i, c in enumerate(self.candidates, start=1)]


__all__ = [
    "Candidate",
    "ChunkEnricher",
    "ChunkInfo",
    "GateResult",
    "RetrievalDebug",
    "RetrievalResult",
]
