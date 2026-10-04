"""分数断崖截断（autocut）与相关性闸门（gate）。

**两个不同的东西，容易混为一谈：**

**1. autocut —— "结果里只有前几条是相关的"**
    检索总会返回 `top_k` 条，即使第 4、5 条其实是噪声。
    把噪声塞进 Prompt 有两个害处：占 token（花钱）、干扰模型（更容易编）。
    autocut 的做法：看分数**断崖**——排序后如果第 n 条的分数掉到第 1 条的
    某个比例以下，就把它和后面的一起丢掉。
    这是"自适应 top_k"：简单问题可能只留 1 条，复杂问题留 5 条。

**2. gate —— "这一批结果全都不相关"**
    双阈值：`向量相似度 ≥ 阈值` **或** `查询词覆盖率 ≥ 阈值`。
    两条都不满足 → **不调大模型**，直接拒答。
    为什么用"或"而不是"与"：两条信号覆盖不同的失败模式 ——
    向量强在语义改写（"房费"↔"住宿标准"），覆盖率强在专有名词
    （"PAYLOAD_TOO_LARGE"）。用"与"会把两种正确情况都误杀。

**为什么 gate 比 autocut 更重要**：autocut 省的是钱，gate 省的是**幻觉**。
模型手里没有资料时最容易编，而工程上"确保它手里有对的资料"比
"让它别编"有效得多。
"""

from __future__ import annotations

from typing import Final

from knowflow.core.logging import get_logger
from knowflow.retrieval.types import Candidate, GateResult

logger = get_logger(__name__)

# gate 判定原因的稳定标识（前端与测试按它分支，不要改成给人看的句子）
REASON_VECTOR: Final[str] = "vector_score_above_threshold"
REASON_KEYWORD: Final[str] = "keyword_coverage_above_threshold"
REASON_NO_CANDIDATES: Final[str] = "no_candidates"
REASON_BELOW_THRESHOLDS: Final[str] = "below_all_thresholds"


def autocut(
    candidates: list[Candidate],
    *,
    ratio: float,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> list[Candidate]:
    """保留分数不低于 `top_score * ratio` 的前缀。

    参数语义：
      ratio    0.35 表示"分数掉到第 1 名的 35% 以下就不要了"
      min_keep 至少保留几条（**不能返回空**：如果只因为相对分低就全丢，
               本来能答的问题会变成拒答，这是更严重的错误）
      max_keep 至多保留几条（一般等于 top_k）

    注意 `ratio <= 0` 时退化为"不做截断"（只受 max_keep 限制），
    这是配置项 `AUTOCUT_ENABLED=false` 的实现方式。
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: (-c.score, c.vector_id))
    keep_count = len(ordered)

    if ratio > 0:
        top = ordered[0].score
        if top > 0:
            threshold = top * ratio
            keep_count = 0
            for index, candidate in enumerate(ordered, start=1):
                if candidate.score >= threshold:
                    keep_count = index
                else:
                    break

    keep_count = max(min_keep, keep_count)
    if max_keep is not None:
        keep_count = min(keep_count, max_keep)
    return ordered[:keep_count]


def evaluate_gate(
    candidates: list[Candidate],
    *,
    vector_threshold: float,
    keyword_threshold: float,
) -> GateResult:
    """双阈值闸门判定。

    `best_vector_score` 只看**有向量分数**的候选（BM25 独有候选没有向量分，
    不能当成 0 参与 max，否则会把"关键词命中但向量没召回"的情况误杀）。
    `best_keyword_coverage` 同理是全体候选的最大值。
    """
    if not candidates:
        return GateResult(
            passed=False,
            reason=REASON_NO_CANDIDATES,
            vector_threshold=vector_threshold,
            keyword_threshold=keyword_threshold,
            best_vector_score=0.0,
            best_keyword_coverage=0.0,
        )

    vector_values = [c.vector_score for c in candidates if c.vector_score is not None]
    best_vector = max(vector_values) if vector_values else 0.0
    best_coverage = max((c.keyword_coverage for c in candidates), default=0.0)

    vector_ok = best_vector >= vector_threshold
    keyword_ok = best_coverage >= keyword_threshold

    if vector_ok:
        reason = REASON_VECTOR
    elif keyword_ok:
        reason = REASON_KEYWORD
    else:
        reason = REASON_BELOW_THRESHOLDS

    return GateResult(
        passed=vector_ok or keyword_ok,
        reason=reason,
        vector_threshold=vector_threshold,
        keyword_threshold=keyword_threshold,
        best_vector_score=best_vector,
        best_keyword_coverage=best_coverage,
    )


def trim_to_budget(
    candidates: list[Candidate], *, max_chars: int, min_keep: int = 1
) -> tuple[list[Candidate], int]:
    """按字符预算裁剪上下文，返回 `(保留的候选, 丢掉的条数)`。

    **为什么要按字符而不是按条数**：一条父块可能 1800 字，也可能 200 字，
    按条数截断会让 prompt 大小在 1k~9k 之间剧烈波动，成本和延迟都不可控。

    **为什么"从前往后装、装不下就整条跳过"而不是截断中间那条**：
    半截的 chunk 会让引用编号指向不完整内容，模型可能基于被切断的句子编答案。
    宁可少给一条。

    最后一条铁律：**至少保留 `min_keep` 条**。全都装不下也要留第一条，
    否则会出现"检索到了但上下文是空的"这种自相矛盾的状态。
    """
    if not candidates:
        return [], 0

    kept: list[Candidate] = []
    used = 0
    dropped = 0
    for candidate in candidates:
        cost = len(candidate.context_text)
        # 至少保留 min_keep 条：前 min_keep 条无条件留下（哪怕单条就已经超预算），
        # 之后装不下的才整条跳过。只写 `if kept and ...` 会让 min_keep>=2 形同虚设
        # （第二、三条照样被丢，返回条数恒为 1），而 min_keep=1 时两者等价。
        if len(kept) >= min_keep and used + cost > max_chars:
            dropped += 1
            continue
        kept.append(candidate)
        used += cost

    if not kept:
        kept = candidates[:min_keep]
        dropped = max(0, len(candidates) - len(kept))
    return kept, dropped


__all__ = [
    "REASON_BELOW_THRESHOLDS",
    "REASON_KEYWORD",
    "REASON_NO_CANDIDATES",
    "REASON_VECTOR",
    "autocut",
    "evaluate_gate",
    "trim_to_budget",
]
