"""重排（rerank）。

**为什么要重排**：召回阶段的目标是"别漏"（高召回），所以会放宽到 `fetch_k=20` 条；
但"相关的"和"相似的"不是一回事 —— 向量检索很容易把"讲同一主题但不回答问题"
的片段排到前面。重排用**更强的判据**（能同时看到 query 和候选）重新打分，
把真正有用的提到前面。代价是它很贵，所以**只对前 `RERANK_TOP_N` 条做**。

两条实现，各有存在理由：

1. `llm_rerank` —— 让模型逐条打 0-10 分。判据最强，但要花一次模型调用（约 0.5~2 秒）。
2. `heuristic_rerank` —— 用「查询词覆盖率 + 小节/文件名命中 + 向量分」加权。
   零成本、毫秒级。**它的价值不是"替代 LLM"，而是兜底**：
   离线模式、模型超时、返回格式坏了的时候，链路不能断。

**失败必须降级而不是抛错**：重排失败时用启发式结果继续，并记 warning。
理由和"闸门"一致——重排是**优化**，不是**正确性前提**。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.llm.base import ChatMessage, ChatModel
from knowflow.llm.parsing import extract_json_object
from knowflow.llm.prompts import build_rerank_messages
from knowflow.retrieval.fusion import minmax_normalize
from knowflow.retrieval.text import coverage
from knowflow.retrieval.types import Candidate

logger = get_logger(__name__)

# 启发式重排的权重。为什么要 "文本覆盖 0.60" 最大：
# 重排要回答的是"这条能不能回答这个问题"，词面覆盖是最直接的代理信号；
# 小节路径/文件名的权重给 0.25，因为命中标题往往是强信号（问"住宿标准"命中
# "差旅报销标准 > 住宿标准"基本就是答案所在）；
# 向量分只给 0.15 —— 它已经用过一次了（召回），这里再给高权重等于重复计票。
W_CONTENT: Final[float] = 0.60
W_SECTION: Final[float] = 0.25
W_VECTOR: Final[float] = 0.15

# 融合分在最终分里的保留权重：完全相信 LLM 重排会放大它的随机性，
# 留 30% 给融合分做"稳定器"。
BLEND_RERANK: Final[float] = 0.70
BLEND_FUSED: Final[float] = 0.30


def heuristic_score(query: str, candidate: Candidate) -> float:
    """[0,1] 的启发式相关性分。"""
    content_cov = coverage(query, candidate.content)
    section_text = " ".join(filter(None, [candidate.section_path, candidate.doc_name]))
    section_cov = coverage(query, section_text) if section_text else 0.0
    vector_part = candidate.vector_score if candidate.vector_score is not None else 0.0
    vector_part = max(0.0, min(1.0, vector_part))
    return W_CONTENT * content_cov + W_SECTION * section_cov + W_VECTOR * vector_part


def heuristic_rerank(query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
    """零成本重排。就地写入 `rerank_score` 并返回按新分排序的副本。"""
    rescored: list[Candidate] = []
    for candidate in candidates:
        score = heuristic_score(query, candidate)
        candidate.rerank_score = round(score, 6)
        rescored.append(candidate)
    rescored.sort(key=lambda c: (-(c.rerank_score or 0.0), c.vector_id))
    return rescored


def _parse_scores(text: str, valid_ids: set[int]) -> dict[int, float]:
    """解析重排打分。任何不合法的项都跳过（而不是整体失败）。"""
    obj = extract_json_object(text)
    if not obj:
        return {}
    raw_items = obj.get("scores") or obj.get("items") or []
    if not isinstance(raw_items, list):
        return {}

    scores: dict[int, float] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        raw_score = item.get("score")
        if raw_id is None or raw_score is None:
            continue
        try:
            item_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if item_id not in valid_ids:
            continue
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            continue
        # 夹到 [0,1]：模型偶尔会给 15 或 -3，不夹会让排序完全错乱
        scores[item_id] = max(0.0, min(1.0, value / 10.0))
    return scores


def llm_rerank(
    query: str,
    candidates: Sequence[Candidate],
    *,
    model: ChatModel,
    top_n: int,
) -> tuple[list[Candidate], bool]:
    """用模型打分为前 `top_n` 条重排。

    返回 `(排好序的候选, 是否真的用上了 LLM 分数)`。
    **只对前 top_n 条调用模型**，其余候选保持原相对顺序排在后面 ——
    它们本来也进不了最终 top_k，没必要花钱。
    """
    head = list(candidates[:top_n])
    tail = list(candidates[top_n:])
    if not head:
        return [], False

    contexts = [(index, c.content) for index, c in enumerate(head, start=1)]
    messages: list[ChatMessage] = build_rerank_messages(query=query, contexts=contexts)

    try:
        result = model.complete(messages, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - 重排是优化，失败不能中断链路
        logger.warning("rerank.llm_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        return list(candidates), False

    valid_ids = {index for index, _ in contexts}
    scores = _parse_scores(result.text, valid_ids)
    if not scores:
        logger.warning("rerank.llm_unparsable", preview=result.text[:120])
        return list(candidates), False

    for index, candidate in enumerate(head, start=1):
        llm_score = scores.get(index)
        fallback = heuristic_score(query, candidate)
        # 模型没给某条打分（漏项）时用启发式补，避免那条被当成 0 分沉底
        candidate.rerank_score = round(llm_score if llm_score is not None else fallback, 6)

    head.sort(key=lambda c: (-(c.rerank_score or 0.0), c.vector_id))
    return head + tail, True


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    *,
    model: ChatModel | None = None,
    top_n: int | None = None,
    enabled: bool = True,
    settings: Settings | None = None,
) -> tuple[list[Candidate], dict[str, object]]:
    """重排入口。返回 `(候选, 元信息)`，元信息用于 trace 与 `/search` 的 debug。

    `enabled=False` 或没有模型时走启发式 —— 但**仍然执行**，
    因为启发式会重写 `rerank_score`，`/search` 界面靠它显示"重排后分数"。
    """
    cfg = settings or get_settings()
    limit = top_n if top_n is not None else cfg.rerank_top_n
    items = list(candidates)
    if not items:
        return [], {"reranked": False, "strategy": "none", "count": 0}

    if enabled and model is not None and not getattr(model, "offline", False):
        ordered, used_llm = llm_rerank(query, items, model=model, top_n=limit)
        if used_llm:
            _blend_with_fused(ordered)
            return ordered, {"reranked": True, "strategy": "llm", "count": len(ordered)}
        strategy = "heuristic(llm_failed)"
    else:
        strategy = (
            "heuristic(offline)"
            if model is None or getattr(model, "offline", False)
            else "heuristic(disabled)"
        )

    ordered = heuristic_rerank(query, items)
    _blend_with_fused(ordered)
    return ordered, {"reranked": True, "strategy": strategy, "count": len(ordered)}


def _blend_with_fused(candidates: list[Candidate]) -> None:
    """把标准化后的融合分与重排分混合成最终 `score`，并就地按 `score` 重排。

    标准化在**本批候选内**做 min-max —— 这样即使融合分与重排分尺度不同，
    也能稳定地按 7:3 混合。
    """
    fused_norm = minmax_normalize({c.vector_id: c.fused_score for c in candidates})
    for candidate in candidates:
        rerank_part = candidate.rerank_score or 0.0
        fused_part = fused_norm.get(candidate.vector_id, 0.0)
        candidate.score = round(BLEND_RERANK * rerank_part + BLEND_FUSED * fused_part, 6)
    candidates.sort(key=lambda c: (-c.score, c.vector_id))


__all__ = [
    "BLEND_FUSED",
    "BLEND_RERANK",
    "heuristic_rerank",
    "heuristic_score",
    "llm_rerank",
    "rerank",
]
