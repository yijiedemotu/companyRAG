"""多路召回融合。

**核心问题：两路分数的量纲完全不同，不能直接相加。**

- 向量分数是 cosine 相似度，范围 [0, 1]，分布集中在 0.4~0.9；
- BM25 分数无上界，取决于 IDF 与文档长度，实测可能 0.5 也可能是 37。

直接 `0.6*vector + 0.4*bm25` 的结果是 BM25 完全主导（数值大 10 倍以上），
表现为"关键词搜出来的东西永远排第一，向量检索形同虚设"。

两种主流解法，本项目都实现，**默认 RRF**：

**方案 A：加权 min-max 归一化**
    把两路各自压到 [0,1] 再加权。
    优点：可解释、参数少（只有一个 alpha）、保留"分数差距"的信息。
    缺点：**对离群值敏感**——如果一路里有个 0.99 的极端值，其余会被压到 0 附近，
    归一化后的相对差异失真；而且每次查询的 min/max 不同，
    同样的分数在不同查询里含义不一样。

**方案 B：RRF（Reciprocal Rank Fusion）**
    `score = Σ 1/(k + rank_i)`，只用排名不用分数。
    优点：**对分数尺度漂移免疫**——换 embedding 模型、改 BM25 参数都不需要重新调参；
    天然抑制单路离群值。k=60 是原论文推荐值（作用是把前几名的差距拉平一些）。
    缺点：丢弃了"分数差距"信息（第 1 名和第 2 名相差 0.5 还是 0.001，RRF 不看）。

工业界默认 RRF，原因就是"稳"：**在没人持续调参的系统里，鲁棒性比理论最优更重要。**
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

DEFAULT_RRF_K: Final[int] = 60


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """把一组分数线性压到 [0,1]。

    边界处理很重要：**所有分数相同时返回全 1.0 而不是全 0.0**。
    返回 0 会让后续"阈值判定"误判为"全都不相关"，而"大家一样好"
    显然不等于"大家一样差"。只有一个样本时同理返回 1.0。
    """
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    span = high - low
    if span <= 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / span for key, value in scores.items()}


def weighted_fuse(
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    *,
    alpha: float = 0.6,
) -> dict[str, float]:
    """方案 A：归一化后加权。`alpha` 是向量的权重（0=纯 BM25，1=纯向量）。

    两路各自 min-max 归一化**在各自的候选集合内**进行。
    只出现在一路里的候选，另一路按 0 计（而不是丢弃）——
    这样"只在 BM25 里命中"的专有名词仍然能进入结果，只是需要靠 alpha 平衡。
    """
    alpha = min(1.0, max(0.0, alpha))
    vec_norm = minmax_normalize(vector_scores)
    bm25_norm = minmax_normalize(bm25_scores)

    fused: dict[str, float] = {}
    for key in set(vec_norm) | set(bm25_norm):
        fused[key] = alpha * vec_norm.get(key, 0.0) + (1.0 - alpha) * bm25_norm.get(key, 0.0)
    return fused


def rrf_fuse(rankings: Sequence[Sequence[str]], *, k: int = DEFAULT_RRF_K) -> dict[str, float]:
    """方案 B：RRF。每个 ranking 是按相关性降序排列的 id 列表。

    `1/(k + rank)`，rank 从 **1** 开始（不是 0）。
    从 0 开始会让第 1 名的权重从 1/61 变成 1/60，虽然差别很小，
    但复现论文/他人结果时会发现对不上，属于低级但很烦的错误。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def rank_of(scores: dict[str, float]) -> dict[str, int]:
    """`id -> 排名（1 起）`，按分数降序。用于记录 `vector_rank` / `bm25_rank`。"""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {key: rank for rank, (key, _) in enumerate(ordered, start=1)}


__all__ = [
    "DEFAULT_RRF_K",
    "minmax_normalize",
    "rank_of",
    "rrf_fuse",
    "weighted_fuse",
]
