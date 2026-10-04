"""融合（RRF / 加权归一化）、分数断崖截断（autocut）、相关性闸门、字符预算裁剪。

这四件事是"检索结果能不能用"的最后一道关，各自的承诺：

* **归一化**：所有分数相同时返回全 1.0（"大家一样好"不等于"大家一样差"）；
* **RRF**：`rank` 从 **1** 开始（第 1 名是 `1/(k+1)`，不是 `1/k`）；
* **autocut**：按分数断崖截断，但**永远不返回空**（本来能答的问题变成拒答更严重）；
* **gate**：向量达标 **或** 覆盖率达标就放行（用"与"会把两种正确情况都误杀）；
  且 `best_vector_score` 只统计**有向量分**的候选，不能被 BM25 独有候选带偏；
* **预算裁剪**：装不下就**整条跳过**，绝不截断半条（半截 chunk 会让引用指向不完整内容）。
"""

from __future__ import annotations

import pytest

from knowflow.retrieval.autocut import (
    REASON_BELOW_THRESHOLDS,
    REASON_KEYWORD,
    REASON_NO_CANDIDATES,
    REASON_VECTOR,
    autocut,
    evaluate_gate,
    trim_to_budget,
)
from knowflow.retrieval.fusion import (
    DEFAULT_RRF_K,
    minmax_normalize,
    rank_of,
    rrf_fuse,
    weighted_fuse,
)
from knowflow.retrieval.types import Candidate


def cand(
    vector_id: str,
    score: float = 0.0,
    *,
    vector_score: float | None = None,
    bm25_score: float | None = None,
    keyword_coverage: float = 0.0,
    text: str = "x" * 10,
) -> Candidate:
    return Candidate(
        vector_id=vector_id,
        doc_id=1,
        doc_name="a.md",
        content=text,
        context_text=text,
        vector_score=vector_score,
        bm25_score=bm25_score,
        keyword_coverage=keyword_coverage,
        score=score,
    )


# ======================================================================================
# 融合
# ======================================================================================
def test_所有分数相同时归一化返回全1而不是全0() -> None:
    """返回 0 会让后续阈值判定误判为"全都不相关"，而"大家一样好"显然不是"一样差"。"""
    assert minmax_normalize({"a": 0.5, "b": 0.5, "c": 0.5}) == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_只有一个样本时归一化返回1() -> None:
    assert minmax_normalize({"only": 3.0}) == {"only": 1.0}


def test_空输入归一化返回空字典() -> None:
    assert minmax_normalize({}) == {}


def test_归一化把最小最大压到0与1() -> None:
    assert minmax_normalize({"a": 0.0, "b": 1.0, "c": 0.5}) == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_RRF的第一名权重是k加1而不是k() -> None:
    """从 0 开始会让第 1 名从 1/61 变成 1/60：差别很小，但复现论文结果时会对不上。"""
    fused = rrf_fuse([["a", "b"]], k=DEFAULT_RRF_K)
    assert fused["a"] == pytest.approx(1.0 / (60 + 1))
    assert fused["a"] != pytest.approx(1.0 / 60)
    assert fused["b"] == pytest.approx(1.0 / 62)


def test_RRF把多路排名相加() -> None:
    """同一 id 出现在两路里应该累加，这正是"两路都认可"的信号。"""
    fused = rrf_fuse([["a", "b"], ["a"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_RRF空排名列表返回空字典() -> None:
    assert rrf_fuse([], k=60) == {}
    assert rrf_fuse([[]], k=60) == {}


def test_rank_of从1开始且按分数降序() -> None:
    assert rank_of({"a": 0.1, "b": 0.9, "c": 0.5}) == {"b": 1, "c": 2, "a": 3}


def test_加权融合按alpha分配权重() -> None:
    """alpha=0.5 时两路权重相同；两路各自在**自己的候选集合内**归一化。"""
    fused = weighted_fuse({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 10.0}, alpha=0.5)
    assert fused["a"] == pytest.approx(0.5)
    assert fused["b"] == pytest.approx(0.5)


def test_只出现在一路的候选不会被丢弃() -> None:
    """专有名词只命中 BM25、同义改写只命中向量，两类都不能丢。"""
    fused = weighted_fuse({"a": 0.9}, {"b": 5.0}, alpha=0.5)
    assert set(fused) == {"a", "b"}
    assert fused["a"] == pytest.approx(0.5)  # 向量归一化后唯一项=1，BM25 侧按 0 计
    assert fused["b"] == pytest.approx(0.5)


@pytest.mark.parametrize("alpha", [-1.0, 2.0])
def test_加权融合的alpha被夹到0与1之间(alpha: float) -> None:
    fused = weighted_fuse({"a": 1.0}, {"b": 1.0}, alpha=alpha)
    assert 0.0 <= fused["a"] <= 1.0
    assert 0.0 <= fused["b"] <= 1.0


# ======================================================================================
# autocut
# ======================================================================================
def test_autocut在分数断崖处截断() -> None:
    """第 3 条掉到第 1 条的 10%（< 35%）-> 从它开始整批丢掉。"""
    candidates = [cand("1:0", 1.0), cand("1:1", 0.5), cand("1:2", 0.1)]
    kept = autocut(candidates, ratio=0.35)
    assert [c.vector_id for c in kept] == ["1:0", "1:1"]


def test_autocut保持分数降序() -> None:
    candidates = [cand("1:2", 0.1), cand("1:0", 1.0), cand("1:1", 0.5)]
    assert [c.vector_id for c in autocut(candidates, ratio=0.0)] == ["1:0", "1:1", "1:2"]


@pytest.mark.parametrize("ratio", [10.0, 1.0])
def test_autocut结果永不为空(ratio: float) -> None:
    """全丢会让"本来能答"变成"拒答"，比多留一条噪声严重得多。"""
    kept = autocut([cand("1:0", 0.4)], ratio=ratio, min_keep=1)
    assert len(kept) == 1


def test_autocut的min_keep会补足条数() -> None:
    candidates = [cand("1:0", 1.0), cand("1:1", 0.2), cand("1:2", 0.1)]
    assert len(autocut(candidates, ratio=0.9, min_keep=2)) == 2


def test_autocut的ratio不大于0时只受max_keep限制() -> None:
    """`AUTOCUT_ENABLED=false` 就是用 ratio<=0 实现的：不做断崖截断，但要遵守 top_k。"""
    candidates = [cand(f"1:{i}", 1.0 - i * 0.01) for i in range(5)]
    assert len(autocut(candidates, ratio=0.0, max_keep=2)) == 2
    assert len(autocut(candidates, ratio=-1.0, max_keep=None)) == 5


def test_autocut空输入返回空列表() -> None:
    assert autocut([], ratio=0.5) == []


# ======================================================================================
# 闸门
# ======================================================================================
def test_向量分达标时闸门放行并给出向量原因() -> None:
    gate = evaluate_gate(
        [cand("1:0", vector_score=0.8)], vector_threshold=0.6, keyword_threshold=0.4
    )
    assert gate.passed is True
    assert gate.reason == REASON_VECTOR
    assert gate.best_vector_score == pytest.approx(0.8)


def test_只有覆盖率达标时闸门同样放行() -> None:
    """专有名词（PAYLOAD_TOO_LARGE）往往只能靠覆盖率救回来，用"与"会误杀。"""
    gate = evaluate_gate(
        [cand("1:0", vector_score=0.1, keyword_coverage=0.9)],
        vector_threshold=0.6,
        keyword_threshold=0.4,
    )
    assert gate.passed is True
    assert gate.reason == REASON_KEYWORD


def test_两个阈值都不达标时拒绝放行() -> None:
    gate = evaluate_gate(
        [cand("1:0", vector_score=0.2, keyword_coverage=0.1)],
        vector_threshold=0.6,
        keyword_threshold=0.4,
    )
    assert gate.passed is False
    assert gate.reason == REASON_BELOW_THRESHOLDS


def test_没有候选时原因是no_candidates() -> None:
    gate = evaluate_gate([], vector_threshold=0.6, keyword_threshold=0.4)
    assert gate.passed is False
    assert gate.reason == REASON_NO_CANDIDATES
    assert gate.best_vector_score == 0.0
    assert gate.best_keyword_coverage == 0.0


def test_负向量分不会被BM25独有候选的占位0掩盖() -> None:
    """BM25 独有候选没有向量分，绝不能当成 0 参与 max：
    否则"唯一的向量分是 -0.1"会被抬成 0.0，在阈值 0.0 时错误放行。"""
    gate = evaluate_gate(
        [
            cand("2:0", bm25_score=5.0, vector_score=None, keyword_coverage=0.0),
            cand("1:0", vector_score=-0.1, keyword_coverage=0.0),
        ],
        vector_threshold=0.0,
        keyword_threshold=0.5,
    )
    assert gate.best_vector_score == pytest.approx(-0.1)
    assert gate.passed is False


def test_闸门取全体候选里最好的覆盖率() -> None:
    gate = evaluate_gate(
        [cand("1:0", keyword_coverage=0.2), cand("1:1", keyword_coverage=0.7)],
        vector_threshold=0.9,
        keyword_threshold=0.4,
    )
    assert gate.best_keyword_coverage == pytest.approx(0.7)
    assert gate.passed is True


def test_闸门结果可序列化成契约结构() -> None:
    gate = evaluate_gate(
        [cand("1:0", vector_score=0.8, keyword_coverage=0.5)],
        vector_threshold=0.6,
        keyword_threshold=0.4,
    )
    payload = gate.to_dict()
    assert payload["passed"] is True
    assert payload["reason"] == REASON_VECTOR
    assert payload["best_vector_score"] == 0.8


# ======================================================================================
# 预算裁剪
# ======================================================================================
def test_超预算的候选项被整条跳过而不是截断() -> None:
    """半截 chunk 会让引用编号指向不完整内容，模型可能基于被切断的句子编答案。"""
    candidates = [cand(f"1:{i}", text="y" * 100) for i in range(10)]
    kept, dropped = trim_to_budget(candidates, max_chars=250)
    assert [len(c.context_text) for c in kept] == [100, 100]
    assert dropped == 8
    assert kept[1].context_text == "y" * 100  # 没有被切成 50 字


def test_预算充足时全部保留() -> None:
    candidates = [cand(f"1:{i}", text="y" * 10) for i in range(3)]
    kept, dropped = trim_to_budget(candidates, max_chars=1000)
    assert len(kept) == 3
    assert dropped == 0


def test_单条就超预算时仍然保留至少min_keep条() -> None:
    """ "检索到了但上下文是空的"是自相矛盾的状态，必须避免。"""
    candidates = [cand(f"1:{i}", text="y" * 100) for i in range(5)]
    kept, dropped = trim_to_budget(candidates, max_chars=1)
    assert len(kept) == 1
    assert dropped == 4


def test_min_keep大于1时也生效() -> None:
    """`if kept and ...` 这种写法会让 min_keep>=2 形同虚设（返回条数恒为 1）。"""
    candidates = [cand(f"1:{i}", text="y" * 100) for i in range(5)]
    kept, dropped = trim_to_budget(candidates, max_chars=1, min_keep=3)
    assert len(kept) == 3
    assert dropped == 2


def test_空候选列表返回空与0() -> None:
    assert trim_to_budget([], max_chars=100) == ([], 0)
