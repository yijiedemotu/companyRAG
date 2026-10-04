"""重排（rerank）：启发式权重、LLM 打分的解析与降级、融合分混合。

核心承诺是**降级不中断链路**：重排只是"优化"，不是"正确性前提"。
模型超时、返回坏 JSON、离线模式，都要退回启发式继续跑，而不是把错误抛给用户。

另外两条容易写错的：
* `rerank_score` 必须**始终**被写回（`/search` 界面靠它显示"重排后分数"），
  即使走的是启发式；
* 模型漏打某条的分时，那条要用启发式补分，**不能当成 0 沉底**。
"""

from __future__ import annotations

import pytest

from knowflow.llm.base import ChatResult, ChatUsage
from knowflow.retrieval.rerank import (
    BLEND_FUSED,
    BLEND_RERANK,
    W_CONTENT,
    W_SECTION,
    W_VECTOR,
    heuristic_rerank,
    heuristic_score,
    llm_rerank,
    rerank,
)
from knowflow.retrieval.types import Candidate


class FakeOnlineModel:
    """ "在线"模型替身：`offline=False`，返回预置文本（或按需抛异常）。"""

    offline = False
    model = "fake-online"

    def __init__(self, text: str = "{}", *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[str] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, stop=None):  # noqa: ANN001, ANN202
        self.calls.append("\n".join(m.content for m in messages))
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.text, usage=ChatUsage(model=self.model, llm_calls=1))

    def stream(self, messages, **kwargs):  # noqa: ANN001, ANN202
        yield from ()

    def supports_tools(self) -> bool:
        return False


def cand(
    vector_id: str,
    *,
    content: str = "无关内容",
    doc_name: str = "a.md",
    section_path: str | None = None,
    vector_score: float | None = None,
    fused_score: float = 0.0,
) -> Candidate:
    return Candidate(
        vector_id=vector_id,
        doc_id=1,
        doc_name=doc_name,
        content=content,
        context_text=content,
        section_path=section_path,
        vector_score=vector_score,
        fused_score=fused_score,
    )


# ======================================================================================
# 启发式打分
# ======================================================================================
def test_启发式权重之和为1() -> None:
    assert pytest.approx(1.0) == W_CONTENT + W_SECTION + W_VECTOR


@pytest.mark.parametrize(
    ("content", "doc_name", "section_path", "vector_score", "expected"),
    [
        # 只有正文命中：0.60
        ("一线城市住宿标准为每晚 600 元", "a.md", None, None, W_CONTENT),
        # 只有小节路径/文件名命中：0.25
        ("与问题无关的正文", "住宿标准.md", None, None, W_SECTION),
        # 只有向量分：0.15
        ("与问题无关的正文", "a.md", None, 1.0, W_VECTOR),
        # 三者全中：1.0
        ("一线城市住宿标准为每晚 600 元", "住宿标准.md", "住宿标准", 1.0, 1.0),
        # 向量分超出 [0,1] 时被夹住（模型偶尔会给 15 或 -3）
        ("与问题无关的正文", "a.md", None, 5.0, W_VECTOR),
        ("与问题无关的正文", "a.md", None, -3.0, 0.0),
    ],
)
def test_启发式打分是覆盖率与向量分的加权和(
    content: str,
    doc_name: str,
    section_path: str | None,
    vector_score: float | None,
    expected: float,
) -> None:
    candidate = cand(
        "1:0",
        content=content,
        doc_name=doc_name,
        section_path=section_path,
        vector_score=vector_score,
    )
    assert heuristic_score("住宿标准", candidate) == pytest.approx(expected, abs=1e-9)


def test_启发式重排写回rerank_score并按新分排序() -> None:
    weak = cand("1:0", content="完全无关", vector_score=0.1)
    strong = cand("1:1", content="一线城市住宿标准为每晚 600 元", vector_score=0.9)
    ordered = heuristic_rerank("住宿标准", [weak, strong])
    assert [c.vector_id for c in ordered] == ["1:1", "1:0"]
    assert ordered[0].rerank_score is not None
    assert ordered[0].rerank_score > ordered[1].rerank_score


# ======================================================================================
# LLM 重排
# ======================================================================================
def test_LLM重排按模型给的分重新排序() -> None:
    model = FakeOnlineModel('{"scores": [{"id": 1, "score": 1}, {"id": 2, "score": 9}]}')
    first = cand("1:0", content="第一条")
    second = cand("1:1", content="第二条")
    ordered, used_llm = llm_rerank("问题", [first, second], model=model, top_n=5)
    assert used_llm is True
    assert [c.vector_id for c in ordered] == ["1:1", "1:0"]
    assert ordered[0].rerank_score == pytest.approx(0.9)  # score/10 归一化


def test_LLM重排只把前top_n条发给模型() -> None:
    """尾部候选本来也进不了最终 top_k，没必要为它们花钱。"""
    model = FakeOnlineModel('{"scores": [{"id": 1, "score": 5}]}')
    candidates = [cand(f"1:{i}", content=f"候选{i}") for i in range(4)]
    ordered, used_llm = llm_rerank("问题", candidates, model=model, top_n=1)
    assert used_llm is True
    prompt = model.calls[0]
    assert "候选0" in prompt
    assert "候选1" not in prompt
    assert len(ordered) == 4  # 尾部候选仍在，只是没被重排


def test_LLM重排漏打分的候选会用启发式补分而不是当0() -> None:
    """模型漏项时把那条当 0 会让它莫名沉底 —— 这是"重排反而变差"的常见原因。"""
    model = FakeOnlineModel('{"scores": [{"id": 1, "score": 0}]}')
    hit = cand("1:0", content="无关内容")
    missed = cand("1:1", content="一线城市住宿标准为每晚 600 元")
    ordered, used_llm = llm_rerank("住宿标准", [hit, missed], model=model, top_n=5)
    assert used_llm is True
    assert ordered[0].vector_id == "1:1"
    assert ordered[0].rerank_score == pytest.approx(W_CONTENT)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("这不是 JSON", None),
        ("[]", None),  # 数组而不是对象 -> 调用方走降级分支
        ('{"scores": []}', None),
        ("{}", RuntimeError("模型超时")),
    ],
)
def test_LLM重排拿不到可用分数时保持原顺序并报告未使用(text: str, error: Exception | None) -> None:
    model = FakeOnlineModel(text, error=error)
    candidates = [cand("1:0", content="a"), cand("1:1", content="b")]
    ordered, used_llm = llm_rerank("问题", candidates, model=model, top_n=5)
    assert used_llm is False
    assert [c.vector_id for c in ordered] == ["1:0", "1:1"]


def test_LLM重排丢弃不存在的编号与非法分值() -> None:
    model = FakeOnlineModel(
        '{"scores": [{"id": 99, "score": 9}, {"id": 1, "score": "坏值"}, {"id": 2, "score": 15}]}'
    )
    candidates = [cand("1:0", content="无关内容"), cand("1:1", content="也无关")]
    ordered, used_llm = llm_rerank("问题", candidates, model=model, top_n=5)
    assert used_llm is True
    # id=2 的 15 分被夹到 1.0；id=1 的非法值走启发式补分
    assert ordered[0].vector_id == "1:1"
    assert ordered[0].rerank_score == pytest.approx(1.0)


def test_LLM重排遇到空候选列表直接返回() -> None:
    model = FakeOnlineModel('{"scores": []}')
    assert llm_rerank("问题", [], model=model, top_n=5) == ([], False)


# ======================================================================================
# rerank 入口
# ======================================================================================
def test_入口在没有候选时返回none策略() -> None:
    assert rerank("问题", [], model=None, top_n=5) == (
        [],
        {"reranked": False, "strategy": "none", "count": 0},
    )


@pytest.mark.parametrize("model_kind", ["none", "offline"])
def test_离线或没有模型时走启发式但仍标记已重排(model_kind: str, settings) -> None:  # noqa: ANN001
    from knowflow.llm.mock import MockChatModel

    model = None if model_kind == "none" else MockChatModel(settings)
    ordered, meta = rerank(
        "住宿标准", [cand("1:0", content="一线城市住宿标准")], model=model, top_n=5
    )
    assert meta["reranked"] is True
    assert str(meta["strategy"]).startswith("heuristic")
    assert ordered[0].rerank_score is not None, "启发式也必须写回 rerank_score"


def test_在线模型返回坏JSON时降级为启发式并标注原因() -> None:
    ordered, meta = rerank(
        "住宿标准",
        [cand("1:0", content="一线城市住宿标准")],
        model=FakeOnlineModel("坏输出"),
        top_n=5,
    )
    assert meta["strategy"] == "heuristic(llm_failed)"
    assert ordered[0].rerank_score is not None


def test_显式关闭重排时策略标注为disabled() -> None:
    _ordered, meta = rerank(
        "住宿标准",
        [cand("1:0", content="一线城市住宿标准")],
        model=FakeOnlineModel('{"scores": [{"id": 1, "score": 9}]}'),
        enabled=False,
    )
    assert meta["strategy"] == "heuristic(disabled)"
    assert meta["reranked"] is True


def test_在线模型给出合法分数时策略为llm() -> None:
    ordered, meta = rerank(
        "住宿标准",
        [cand("1:0", content="无关"), cand("1:1", content="一线城市住宿标准")],
        model=FakeOnlineModel('{"scores": [{"id": 1, "score": 0}, {"id": 2, "score": 10}]}'),
        top_n=5,
    )
    assert meta["strategy"] == "llm"
    assert [c.vector_id for c in ordered] == ["1:1", "1:0"]


def test_最终分是重排分与融合分的加权混合() -> None:
    """完全相信 LLM 会放大它的随机性，所以留 30% 给融合分做稳定器。"""
    model = FakeOnlineModel('{"scores": [{"id": 1, "score": 10}, {"id": 2, "score": 5}]}')
    best = cand("1:0", content="一线城市住宿标准", fused_score=0.9)
    second = cand("1:1", content="也相关一些", fused_score=0.1)
    ordered, meta = rerank("住宿标准", [best, second], model=model, top_n=5)
    assert meta["strategy"] == "llm"
    first, last = ordered[0], ordered[1]
    assert first.score == pytest.approx(
        BLEND_RERANK * (first.rerank_score or 0.0) + BLEND_FUSED * 1.0, abs=1e-6
    )
    assert last.score == pytest.approx(
        BLEND_RERANK * (last.rerank_score or 0.0) + BLEND_FUSED * 0.0, abs=1e-6
    )
