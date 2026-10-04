"""自研 BM25：公式、IDF 的 ``+1``、幂等、按 KB 隔离。

为什么值得钉死公式：BM25 的分数没有"看起来对不对"这回事 —— 参数或归一化写错，
排序会整体变样但**不会报错**。所以这里用手算值逐条对照，而不是断言"分数大于 0"。

重点中的重点是 **IDF 的 `+1`**：经典形式的 IDF 在 `df > N/2` 时变负，
会让"包含该高频词的文档反而被扣分"（表现是"越常见的关键词越查不到"）。
"""

from __future__ import annotations

import math

import pytest

from knowflow.retrieval.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index, BM25Registry


def make_payload(content: str = "") -> dict[str, object]:
    return {"content": content, "doc_id": 1, "doc_name": "a.md", "chunk_index": 0}


# --------------------------------------------------------------------------------------
# 公式：手算对照
# --------------------------------------------------------------------------------------
def test_两篇文档的BM25分数与手算值一致() -> None:
    """d1=`aa bb`、d2=`aa cc`，query=`aa bb`。

    N=2，avgdl=2，k1=1.5，b=0.75；两篇长度都等于 avgdl，所以长度归一化因子恰好为 1，
    分数就等于 IDF 之和：
        IDF(aa)=ln(1+(2-2+0.5)/(2+0.5))=ln(1.2)=0.1823215567939546
        IDF(bb)=ln(1+(2-1+0.5)/(1+0.5))=ln(2)  =0.6931471805599453
    """
    index = BM25Index()
    index.add("d1", "aa bb", make_payload("aa bb"))
    index.add("d2", "aa cc", make_payload("aa cc"))

    assert index.avgdl == pytest.approx(2.0)
    scores = {match.doc_key: match.score for match in index.search("aa bb", top_k=5)}
    assert scores["d1"] == pytest.approx(0.8754687373538999, abs=1e-9)
    assert scores["d2"] == pytest.approx(0.1823215567939546, abs=1e-9)
    assert scores["d1"] == pytest.approx(math.log(1.2) + math.log(2.0), abs=1e-12)


def test_长度归一化按公式生效() -> None:
    """d1=`aa`（长 1）、d2=`aa bb cc dd`（长 4），query=`aa`；N=2，avgdl=2.5。

    分母 = tf + k1*(1-b+b*|D|/avgdl)，所以短文档得分更高：
        d1 分母 = 1 + 1.5*(0.25 + 0.75*1/2.5) = 1.825
        d2 分母 = 1 + 1.5*(0.25 + 0.75*4/2.5) = 2.8
    期望值按公式独立算一遍（不调用被测实现），再与实现比对。
    """
    index = BM25Index()
    index.add("d1", "aa", make_payload("aa"))
    index.add("d2", "aa bb cc dd", make_payload("aa bb cc dd"))

    idf_aa = math.log(1.0 + (2 - 2 + 0.5) / (2 + 0.5))
    expected_d1 = (
        idf_aa * (1 * (DEFAULT_K1 + 1.0)) / (1 + DEFAULT_K1 * (1 - DEFAULT_B + DEFAULT_B * 1 / 2.5))
    )
    expected_d2 = (
        idf_aa * (1 * (DEFAULT_K1 + 1.0)) / (1 + DEFAULT_K1 * (1 - DEFAULT_B + DEFAULT_B * 4 / 2.5))
    )

    scores = {match.doc_key: match.score for match in index.search("aa", top_k=5)}
    assert scores["d1"] == pytest.approx(expected_d1, abs=1e-12)
    assert scores["d2"] == pytest.approx(expected_d2, abs=1e-12)
    assert scores["d1"] > scores["d2"], "长文档天然更容易命中，必须被惩罚"


def test_高频词的IDF仍然为正所以包含它的文档不会得负分() -> None:
    """3 篇文档全都含 `aa`（df=3 > N/2）—— 这正是经典 IDF 变负的区间。

    经典形式 ln((N-df+0.5)/(df+0.5)) = ln(0.5/3.5) < 0；
    加了 +1 之后 ln(1 + 0.5/3.5) > 0。
    """
    index = BM25Index()
    index.add("d1", "aa aa", make_payload("aa aa"))
    index.add("d2", "aa bb", make_payload("aa bb"))
    index.add("d3", "aa cc", make_payload("aa cc"))

    classic = math.log((3 - 3 + 0.5) / (3 + 0.5))
    assert classic < 0, "前提：不加 +1 的经典 IDF 在这个区间确实是负数"
    assert index._idf("aa") == pytest.approx(0.13353139262452257, abs=1e-12)

    scores = {match.doc_key: match.score for match in index.search("aa", top_k=5)}
    assert all(score > 0 for score in scores.values()), scores
    # 词频高的排前面（词频饱和但不倒扣）
    assert scores["d1"] > scores["d2"]
    assert scores["d1"] == pytest.approx(0.1907591323207465, abs=1e-9)


def test_长度为1的文档不会因为除零炸掉() -> None:
    """空文档（tokens 为空）长度记为 1，不能出现 ZeroDivision 或 NaN。"""
    index = BM25Index()
    index.add("empty", "", make_payload(""))
    index.add("d1", "aa", make_payload("aa"))
    assert index.doc_count == 2
    scores = {match.doc_key: match.score for match in index.search("aa", top_k=5)}
    assert math.isfinite(scores["d1"])


# --------------------------------------------------------------------------------------
# 幂等与增删
# --------------------------------------------------------------------------------------
def test_同一doc_key重复add不会重复计数() -> None:
    """覆盖式写入：入库重跑（或重建索引）不能把同一块算两遍，否则 df/avgdl 全部漂移。"""
    index = BM25Index()
    index.add("k", "住宿标准", make_payload("住宿标准"))
    index.add("k", "住宿标准", make_payload("住宿标准"))
    assert index.doc_count == 1
    assert index._df["住"] == 1
    assert index.avgdl == pytest.approx(7.0)  # 4 个单字 + 3 个 bigram


def test_覆盖写入会替换正文与分数() -> None:
    index = BM25Index()
    index.add("k", "餐饮补贴", make_payload("餐饮补贴"))
    index.add("k", "住宿标准", make_payload("住宿标准"))
    assert index.search("住宿标准", top_k=5)
    assert index.search("餐饮补贴", top_k=5) == []


def test_remove会同时更新df与平均长度() -> None:
    index = BM25Index()
    index.add("d1", "aa bb", make_payload("aa bb"))
    index.add("d2", "aa", make_payload("aa"))
    assert index.remove("d1") is True
    assert index.remove("d1") is False  # 再删一次返回 False，不做无谓操作
    assert index.doc_count == 1
    assert index._df.get("bb") is None
    assert index.avgdl == pytest.approx(1.0)


def test_remove_where按payload批量删除() -> None:
    index = BM25Index()
    index.add("1:0", "aa", {"content": "aa", "doc_id": 1})
    index.add("1:1", "aa", {"content": "aa", "doc_id": 1})
    index.add("2:0", "aa", {"content": "aa", "doc_id": 2})
    assert index.remove_where(lambda payload: payload.get("doc_id") == 1) == 2
    assert index.doc_count == 1
    assert [match.doc_key for match in index.search("aa", top_k=5)] == ["2:0"]


def test_clear清空全部统计() -> None:
    index = BM25Index()
    index.add("d1", "aa bb", make_payload("aa bb"))
    index.clear()
    assert index.doc_count == 0
    assert index.term_count == 0
    assert index.avgdl == 0.0
    assert index.search("aa", top_k=5) == []


# --------------------------------------------------------------------------------------
# 检索边界
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_切不出词元的查询返回空结果(query: str) -> None:
    index = BM25Index()
    index.add("d1", "住宿标准", make_payload("住宿标准"))
    assert index.search(query, top_k=5) == []


def test_空索引检索返回空结果() -> None:
    assert BM25Index().search("住宿标准", top_k=5) == []


def test_查询词不在任何文档里时返回空结果() -> None:
    index = BM25Index()
    index.add("d1", "住宿标准", make_payload("住宿标准"))
    assert index.search("量子计算机", top_k=5) == []


def test_top_k会截断结果() -> None:
    index = BM25Index()
    for i in range(5):
        index.add(f"d{i}", "aa bb", make_payload("aa bb"))
    assert len(index.search("aa", top_k=2)) == 2


def test_stats反映索引规模() -> None:
    index = BM25Index()
    index.add("d1", "aa bb", make_payload("aa bb"))
    stats = index.stats()
    assert stats["docs"] == 1
    assert stats["terms"] == 2
    assert stats["k1"] == DEFAULT_K1
    assert stats["b"] == DEFAULT_B


def test_coverage_of用文档正文算覆盖率() -> None:
    index = BM25Index()
    index.add("d1", "住宿标准", make_payload("一线城市住宿标准为每晚 600 元"))
    assert index.coverage_of("住宿标准", "d1") == 1.0
    assert index.coverage_of("住宿标准", "不存在") == 0.0


# --------------------------------------------------------------------------------------
# Registry：按 KB 隔离
# --------------------------------------------------------------------------------------
def make_chunk(vector_id: str, content: str, doc_id: int, name: str) -> dict[str, object]:
    return {
        "vector_id": vector_id,
        "content": content,
        "doc_id": doc_id,
        "doc_name": name,
        "chunk_index": 0,
        "page_no": None,
        "section_path": None,
    }


@pytest.fixture
def registry() -> BM25Registry:
    registry = BM25Registry()
    registry.add_chunks(1, [make_chunk("1:0", "一线城市住宿标准六百元", 1, "报销制度.md")])
    registry.add_chunks(2, [make_chunk("2:0", "餐饮补贴每天一百元", 2, "补贴制度.md")])
    return registry


def test_不同KB的文档不会串到彼此的结果里(registry: BM25Registry) -> None:
    """共用一个索引会让一个 KB 的 avgdl/df 影响另一个 KB 的打分（跨库污染）。"""
    kb1 = registry.search(kb_id=1, query="住宿标准", top_k=5)
    kb2 = registry.search(kb_id=2, query="住宿标准", top_k=5)
    assert [row["vector_id"] for row in kb1] == ["1:0"]
    assert kb2 == []


def test_kb_id为None时跨库合并检索(registry: BM25Registry) -> None:
    merged = registry.search(kb_id=None, query="标准 补贴", top_k=5)
    assert {row["vector_id"] for row in merged} == {"1:0", "2:0"}


def test_不存在的KB检索返回空而不是抛异常(registry: BM25Registry) -> None:
    assert registry.search(kb_id=999, query="住宿标准", top_k=5) == []


def test_remove_document按doc_id清除并rebuild可恢复(registry: BM25Registry) -> None:
    assert registry.remove_document(1, doc_id=1) == 1
    assert registry.doc_count_of(1) == 0
    assert registry.search(kb_id=1, query="住宿标准", top_k=5) == []


def test_drop_kb清掉整个索引(registry: BM25Registry) -> None:
    assert registry.drop_kb(1) is True
    assert registry.drop_kb(1) is False
    assert registry.doc_count_of(1) == 0
    assert registry.doc_count_of(2) == 1


def test_doc_count与stats汇总正确(registry: BM25Registry) -> None:
    assert registry.doc_count == 2
    assert registry.stats()["kb_count"] == 2
    assert registry.stats()["per_kb"] == {"1": 1, "2": 1}


def test_add_chunks返回写入条数() -> None:
    registry = BM25Registry()
    chunks = [make_chunk(f"1:{i}", "aa", 1, "a.md") for i in range(3)]
    assert registry.add_chunks(1, chunks) == 3
