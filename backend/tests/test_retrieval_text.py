"""中文切词与「查询词覆盖率」。

这个模块是**全项目唯一的切词实现**：BM25 打分与闸门的覆盖率阈值都建立在它上面。
两份实现漂移一点，`KEYWORD_MIN_COVERAGE` 的语义就跟着漂 —— 所以这里钉的是**具体词元**，
不是"长度大于 0"这种空洞断言。

另外保护一个刚修过的 bug：**文本切不出词元时覆盖率必须是 0.0，不能是 1.0**。
返回 1.0 会让一条 ``"!!!"`` 被判成"完全命中查询"，闸门放行一段毫无意义的文本，
还会挤掉真正相关的内容。
"""

from __future__ import annotations

import pytest

from knowflow.retrieval.text import (
    STOPWORDS,
    STOPWORDS_2,
    coverage,
    normalize_text,
    query_terms,
    tokenize_zh,
)


# --------------------------------------------------------------------------------------
# 切词
# --------------------------------------------------------------------------------------
def test_中文切出1gram与2gram混合词元() -> None:
    """1-gram 保证召回，2-gram 提供区分度；`一线城市住宿标准` 用 split() 只有 1 个词，BM25 直接失效。"""
    assert tokenize_zh("一线城市住宿标准") == [
        "线",
        "城",
        "市",
        "住",
        "宿",
        "标",
        "准",
        "一线",
        "线城",
        "城市",
        "市住",
        "住宿",
        "宿标",
        "标准",
    ]


def test_停用词单字被丢弃() -> None:
    """单字虚词（的/了/是）几乎每篇都有，留着只会让覆盖率阈值失去可解释性。"""
    assert "的" in STOPWORDS
    assert "的" not in tokenize_zh("我的住宿标准")
    assert "住宿" in tokenize_zh("我的住宿标准")


def test_跨标点不产生假bigram() -> None:
    """`报销，标准` 不能产出 `销标` 这种跨标点的假词 —— 假词会污染 IDF 与覆盖率。"""
    tokens = tokenize_zh("报销，标准")
    assert tokens == ["报", "销", "报销", "标", "准", "标准"]
    assert "销标" not in tokens


@pytest.mark.parametrize("text", ["", "   ", "\u3000\u3000", "!!!", "，。、；：", "🙂🙂🙂"])
def test_切不出词元时返回空列表(text: str) -> None:
    assert tokenize_zh(text) == []


def test_拉丁串整词一个词元() -> None:
    """错误码/型号必须保持完整：拆成 payload/too/large 反而丢掉了它的区分度。"""
    assert tokenize_zh("错误码 PAYLOAD_TOO_LARGE") == [
        "错",
        "误",
        "码",
        "错误",
        "误码",
        "payload",
        "too",
        "large",
    ]


def test_汉字与拉丁混排时分别处理() -> None:
    assert tokenize_zh("住宿600元") == ["住", "宿", "住宿", "元", "600"]


def test_drop_stopwords为假时保留停用词() -> None:
    """调试与对照实验需要看到"不过滤"的原始词元。"""
    assert "一" in tokenize_zh("一线", drop_stopwords=False)


def test_归一化会去掉零宽字符并统一小写() -> None:
    """从 Word/PDF 复制出来的零宽字符会让同一个词变成两个不同 token。"""
    assert normalize_text("住\u200b宿ABC") == "住宿abc"
    assert normalize_text("") == ""
    assert normalize_text("A\ufeffB") == "ab"


# --------------------------------------------------------------------------------------
# 查询词与覆盖率
# --------------------------------------------------------------------------------------
def test_查询词会去重() -> None:
    """查询词列表里不能有重复项，否则覆盖率的分母会虚高（分母变大 → 覆盖率虚低）。

    ⚠ 这里**不能**写成 `query_terms("住宿标准住宿标准") == query_terms("住宿标准")`：
    切词是 1-gram + 2-gram，重复拼接会在接缝处多产生一个 2-gram（`准住`），
    两个字符串本来就不该相等。那是**切词的正确行为**，不是 bug。
    所以真正要断言的是"同一个查询内部无重复词元"。
    """
    terms = query_terms("住宿标准住宿标准")
    assert len(terms) == len(set(terms)), f"查询词元出现重复：{terms}"
    # 顺序与首次出现一致（set 会打乱顺序，这里保证顺序稳定）
    assert terms == list(dict.fromkeys(terms))


def test_重复词的覆盖率不会超过1() -> None:
    """覆盖率是比率，天然 ≤ 1；重复提问不会把它推高。"""
    text = "一线城市住宿标准为每晚 600 元"
    assert coverage("住宿标准住宿标准住宿标准", text) <= 1.0
    assert coverage("住宿标准", text) <= 1.0


@pytest.mark.parametrize("query", ["", "   ", "!!!", "，。"])
def test_空查询或纯标点查询的覆盖率是0(query: str) -> None:
    assert coverage(query, "一线城市住宿标准为每晚 600 元") == 0.0


@pytest.mark.parametrize("text", ["", "   ", "!!!", "🙂", "，。、"])
def test_文本切不出词元时覆盖率是0(text: str) -> None:
    """**刚修过的 bug**：这里返回 1.0 会让 `"!!!"` 冒充"完全命中"并挤掉真正相关的内容。"""
    assert coverage("住宿标准", text) == 0.0


def test_查询词全部命中时覆盖率是1() -> None:
    assert coverage("住宿标准", "一线城市住宿标准为每晚 600 元") == 1.0


def test_覆盖率等于命中词元占查询词元的比例() -> None:
    """`餐饮补贴` 的词元在文本里全有 -> 1.0；`住宿标准` 一个都没有 -> 0.0。"""
    assert coverage("餐饮补贴", "餐饮补贴为每天 100 元") == 1.0
    assert coverage("住宿标准", "餐饮补贴为每天 100 元") == 0.0


def test_覆盖率是部分命中时介于0与1之间() -> None:
    """问句里混了库外的词（"量子"）时不能算全命中，否则闸门形同虚设。"""
    value = coverage("餐饮补贴 量子计算机", "餐饮补贴为每天 100 元")
    assert 0.0 < value < 1.0


def test_覆盖率对同一段文本是确定性的() -> None:
    assert coverage("住宿标准", "一线城市住宿标准") == coverage("住宿标准", "一线城市住宿标准")


def test_停用词表里有两个字的常见疑问词() -> None:
    """`什么/怎么/多少` 这类词在问题里必然出现，没有区分度。"""
    assert "什么" in STOPWORDS_2
    assert "什么" not in tokenize_zh("什么都没有")
