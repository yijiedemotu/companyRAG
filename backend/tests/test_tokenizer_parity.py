"""切词口径一致性：`evaluation.metrics` 与 `retrieval.text` 必须同源。

**这个测试补的是一个真实缺口。** `evaluation/metrics.py` 的模块 docstring 第 4 条写着：

    切词口径与 BM25 同源…… 之所以不去 `import retrieval.text`，是为了让 evaluation 包
    保持**零项目内依赖**（它必须能在最干净的环境里被单独导入/单测）。
    代价就是需要人工保证上面这套切分策略同步 ——
    `backend/scripts/_tmp_check_eval_metrics.py` 里有对照用例，改任一侧会立刻暴露。

但那个 `_tmp_check_eval_metrics.py` 是**临时验证脚本，已经被删除**了 ——
于是这段承诺变成了空头支票：**改任一侧不会有任何东西报警**。
本文件就是那条"对照用例"的正式归宿（放 pytest 里，进 CI，才叫真的有保护）。

**要断言的两件事**：

1. **子集关系**：evaluation 的词元集合 ⊆ 线上闸门的词元集合。
   因为线上是 **1-gram + 2-gram + 停用词过滤**，evaluation 是 **纯 2-gram + 拉丁小写词**。
   所以 **evaluation 的覆盖率恒 ≤ 线上覆盖率**。
2. **确实存在数值不同的样例**：如果两者永远相等，那"不能直接比阈值"这句警告就是多余的
   ——用一个真实样例把它钉住，读者才知道差异有多大。

⚠ **绝不要**把这两个函数"统一"成同一个：它们的**用途不同**。
线上闸门要召回（1-gram 保证不漏），评测指标要**内部可比可回归**（口径稳定）。
真正要防的是"改了一侧忘了同步另一侧"，而不是"两边必须一样"。
"""

from __future__ import annotations

import pytest

from knowflow.evaluation.metrics import keyword_coverage as eval_coverage
from knowflow.retrieval.text import STOPWORDS_2, coverage as gate_coverage, tokenize_zh

#: `(query, text)` 样例：前者取自语料里的真实问法，后者取自样例文档的正文片段
SAMPLES = [
    ("什么是税", "关于个人所得税的说明"),
    ("报销标准住宿", "报销标准与住宿标准"),
    ("一线城市住宿标准", "一线城市住宿标准为每晚 600 元"),
    ("年假有几天", "员工每年可以享受几天年假"),
    ("PAYLOAD_TOO_LARGE", "请求体超过网关上限时返回 PAYLOAD_TOO_LARGE"),
]


def _eval_tokens(text: str) -> set[str]:
    """取出 evaluation 侧实际使用的词元集合。

    直接调私有函数是不好的，但这里**就是要测私有实现**：
    要断言的正是"它的切词策略与线上同源"，这在公开 API 上看不见。
    """
    from knowflow.evaluation import metrics as metrics_module

    return set(metrics_module._dedup(metrics_module.tokenize(text)))


@pytest.mark.parametrize(("query", "text"), SAMPLES)
def test_evaluation词元集合只比线上多出停用词(query: str, text: str) -> None:
    """evaluation 侧的词元，**除了 2-gram 停用词以外**，都必须能在线上的切词结果里找到。

    为什么允许停用词差集 —— 这是**实测出来的真实差异，不是随手放宽**：

    线上 `retrieval/text.py` 有一个 `STOPWORDS_2` 列表（"可以"、"是否"、"什么"…），
    `tokenize_zh` 默认会把它过滤掉；而 `evaluation/metrics.py` 为了保持
    **零项目内依赖**（不 import retrieval），没有复制这份列表。
    于是 `可以` 会出现在 evaluation 的词元里、不出现在线上的词元里。

    ⚠ **这条断言是有意写成"差集 ⊆ STOPWORDS_2"而不是"差集为空"的**：
    - 写成"差集为空"会**永远失败**（真实存在这个差集）；
    - 写成"不做断言"就**失去了保护**；
    - 限定成"差集只能来自 STOPWORDS_2"则两头都保住：
      如果哪天有人把 bigram 改成 1-gram（或改了分隔符正则），
      多出来的会是单字或跨标点的假词 —— **它们不在 `STOPWORDS_2` 里，测试立刻失败**。
    """
    gate_tokens = set(tokenize_zh(text, drop_stopwords=True))
    extra = _eval_tokens(text) - gate_tokens
    unexpected = {token for token in extra if token not in STOPWORDS_2}
    assert not unexpected, (
        f"evaluation 侧出现了既不在线上词表、也不属于 2-gram 停用词的词元：{sorted(unexpected)}；"
        f"两侧切分策略已经漂移，请同步 evaluation/metrics.py 与 retrieval/text.py"
    )


@pytest.mark.parametrize(("query", "text"), SAMPLES)
def test_evaluation覆盖率不高于线上覆盖率(query: str, text: str) -> None:
    """子集关系的直接推论：**evaluation 的覆盖率恒 ≤ 线上闸门的覆盖率**。

    这就是 `KEYWORD_MIN_COVERAGE` 阈值**不能**直接拿来和 `keyword_coverage` 输出比较的原因 ——
    拿它比会**低估**通过率，把本该放行的片段判成无关。
    """
    assert eval_coverage(query, text) <= gate_coverage(query, text) + 1e-9


def test_两个口径确实存在数值差异_所以警告不是多余的() -> None:
    """如果两个函数永远返回同一个数，那"不能直接比阈值"这句警告就是废话。

    这里用一个真实样例把它钉住：query 里的单字（"税"）在 evaluation 侧不参与计分
    （它只切 2-gram），在线上侧参与（1-gram），所以线上覆盖率更高。
    """
    query, text = "什么是税", "关于个人所得税的说明"
    eval_value = eval_coverage(query, text)
    gate_value = gate_coverage(query, text)
    assert eval_value != gate_value, (
        f"两个口径在这条样例上应当有差异（evaluation={eval_value}, gate={gate_value}）；"
        "如果确实一致了，请重新确认 metrics.py docstring 里那句警告是否还需要"
    )
    # 差异的方向是确定的：线上（1-gram + 2-gram）≥ evaluation（纯 2-gram）
    assert gate_value > eval_value


def test_两者在完全命中时都返回1() -> None:
    """边界一致性：文本完整包含 query 时，两个口径都应当是 1.0（而不是 0.9x）。"""
    query = text = "一线城市住宿标准为每晚 600 元"
    assert eval_coverage(query, text) == pytest.approx(1.0)
    assert gate_coverage(query, text) == pytest.approx(1.0)


@pytest.mark.parametrize("empty", ["", "   ", "!!!", "，。"])
def test_两侧对空查询与纯标点的约定一致_都是0(empty: str) -> None:
    """没有信号就不能当作"通过"：两侧必须都返回 0.0。

    （`retrieval.text.coverage` 曾经在"文本切不出词元"时返回 1.0，
    让 `"!!!"` 冒充"完全命中"，是修过的 bug；这条测试防止它回归。）
    """
    assert eval_coverage(empty, "一线城市住宿标准") == 0.0
    assert gate_coverage(empty, "一线城市住宿标准") == 0.0
    assert eval_coverage("住宿标准", empty) == 0.0
    assert gate_coverage("住宿标准", empty) == 0.0
