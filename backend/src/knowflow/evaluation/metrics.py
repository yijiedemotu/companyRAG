"""评测指标：纯函数、无 IO、可单测。

**一句话定位**：把"检索/回答得好不好"变成 0~1 的实数，供消融实验与 CI 门槛使用。

**在链路中的位置**：
``evaluation.harness`` 跑完一次评测 -> **本模块算指标** -> ``eval_runs.metrics_json``
-> ``/eval/compare`` 做显著性检验（``stats.py``）。

**关键设计取舍**：

1. **纯函数、零依赖、无 IO**：指标算错是最隐蔽的 bug（图形看起来对，结论全反），
   所以这里不碰数据库、不碰模型，输入输出都是普通 list/float，能用手算值逐条对照。
2. **所有函数先去重再算**（`_dedup`）：检索层返回重复 chunk 时（多路召回合流很容易
   出现），不去重会让 `recall@k` 虚高、"命中数"超过真实命中数。这是**指标虚高**
   最容易被忽略的一处。
3. **无标准答案的用例约定为 0 分**（`relevant` 为空 -> `0.0`，而不是 `1.0` 或抛异常）：
   评测集里允许出现"这条题本来就没有必中片段"的用例，约定 0 分能让它**拉低**
   平均分、逼标注者补全，而不是被静默当成满分（那会让指标系统性偏乐观）。
4. **切词口径与 BM25 同源**：`tokenize` 复制了 ``retrieval/text.py`` 的**切分策略**
   （同一套分隔符正则、同一批汉字区段、同样的"连续汉字段滑窗"），
   即 **中文 bigram + 英文小写词**。

   ⚠ **与线上闸门 `retrieval.text.coverage` 有一处刻意的差异，必须知道**：
   线上切词是 **1-gram + 2-gram 混合 + 停用词过滤**，本模块是**纯 2-gram、不过滤停用词**。
   后果：同一条 query/文本对，本模块算出的覆盖率**数值会比线上闸门低**
   （分母少了那些单字词元），**不能把本函数的输出直接和 `KEYWORD_MIN_COVERAGE=0.25`
   比大小**。要做阈值标定时应当调用线上实现 `retrieval.text.coverage`；
   本函数面向的是**评测指标**（同一套口径内部可比、可回归）。

   之所以不去 `import retrieval.text`，是为了让 evaluation 包保持**零项目内依赖**：
   它必须能在最干净的环境里被单独导入/单测（评测层不该因为检索层动了
   一行代码就跟着挂）。代价就是需要人工保证上面这套切分策略同步——
   `backend/tests/test_tokenizer_parity.py` 里有对照用例，改任一侧会立刻暴露。
    ⚠ 那个测试是**补上来的**：本 docstring 原先指向 `backend/scripts/_tmp_check_eval_metrics.py`，
    但它是临时验证脚本、已经被删除 —— 于是「改任一侧会立刻暴露」变成了一句空头支票。
    放 pytest 里、进 CI，才叫真的有保护（这是「文档里的承诺必须有测试兜住」的具体例子）。
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence

# --------------------------------------------------------------------------------------
# 切词：与 retrieval/text.py 同一套策略（见模块 docstring 第 4 条）
# --------------------------------------------------------------------------------------
#: 需要被当作分隔符的字符：空白、全角空格、ASCII 标点、中文标点、常见符号。
#: 与 `retrieval/text.py` 的正则逐字一致——**改一处就必须改另一处**。
_SEPARATOR = re.compile(
    r"[\s\u3000"
    r"\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e"
    r"\u3001-\u3003\u3008-\u3011\u3014-\u301f"
    r"\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65"
    r"\u2018\u2019\u201c\u201d\u2014\u2026\u00b7"
    r"]+"
)

#: 中文字符（含扩展区）与日韩汉字。用连续段而不是逐字判断，
#: 是为了**不产生跨语言垃圾词元**（`销s`、`1报`），那类词元会污染 BM25 的 IDF。
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]+")

#: 拉丁字母数字串（英文单词、错误码、型号、数字）。
#: `[a-z0-9_]+` 让 `kb_id` 这类标识符成为一个整体词元。
_LATIN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """中文 bigram + 英文小写词的切词函数。

    **必须与 BM25 用同一套切词，否则阈值不可比**（详见模块 docstring 第 4 条）。

    规则：

    - 先整体 `lower()`、去掉零宽字符（BOM/零宽空格会安静地毁掉第一条数据）；
    - 按分隔符切段，**段内**才滑窗，所以 `报销，标准` 不会产出 `销标` 这种跨标点假词；
    - 连续汉字段取相邻 **2-gram**；长度为 1 的汉字段保留该单字
      （否则单字查询"税"覆盖率恒为 0，闸门会把这类查询全部拦掉）；
    - 拉丁串按 `[a-z0-9_]+` 取整词；
    - 其余字符（标点、空白、全角符号）直接丢弃，不产生词元。

    保留重复词元（不去重）：BM25 需要词频，覆盖率侧自己 `_dedup`。
    """
    if not text:
        return []
    normalized = text.lower().replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    if not normalized:
        return []

    tokens: list[str] = []
    for segment in _SEPARATOR.split(normalized):
        if not segment:
            continue
        for run in _CJK_RUN.finditer(segment):
            chunk = run.group(0)
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        tokens.extend(_LATIN.findall(segment))
    return tokens


def _dedup(items: Sequence[str] | Collection[str]) -> list[str]:
    """保序去重。用 dict 而不是 `set()`：需要稳定顺序（便于调试与复现）。"""
    return list(dict.fromkeys(str(item) for item in items))


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Recall@k = |top-k 命中| / |relevant|。

    边界行为（评测点）：

    - `k <= 0` -> `0.0`（截断到空集合，没检索就没有召回）；
    - `relevant` 为空 -> `0.0`（**刻意不是 1.0**：无标准答案的用例约定 0 分，
      见模块 docstring 第 3 条）；
    - 检索结果少于 k 条：不补空、不惩罚，直接按实际条数算（真实系统就是这样）；
    - 重复检索结果先去重，否则同一片段出现两次会让 recall 虚高。
    """
    if k <= 0:
        return 0.0
    relevant = set(_dedup(relevant_ids))
    if not relevant:
        return 0.0
    top_k = set(_dedup(retrieved_ids)[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved_ids: Sequence[str], relevant_ids: Collection[str]) -> float:
    """MRR = 第一个命中项的排名倒数（1/rank，rank 从 1 开始）。

    没命中返回 `0.0`；`relevant` 为空同样返回 `0.0`（约定 0 分）。
    MRR 只看**第一条**相关结果，衡量"用户第一眼看到的东西对不对"，
    与 Recall 互补（Recall 看"有没有捞全"）。
    """
    relevant = set(_dedup(relevant_ids))
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(_dedup(retrieved_ids), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """二值相关性的 nDCG@k。

    ``DCG = Σ rel_i / log2(i + 1)``（i 从 **1** 开始，即第一名除以 log2(2)=1）；
    ``IDCG`` 按"理想排序"算：前 `min(k, |relevant|)` 个位置为 1，其余为 0。

    nDCG 与 Recall 的差别：它不仅看"捞到没"，还看**捞到的排第几**——
    把相关结果压到第 10 位和第 1 位，nDCG 会差很多。这正是重排（rerank）的效果度量。

    边界：`k <= 0`、`relevant` 为空、`IDCG == 0` 一律返回 `0.0`（不抛异常）。
    """
    if k <= 0:
        return 0.0
    relevant = set(_dedup(relevant_ids))
    if not relevant:
        return 0.0

    top_k = _dedup(retrieved_ids)[:k]
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate(hits: Sequence[bool]) -> float:
    """命中率 = 命中用例数 / 总用例数。空输入返回 `0.0`。

    与 Recall 的区别：HitRate 是**按用例**平均（每条题只要有一条相关就算 1），
    Recall 是**按相关片段**平均。评测集里"一题多个必中片段"时两者会明显不同，
    所以两个都报，避免用单一数字掩盖问题。
    """
    if not hits:
        return 0.0
    return sum(1 for item in hits if item) / len(hits)


def citation_precision(cited_ids: Sequence[str], relevant_ids: Collection[str]) -> float:
    """引用精确率 = 答案真正引用到的相关片段数 / 引用总数。

    衡量"答案引的 [1][2] 是不是真有依据"。引用为空返回 `0.0`
    （没引用的答案在本项目里等于"没有依据"，按 0 分处理；
    如果当成满分，模型只要不引用就能刷高这个指标）。
    重复引用先去重：同一片段引两次不应该算两次精确。
    """
    cited = _dedup(cited_ids)
    if not cited:
        return 0.0
    relevant = set(_dedup(relevant_ids))
    if not relevant:
        return 0.0
    return len([doc_id for doc_id in cited if doc_id in relevant]) / len(cited)


def keyword_coverage(query: str, text: str) -> float:
    """查询词在文本中的覆盖率（"有多少比例的查询词元出现在文本里"）。

    口径：

    - 查询侧：`tokenize` 后**去重**，作为分母；
    - 文本侧：`tokenize` 后放进**集合**再判成员。
      这里刻意不用"子串包含"：BM25 的词元是 n-gram，子串判断会把
      `住宿标准不报销` 里的假词也算命中，而倒排索引只按**整词元**匹配，
      两边会算出不同的覆盖率。

    空 query / 切不出词元 -> `0.0`（约定：没有信号就不能当作"通过闸门"）。

    ⚠ 与线上闸门 `retrieval.text.coverage`（1-gram + 2-gram + 停用词过滤）
    的口径差异见模块 docstring 第 4 条：**本函数输出不能直接与
    `KEYWORD_MIN_COVERAGE` 阈值比大小**。
    """
    terms = _dedup(tokenize(query))
    if not terms:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    matched = sum(1 for term in terms if term in text_tokens)
    return matched / len(terms)


__all__ = [
    "citation_precision",
    "hit_rate",
    "keyword_coverage",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "tokenize",
]
