"""中文切词：全项目唯一的实现。

**为什么必须唯一**：BM25 的打分和「查询词覆盖率」的阈值都建立在这套切词上。
如果两处各写一份，词表稍微不一致，阈值就不再可比 ——
配置里写 `KEYWORD_MIN_COVERAGE=0.25` 的含义会随实现漂移，
调参调出来的数字下次就失效。所以：**要用切词，就 import 这里。**

**为什么不用 jieba**：引入一个 30MB 词典 + 变长依赖，收益主要是词边界更准。
但 BM25 对本项目这种短文档（600 字 chunk）用 **1-gram + 2-gram** 已经足够：
- 1-gram 保证召回（任何包含相同字的片段都能被捞到，不会因分词错误漏召回）；
- 2-gram 提供区分度（"城市"、"住宿"比单字承载更多信息）；
- 且**白板可推导**——面试时能完整讲清楚，比"我调了个库"有价值。

代价（诚实说）：词表比 jieba 大 2~3 倍，索引内存占用更高；
同义词（"房费" vs "住宿费"）依然匹配不上，那部分交给向量检索兜底。
"""

from __future__ import annotations

import re
from typing import Final

# 需要被当作分隔符的字符：空白、全角空格、ASCII 标点、中文标点、常见符号
_SEPARATOR = re.compile(
    r"[\s\u3000"
    r"\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e"
    r"\u3001-\u3003\u3008-\u3011\u3014-\u301f"
    r"\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65"
    r"\u2018\u2019\u201c\u201d\u2014\u2026\u00b7"
    r"]+"
)

# 中文字符（含扩展区）与日韩汉字
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]+")

# 拉丁字母数字串（英文单词、错误码、型号、数字都归到这里）
_LATIN = re.compile(r"[A-Za-z0-9_]+")

# 停用词：单字 + 常见虚词。不过滤的话 BM25 会给"的/了/是"很高权重，
# 因为它们几乎每篇文档都有——但 BM25 的 IDF 项本就会压低高频词，
# 这里再过滤一次主要是为了**压缩词表体积**与提高覆盖率阈值的可解释性。
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # 单字虚词
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "或",
        "有",
        "我",
        "你",
        "他",
        "她",
        "它",
        "这",
        "那",
        "个",
        "们",
        "都",
        "也",
        "就",
        "还",
        "而",
        "但",
        "被",
        "把",
        "给",
        "对",
        "从",
        "到",
        "为",
        "以",
        "及",
        "等",
        "中",
        "上",
        "下",
        "里",
        "着",
        "过",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "哦",
        "嗯",
        "一",
        "不",
        "会",
        "能",
        "要",
        "可",
        # 单字疑问词（在问题里几乎必然出现，没有区分度）
        "什",
        "么",
        "怎",
        "样",
        "多",
        "少",
        "哪",
        "谁",
        "何",
        "如",
        # 英文常见虚词
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "is",
        "are",
        "and",
        "or",
        "for",
        "on",
        "at",
        "by",
        "with",
        "how",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
    }
)

# 中文常用疑问/连接双字词，作为 2-gram 时同理没有区分度
STOPWORDS_2: Final[frozenset[str]] = frozenset(
    {
        "什么",
        "怎么",
        "怎样",
        "如何",
        "多少",
        "哪些",
        "哪个",
        "是否",
        "可以",
        "请问",
        "一下",
        "我们",
        "你们",
        "他们",
        "这个",
        "那个",
        "以及",
        "但是",
        "因为",
        "所以",
        "如果",
        "那么",
        "还是",
        "或者",
        "需要",
        "应该",
        "有没有",
    }
)


def normalize_text(text: str) -> str:
    """轻量归一化：统一小写、去掉零宽字符。

    注意**不做全角转半角**：全角字符本身是有信息的（排版），
    而且转错会让"（一）"这类条款编号失去区分度。切词阶段已经能处理。
    """
    if not text:
        return ""
    cleaned = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00ad", "")
    return cleaned.lower()


def tokenize_zh(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """把一段文本切成词元列表（1-gram + 2-gram + 拉丁词）。

    例（**以下输出是实跑结果，不是手写示意**）：
        >>> tokenize_zh("一线城市住宿标准")
        ['线', '城', '市', '住', '宿', '标', '准', '一线', '线城', '城市', '市住', '住宿', '宿标', '标准']
        >>> tokenize_zh("错误码 PAYLOAD_TOO_LARGE")
        ['错', '误', '码', '错误', '误码', 'payload', 'too', 'large']

    ⚠ 两处容易被 docstring 写错、所以特意标出来的细节：

    1. **单字 `一` 被过滤掉了**（它在 `STOPWORDS` 里）。默认 `drop_stopwords=True`，
       所以"看起来应该齐整"的 1..n-gram 列表里会缺几个停用字 —— 这是有意的
       （"的/了/一"几乎每篇文档都有，留着只会撑大词表而不带区分度）。
       要看完整切分请传 `drop_stopwords=False`。
    2. **`PAYLOAD_TOO_LARGE` 被切成 3 个词**（`payload` / `too` / `large`），不是 1 个词 ——
       因为 `_` 落在 `_SEPARATOR` 的 ASCII 标点区间里。这**正合我们想要**：
       用户写 "PAYLOAD TOO LARGE"（带空格）也能命中同一个错误码。
       代价是丢掉"整体错误码"的词面精确性，但 BM25 仍会命中全部 3 个词，效果接近。

    实现要点：中文串**按连续汉字段**处理，跨标点的字不组成 bigram
    （"报销，标准" 不应产出 "销标" 这种跨标点的假词）。

    > 上面两个例子原先写的是"手推的理想输出"（`一` 没被过滤、
    > `payload_too_large` 被当成一个词），**与实跑输出不一致** ——
    > 这是写教程的 agent 用实跑对照时发现的。
    > 教训：**docstring 里的 `>>>` 示例必须来自实跑**，
    > 否则它会变成一份"看起来很权威的假文档"，比没有文档更糟。
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens: list[str] = []
    for segment in _SEPARATOR.split(normalized):
        if not segment:
            continue
        # 一个 segment 里可能同时含汉字和拉丁（如 "住宿600元"），分别处理
        for match in _CJK.finditer(segment):
            run = match.group(0)
            for char in run:
                if not drop_stopwords or char not in STOPWORDS:
                    tokens.append(char)
            for i in range(len(run) - 1):
                bigram = run[i : i + 2]
                if not drop_stopwords or bigram not in STOPWORDS_2:
                    tokens.append(bigram)
        for match in _LATIN.finditer(segment):
            word = match.group(0)
            if not drop_stopwords or word not in STOPWORDS:
                tokens.append(word)
    return tokens


def query_terms(text: str) -> list[str]:
    """查询侧用词元：去重 + 去停用词，用于「查询词覆盖率」。

    去重很关键：问题里重复出现的词不应该让覆盖率超过 1。
    """
    seen: dict[str, None] = {}
    for token in tokenize_zh(text, drop_stopwords=True):
        seen.setdefault(token, None)
    return list(seen)


def coverage(query: str, text: str) -> float:
    """查询词覆盖率 = 出现在文本中的查询词比例。

    比原始 BM25 分数更可靠的信号：BM25 分数受文档长度、词频影响，跨文档不可比；
    覆盖率是 [0,1] 的比率，可以直接当阈值用（"至少 1/4 的查询词出现过"）。

    两个边界都必须返回 0.0（**这是被测出来的 bug，不是理论洁癖**）：
    - 查询切不出词元（空串、纯标点）；
    - 文本切不出词元（纯标点、纯 emoji）。

    第二点尤其重要：如果文本切不出词元时返回 1.0，那么一条"!!!"
    会被判定为"完全命中查询"，闸门放行一条完全无意义的片段，
    而且它还能挤掉真正相关的内容。覆盖率在"没信号"时必须表达"没信号"。
    """
    terms = query_terms(query)
    if not terms:
        return 0.0
    text_tokens = set(tokenize_zh(text, drop_stopwords=True))
    if not text_tokens:
        return 0.0
    hit = sum(1 for t in terms if t in text_tokens)
    return hit / len(terms)


__all__ = [
    "STOPWORDS",
    "STOPWORDS_2",
    "coverage",
    "normalize_text",
    "query_terms",
    "tokenize_zh",
]
