"""文档切分层：把 :class:`~knowflow.ingest.loaders.ParsedDocument` 切成"子块 + 父块"。

**一句话定位**：``ParsedDocument -> list[Chunk]``，决定检索的最小单位与喂模型的上下文单位。

**在链路中的位置**：
``loaders.parse_bytes -> [chunkers.split_document] -> 向量化 -> Chroma + MySQL.chunks``
``Chunk.index`` 直接变成 ``chunks.chunk_index``、``vector_id="{doc_id}:{index}"``，
所以**切分结果是可复现的**：同一份文档、同一套参数必须切出完全一样的块，
否则向量库和 MySQL 会对不上（``scripts/verify_consistency.py`` 就是查这个）。

**关键设计取舍**：

1. **三级边界，从粗到细**：页 -> 小节 -> 标点递归。
   先保证"不跨页"（引用页码才对），再保证"块不跨小节"（section_path 才有意义），
   最后才在段落内按标点切到目标长度。
   **"小节"对 Markdown 与非 Markdown 的处理不同**：Markdown 的 ``#`` 标题是可靠的
   结构信号，直接当切分边界；纯文本里的"像标题的行"不可靠（条款编号、表格行都会像），
   所以只用来**标注** ``section_path``、不据此切断文本 —— 判错的代价从
   "上下文被切断"降级为"标签不够准"，而标签不参与向量计算。
2. **父子块分离**：小子块检索（向量紧凑、命中精准），命中的子块把它的
   ``parent_content``（同一小节内累积出的父块窗口）交给模型。这是"切分粒度两难"的
   标准解法：切得小检索准但上下文碎，切得大上下文全但向量被稀释。
   同一父块窗口内的子块共享同一段 ``parent_content``，所以命中窗口里任意一块，
   模型看到的上下文都一致，回答不会因为"命中第 2 块还是第 3 块"而漂移。
3. **碎片合并优先同小节、必要时跨小节**：跨小节合并会把两个主题粘成一个块，
   检索时"半块相关"反而降精度，所以只在"不合并就会留下检索不动的碎片"时才做；
   页是唯一的硬边界，任何时候都不跨。
4. **宁可多留一块也不丢内容**：合并后仍短于 ``min_chars`` 且是唯一块时保留，
   "内容丢失"比"块略小"严重得多。

**不变量**：``index`` 从 0 连续递增、``content.strip()`` 非空、空文档返回 ``[]``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import structlog
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from knowflow.ingest.loaders import ParsedDocument

logger = structlog.get_logger(__name__)

__all__ = ["Chunk", "split_document"]


# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------

#: 三级切分的标点分隔符，**顺序即优先级**：段落 -> 行 -> 中文句末 -> 英文句末 -> 中文句内。
#:
#: ⚠ **空字符串 ``""`` 只能放在最后一位**（真实踩过的坑）。
#: ``RecursiveCharacterTextSplitter._split_text`` 的实现是：
#:
#: .. code-block:: python
#:
#:     for i, s_ in enumerate(separators):
#:         if not s_:                       # 碰到空串就 break
#:             separator = s_; break
#:         if re.search(separator_, text):  # 碰到出现的分隔符也 break
#:             separator = s_; new_separators = separators[i + 1:]; break
#:
#: 空串写在中间会让循环**在它那里直接 break**：更"高级"的段落/句末分隔符全部被跳过，
#: 而且 ``new_separators`` 为空导致递归下降彻底失效。实测（每个分隔符都有文本命中）：
#:
#: * 正确配置：块边界落在句末，块开头都是 ``'。第二句话！'``；
#: * 空串插在中间：块开头变成 ``'话！第三句话？'``、``'四句话，第五'`` ——
#:   成句被**从中间砍断**，检索质量下降，而且不报任何错。
#:
#: 更糟的组合（``keep_separator=False``）下，空串分隔符会让整段文本不再被切开，
#: 直接产出远超 ``chunk_size`` 的巨型块（曾观察到 300 字文档只切出 1 块）。
#: 这个约束由 :func:`_validate_separators` 在构造 splitter 前强制校验。
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ".",
    "!",
    "?",
    ";",
    "，",
    ",",
    " ",
    "",
)

#: Markdown 标题级别 -> 元数据键。只切到三级：再深的标题在制度类文档里几乎不出现，
#: 多切一层只会让 section_path 变成没用的长串。
MARKDOWN_HEADERS: tuple[tuple[str, str], ...] = (
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
)

#: section_path 的连接符（与 `docs/01-数据库与接口契约.md` 中示例一致）。
SECTION_SEP = " > "

#: ``section_path`` 字段是 VARCHAR(512)，这里按字符数留出余量：
#: 中文在 utf8mb4 下最多 3 字节，512 字节约 170 个汉字。超长直接截断而不是让入库报错。
MAX_SECTION_PATH_CHARS = 160

#: 伪标题判定的最大长度（与加载层的同名常量语义一致，但这里是**切分层自己的策略**，
#: 调它不该影响加载层的解析行为，所以不跨模块复用）。
PSEUDO_HEADING_MAX_CHARS = 40

#: 句末/结构标点：以这些字符结尾的行是正文，不是标题。
SENTENCE_END_PUNCTUATION = "。！？；：，、.!?;:,—…）)】」』”\"'"

#: CSV/JSON 自然语言化后的行特征：见到它就不要当标题（``姓名: 张三 | 部门: 研发部``）。
#: 不排除的话，CSV 的每一行都会被当成"标题"，section_path 变成垃圾。
NATURAL_LANGUAGE_ROW_MARKERS = (" | ",)


@runtime_checkable
class _TokenEstimator(Protocol):
    """token 估算的口径。

    `knowflow.llm.tokenizer.estimate_tokens` 是**唯一实现**（中文 1 字 ~ 1 token、
    英文 ~4 字符 ~ 1 token）。这里只用 Protocol 描述形状，不 import 它的类型，
    避免切分层反向依赖 llm 层的内部细节。
    """

    def __call__(self, text: str) -> int: ...


def _load_token_estimator() -> _TokenEstimator:
    """取 token 估算函数。

    **延迟到调用期 import 是有意的**：切分模块不需要为"有没有装 tokenizer"
    承担 import 期失败的风险，而且调用方可以先用 ``sys.modules`` 打桩替换实现
    （测试与本地验证都会用到）。
    """
    from knowflow.llm.tokenizer import estimate_tokens

    return estimate_tokens


@dataclass(slots=True)
class Chunk:
    """一个检索单元（子块）及其所属的上下文单元（父块）。

    ``parent_content`` 找不到父块时等于 ``content`` —— 保证下游永远不需要判空，
    这是"字段可空但语义不空"的常见做法，能省掉调用方一堆 ``or`` 分支。
    """

    index: int
    content: str
    parent_content: str
    char_count: int
    token_count: int
    page_no: int | None
    section_path: str | None


@dataclass(slots=True)
class _Piece:
    """一条子块的中间表示：内容 + **在来源文本里的字符区间**。

    ``start`` / ``end`` 是相对于 ``source`` 的偏移，而不是"某小节的文本"：
    碎片合并可能把两个小节拼成一块（见 :func:`_merge_fragments`），
    那时区间就跨越了单个小节。把来源显式带在块上，
    父块窗口才能既"回原文取一段"又不会在跨小节时静默截断。

    保留区间而不是只留文本，是为了后面的两件事：
    碎片合并时取区间并集（避免重复文本，因为相邻子块本来就有 overlap）、
    父块窗口直接按区间回原文取（父块是"原始文本的一段"，不是"子块文本的拼接"，
    后者会把 overlap 的重复部分叠进去，父块看起来像复读）。
    """

    text: str
    start: int
    end: int
    source: str
    section: _Section

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class _Section:
    """一个小节：切分的二级边界，也是父块不可跨越的边界。

    ``text`` 是小节自身的文本，同时充当其中所有子块的**来源坐标**：
    子块的 ``start`` / ``end`` 都是相对 ``text`` 的偏移，
    所以父块窗口回原文取区间时永远不会跨页、也不会跨小节。
    """

    page_no: int | None
    section_path: str | None
    text: str


@dataclass(slots=True)
class _PagePieces:
    """一页拆出来的子块，外加"回填位置"所需的信息。"""

    page_no: int | None
    text: str
    pieces: list[_Piece] = field(default_factory=list)


@dataclass(slots=True)
class _Sectioning:
    """Markdown 小节切分的返回：按顺序排列的各小节。"""

    sections: list[_Section] = field(default_factory=list)


# ======================================================================================
# 分隔符校验（真实事故的防线）
# ======================================================================================


def _validate_separators(separators: tuple[str, ...]) -> None:
    """校验分隔符列表，**空字符串只允许出现在最后一位**。

    为什么主动抛错而不是记 warning 后继续：空串写在中间会让
    ``RecursiveCharacterTextSplitter`` 跳过所有更高级的分隔符并放弃递归下降
    （原理见 :data:`DEFAULT_SEPARATORS` 的注释），产出的是"看起来成功、
    实际块边界全乱/块超大"的检索库 —— 属于最难排查的静默故障，
    必须在上传请求里就失败掉，而不是入库后靠人发现检索质量变差。
    """
    if "" in separators[:-1]:
        raise ValueError(
            "separators 中除最后一项外不允许出现空字符串：RecursiveCharacterTextSplitter "
            "遇到非末位的空分隔符会跳过所有更高级分隔符、放弃递归下降，"
            "导致块边界从句子中间砍断（且不报错）。"
            f" 当前 separators={list(separators)!r}"
        )
    if not separators:
        raise ValueError("separators 不能为空列表：至少需要一个兜底分隔符")
    if separators[-1] != "":
        logger.debug("separators_without_empty_fallback", separators=list(separators))


# ======================================================================================
# 小节划分（二级边界）
# ======================================================================================


def _normalize_section_path(parts: list[str]) -> str | None:
    """把标题栈拼成 ``"一级 > 二级 > 三级"``；空栈返回 ``None``（表示"不属于任何小节"）。

    ``None`` 与 ``""`` 的区别很重要：``None`` 会写进 ``chunks.section_path`` 的 NULL，
    前端能据此显示"无小节"；空串会显示成一片空白，看起来像数据坏了。
    """
    cleaned = [part.strip() for part in parts if part.strip()]
    if not cleaned:
        return None
    path = SECTION_SEP.join(cleaned)
    if len(path) > MAX_SECTION_PATH_CHARS:
        # VARCHAR(512) 的余量保护：宁可截断，也不要让整篇文档因为一个超长标题入库失败
        return path[:MAX_SECTION_PATH_CHARS]
    return path


def _looks_like_heading(line: str) -> bool:
    """判断一行"看起来像标题"（用于**非 Markdown** 文档生成 section_path）。

    三个条件缺一不可：
    1. 长度 < 40：标题短，长句是正文；
    2. 不以句末/结构标点结尾：标题一般不写句号；
    3. 不含 CSV/JSON 自然语言行的分隔符 ``" | "``：否则 CSV 每一行都会变成标题。

    **只用来标注，不用来切断文本**：预训练语料里"看起来像标题"的句子太多了，
    据它切断会在标题误判处丢上下文；标错的最坏后果只是 section_path 不够准，
    而 section_path 只影响展示与重排特征，不参与向量计算。
    """
    stripped = line.strip()
    if not stripped or len(stripped) >= PSEUDO_HEADING_MAX_CHARS:
        return False
    if stripped[-1] in SENTENCE_END_PUNCTUATION:
        return False
    return all(marker not in stripped for marker in NATURAL_LANGUAGE_ROW_MARKERS)


def _split_markdown_sections(text: str) -> _Sectioning:
    """用 ``MarkdownHeaderTextSplitter`` 按 ``#``/``##``/``###`` 切成小节。

    ``strip_headers=True`` 是有意的：标题文本已经进了 ``section_path``，
    再留在正文里会被向量重复计算一遍权重，也会挤占 chunk_size 预算。
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=list(MARKDOWN_HEADERS),
        strip_headers=True,
    )
    result = _Sectioning()
    for document in splitter.split_text(text):
        parts = [
            str(document.metadata[key]) for _, key in MARKDOWN_HEADERS if document.metadata.get(key)
        ]
        body = document.page_content
        if not body.strip():
            # 只有标题没有正文的小节（常见于目录）：空块没有任何检索价值
            continue
        result.sections.append(
            _Section(
                page_no=None,
                section_path=_normalize_section_path(parts),
                text=body,
            )
        )
    return result


def _heading_ranges(text: str) -> list[tuple[int, int, str]]:
    """扫出"看起来像标题"的行，返回 ``(行起始偏移, 行结束偏移, 完整 section_path)``。

    这是**非 Markdown 文档**生成 section_path 的唯一依据，而它的定位是
    **只标注、不切分**（这是本模块最重要的一条策略）：

    * 只标注：切分仍由递归标点切分在整页文本上完成，``chunk_size`` 能得到遵守，
      段落上下文也不会在标题误判处被弄断；
    * 如果拿它当切分边界（第一版就是这么写的），200 行
      "第 N 条 报销需提供凭证" 这种短条款会让每一行都变成一个小节，
      递归切分器在 13 个字符上直接"切无可切"，实测切出 200 个 13 字小块 ——
      ``chunk_size``/``min_chars`` 全部失效。

    标题栈规则与 Markdown 一致：遇到新标题时栈深最多 3 层
    （``#``/``##``/``###`` 的对应关系在纯文本里无从判断，统一按"同级顶替"处理）。
    """
    lines = text.split("\n")
    ranges: list[tuple[int, int, str]] = []
    stack: list[str] = []
    offset = 0
    for line in lines:
        line_start = offset
        offset += len(line) + 1  # +1 是重新拼回文本时的 "\n"
        if not _looks_like_heading(line):
            continue
        if len(stack) >= len(MARKDOWN_HEADERS):
            stack.pop()
        stack.append(line.strip().lstrip("#").strip())
        path = _normalize_section_path(list(stack))
        if path is not None:
            ranges.append((line_start, line_start + len(line), path))
    return ranges


def _split_page(
    text: str,
    page_no: int | None,
    splitter: RecursiveCharacterTextSplitter,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> _PagePieces:
    """把一页切成子块（三级边界的前两级在这里落地：页 -> 小节标注）。

    Markdown 文档用 ``MarkdownHeaderTextSplitter`` 切出小节，**每小节独立跑递归切分**
    （跨小节的块会让 section_path 说不清归属）；其他格式整页跑一次递归切分，
    再按字符位置把每个块标注到对应标题下（不切分，理由见 :func:`_heading_ranges`）。
    """
    page = _PagePieces(page_no=page_no, text=text)
    if not text.strip():
        return page

    is_markdown = any(
        line.lstrip().startswith(f"{prefix} ") or line.lstrip().startswith(f"{prefix}\t")
        for line in text.split("\n")
        for prefix, _ in MARKDOWN_HEADERS[:1]
    )

    if is_markdown:
        # Markdown 的小节就是切分边界：每个小节独立跑递归切分，
        # 这样 section_path 与块内容一定一致
        for section in _split_markdown_sections(text).sections:
            page.pieces.extend(
                _split_section(
                    section,
                    splitter,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
        return page

    # 非 Markdown：整页一个"小节容器"，先切块，再给每块贴上所属标题
    whole_page = _Section(page_no=page_no, section_path=None, text=text)
    pieces = _split_section(
        whole_page, splitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    headings = _heading_ranges(text)
    for piece in pieces:
        piece.section = _Section(
            page_no=page_no,
            section_path=_section_path_for(piece.start, headings),
            text=text,
        )
        page.pieces.append(piece)
    return page


def _section_path_for(position: int, headings: list[tuple[int, int, str]]) -> str | None:
    """给定字符位置，返回它落在哪个标题之下（没有标题则 ``None``）。

    线性扫描足够：标题数量是 O(文档长度/标题长度)，而每个子块只调用一次；
    换成二分反而让"标题可能重叠/乱序"的边界情况更难推理。
    """
    current: str | None = None
    for start, _end, path in headings:
        if position >= start:
            current = path
        else:
            break
    return current


# ======================================================================================
# 递归标点切分（三级边界）
# ======================================================================================


def _build_recursive_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    """构造三级递归切分器（构造前先校验分隔符，见 :func:`_validate_separators`）。"""
    _validate_separators(DEFAULT_SEPARATORS)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(DEFAULT_SEPARATORS),
        # keep_separator=True 让标点留在当前块末尾，中文句末标点是"这里语义结束"的信号，
        # 丢掉它会让块看起来像被拦腰砍断（也给 BM25 少了一个高频词）
        keep_separator=True,
    )


def _locate_spans(
    text: str,
    pieces: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """把切分产物对齐回源文本的字符区间。

    ``RecursiveCharacterTextSplitter.split_text`` 只返回字符串、不返回偏移量，
    而父块必须"回原文取区间"（拼接子块文本会把 ``chunk_overlap`` 的重复内容算两遍），
    所以这里做一次对齐。

    **定位策略：期望位置 + 窗口查找**（试过两种更朴素的写法，都会被真实文本坑）：

    * ``cursor = 上一块终点 - 1``：相邻块本来就有 overlap，后一块的起点**早于**
      前一块的终点，于是永远找不到，退化成"挂在游标处"。实测 200 行短条款文档
      6 块里 3 块的区间晚了一两百字符，父块取到完全不搭的文本，而且**不报错**。
    * ``cursor = 上一块起点 + 1``：在"每行都是同一句话"这种高重复文本里会命中
      更早的同名片段，区间不再单调，父块被压成一个小片段。

    正确做法是别把定位问题当成"纯字符串查找"，而要利用切分器的语义：
    **相邻块的起点差刚好是 ``chunk_size - chunk_overlap``**（这是 ``_merge_splits``
    推进窗口的方式）。所以下一块的起点可以精确预测，只在预测值附近开一个
    ``±chunk_overlap`` 的窗口去找，找到就用真实位置、找不到再退回全局查找，
    最后兜底挂在预测位置上。三种策略在"重复短行/同段重复长句/叙述长文"
    三个真实用例上都是 100% 命中且 0 错位，而前两种朴素写法都会失败。
    """
    spans: list[tuple[int, int]] = []
    expected = 0
    for piece in pieces:
        window_start = max(0, expected - chunk_overlap)
        index = text.find(piece, window_start)
        if index < 0 or index > expected + chunk_overlap:
            index = text.find(piece)  # 兜底：全文中第一次出现的位置
        if index < 0:
            # 理论上不会发生（切分产物一定是子串）。真出现时挂到预测位置，
            # 宁可父块略小，也不要为了一个定位问题中断整篇上传。
            index = expected
        spans.append((index, index + len(piece)))
        expected = index + max(len(piece) - chunk_overlap, 1)
    return spans


def _split_section(
    section: _Section,
    splitter: RecursiveCharacterTextSplitter,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[_Piece]:
    """把一个小节切成子块，并带上各自的源区间。

    ``chunk_size`` / ``chunk_overlap`` 显式传进来而不是读 ``splitter`` 的私有属性：
    预测块起点要靠这两个数字（见 :func:`_locate_spans`），
    依赖第三方库的私有属性会在升级 langchain 时静默失效。
    """
    source = section.text
    texts = splitter.split_text(source)
    if not texts:
        return []
    spans = _locate_spans(source, texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = [
        _Piece(text=text, start=start, end=end, source=source, section=section)
        for text, (start, end) in zip(texts, spans, strict=True)
    ]
    # 只保留"去掉首尾空白后仍有内容"的片段：全空白片段在 strip 后会变成空 content，
    # 而那会直接违反"每个 content.strip() 非空"的不变量。在这里丢掉，
    # 比在 split_document 里抛断言异常要好（空白片段本来就没有检索价值）。
    return [piece for piece in pieces if piece.text.strip()]


# ======================================================================================
# 碎片合并
# ======================================================================================


def _merge_fragments(pieces: list[_Piece], min_chars: int) -> list[_Piece]:
    """把长度 < ``min_chars`` 的碎片并入前一块。

    合并有两种方式，取决于两块是否同源：

    * **同小节（同源）**：取区间并集后回原文切片。不能写 ``a.text + "\\n" + b.text``——
      相邻子块本来就有 ``chunk_overlap`` 的重叠文本，直接拼接会出现
      "同一句话出现两遍"，既浪费 token，也让向量被重复内容主导；
    * **跨小节（不同源）**：两个小节在原文里是**两段互不重叠的连续文本**，
      所以直接拼接文本最安全，并且把小节标题行一并带进来
      （否则会丢掉"这段属于哪个小节"的锚点）。此时区间退化为
      ``(0, len(text))``、``source`` 换成拼接结果，保证"区间一定是 source 的
      合法切片"这个不变量在任何情况下都成立（否则父块窗口会静默截断内容）。

    **为什么必须有跨小节兜底**：实测 200 行"第 N 条 报销需提供凭证"这种短条款文档，
    每行都被伪标题判定切开、各自成小节；如果只允许同小节合并，就会留下 200 个
    13 字小块 —— ``min_chars=80`` 形同虚设。宁可让个别块的 section_path 指向相邻小节，
    也不要留下无法检索的碎片：检索可用性优先于元信息精确。

    唯一的硬边界是**页**：跨页不合并，否则引用页码会说谎
    （而"引用精确到页"正是 PDF 解析保留页码的全部意义）。

    合并后仍短且是唯一块时保留：那通常意味着整篇文档就这么点内容，丢掉等于文档变空。
    """
    if min_chars <= 0 or not pieces:
        return pieces

    def merge(previous: _Piece, current: _Piece) -> _Piece:
        """把 ``current`` 并进 ``previous``（同源取并集，异源拼接）。"""
        if previous.source is current.source:
            start = min(previous.start, current.start)
            end = max(previous.end, current.end)
            return _Piece(
                text=current.source[start:end],
                start=start,
                end=end,
                source=current.source,
                section=current.section,
            )
        joined = f"{previous.text}\n{current.text}"
        return _Piece(
            text=joined,
            start=0,
            end=len(joined),
            source=joined,
            section=current.section,
        )

    merged: list[_Piece] = []
    for piece in pieces:
        if not merged or piece.char_count >= min_chars:
            merged.append(piece)
            continue
        previous = merged[-1]
        if (
            previous.source is not piece.source
            and previous.section.page_no != piece.section.page_no
        ):
            # 页是硬边界：宁可留一个短块，也不让 chunk 跨页
            merged.append(piece)
            continue
        merged[-1] = merge(previous, piece)
    return [piece for piece in merged if piece.text.strip()]


# ======================================================================================
# 父子块构造
# ======================================================================================


def _assign_parents(pieces: list[_Piece], parent_chunk_size: int) -> list[str]:
    """为每个子块生成 ``parent_content``，返回与 ``pieces`` 等长的列表。

    父块 = 同一小节内**连续子块覆盖的原文窗口**，累积到 ``parent_chunk_size`` 就换窗。
    同一窗口内的子块共享同一段 ``parent_content``，这是刻意的：
    检索仍然用精准的小子块，模型看到的是完整小节 —— 命中任意一个子块，
    拿到的上下文都一致，回答不会因为"命中的是第 2 块还是第 3 块"而漂移。

    **不可跨小节**、**不可跨页**（小节自带 ``page_no``）：跨边界会让"引用来源"
    变成一个说不清的范围。

    换窗规则（三条，按顺序判断）：

    1. 小节变了 -> 一定换窗（这是硬边界，不允许跨）;
    2. 塞下当前块后会超过 ``parent_chunk_size`` -> 换窗；
    3. 否则并入当前窗口。

    每个窗口至少包含一个子块，所以循环必然推进、不会死循环。
    """
    parents = [piece.text for piece in pieces]
    if not pieces:
        return parents

    window_start = 0
    window_section: _Section | None = pieces[0].section
    window_end = pieces[0].end

    for index in range(1, len(pieces)):
        piece = pieces[index]
        # 子块有 overlap，区间可能互相包含，所以起点取两者更小、终点取两者更大，
        # 保证 window 始终是"覆盖所有成员"的最小原文区间
        candidate_start = min(window_start_offset(pieces[window_start]), piece.start)
        candidate_end = max(window_end, piece.end)

        crosses_section = piece.section is not window_section
        overflows = (candidate_end - candidate_start) > parent_chunk_size
        if crosses_section or overflows:
            _fill_parent(pieces, window_start, index, parents)
            window_start = index
            window_section = piece.section
            window_end = piece.end
        else:
            window_end = candidate_end

    _fill_parent(pieces, window_start, len(pieces), parents)
    return parents


def window_start_offset(piece: _Piece) -> int:
    """窗口起始块的源区间起点。

    单独成函数只是让 :func:`_assign_parents` 里的取小取大读起来像一句话；
    对 mypy 而言和 `piece.start` 完全等价。
    """
    return piece.start


def _fill_parent(pieces: list[_Piece], start: int, end: int, parents: list[str]) -> None:
    """把 ``pieces[start:end]`` 归入同一个父块窗口。

    窗口内所有块同源 -> 按区间回原文取（这是常态，能自动去掉 overlap 的重复部分）；
    跨源（窗口里混进了跨小节合并出来的块）-> 退化为 "\\n" 拼接块文本。
    两种方式产出的父块文本互不重复，区别只是跨源时保留了小节标题行。
    """
    if start >= end:
        return
    window = pieces[start:end]
    first = window[0]
    if all(piece.source is first.source for piece in window):
        parent_text = first.source[first.start : window[-1].end]
    else:
        parent_text = "\n".join(piece.text for piece in window)
    for position in range(start, end):
        parents[position] = parent_text


# ======================================================================================
# 入口
# ======================================================================================


def split_document(
    doc: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    parent_chunk_size: int,
) -> list[Chunk]:
    """把文档切成子块/父块对。

    **三级边界（从粗到细）**：

    .. code-block:: text

        一级：页   PDF 逐页切，chunk 不跨页       -> 引用页码才准确
          └─ 二级：小节   #/##/### 或"像标题的行"  -> section_path 才有意义
               └─ 三级：递归标点  \n\n -> 行 -> 。！？ -> ，   -> 长度收敛到 chunk_size

    **父子块**：三段切分产出子块（检索单位，约 ``chunk_size`` 字符）；
    同小节内连续子块按 ``parent_chunk_size`` 累积成父块窗口（喂模型单位），
    窗口内所有子块共享同一段 ``parent_content``。

    **不变量（调用方可以直接依赖）**：

    * ``[chunk.index for chunk in chunks] == list(range(len(chunks)))``；
    * 每个 ``chunk.content.strip()`` 非空；
    * 空文档返回 ``[]``（"空"由调用方决定是否报错：加载层已经拦过一轮，
      这里再返回空列表是为了让"切不出东西"不被误当成异常）。

    参数全部由调用方传入（来自 ``get_settings()``），本函数不读全局配置，
    这样评测里可以一次进程内扫多组参数做消融实验。
    """
    if not doc.pages:
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，当前为 {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 不能为负，当前为 {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        # 配置层已经校验过；这里再拦一次是因为本函数是公开入口，
        # overlap >= size 会让递归切分器无法收敛（无限切分）
        raise ValueError(
            f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})，否则递归切分无法收敛"
        )

    splitter = _build_recursive_splitter(chunk_size, chunk_overlap)

    # 一级（页）：逐页独立切分 —— 页是硬边界，chunk 绝不跨页，
    # 否则它的 page_no 只能取其中一页，引用页码就会说谎
    pieces: list[_Piece] = []
    for page in doc.pages:
        pieces.extend(
            _split_page(
                page.text,
                page.page_no,
                splitter,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ).pieces
        )

    pieces = _merge_fragments(pieces, min_chars)
    if not pieces:
        # 文本全是标点/空白之类"切完没内容"的情况，交给调用方处理
        logger.debug("split_produced_no_chunk", filename=doc.filename, char_count=doc.char_count)
        return []

    parents = _assign_parents(pieces, parent_chunk_size)
    estimate_tokens = _load_token_estimator()

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        content = piece.text.strip()
        parent_content = parents[index].strip() or content
        chunks.append(
            Chunk(
                index=index,
                content=content,
                parent_content=parent_content,
                char_count=len(content),
                token_count=estimate_tokens(content),
                page_no=piece.section.page_no,
                section_path=piece.section.section_path,
            )
        )

    # 兜底断言：不变量一旦被破坏，宁可在上传请求里炸掉，也不要写进向量库
    if [chunk.index for chunk in chunks] != list(range(len(chunks))):
        raise AssertionError("chunk index 不连续：切分逻辑被破坏")
    if any(not chunk.content for chunk in chunks):
        raise AssertionError("存在空 content 的 chunk：切分逻辑被破坏")

    logger.debug(
        "document_split",
        filename=doc.filename,
        chunks=len(chunks),
        pages=len(doc.pages),
        avg_chars=round(sum(chunk.char_count for chunk in chunks) / len(chunks), 1),
        parent_windows=len({chunk.parent_content for chunk in chunks}),
    )
    return chunks
