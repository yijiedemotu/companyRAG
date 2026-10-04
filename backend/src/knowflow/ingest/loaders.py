"""文档加载层：把任意受支持格式的原始字节，变成"可切分的归一化纯文本 + 元信息"。

**一句话定位**：`bytes -> ParsedDocument`，是入库链路的第一个环节。

**在链路中的位置**：
``上传 -> [loaders.parse_bytes] -> chunkers.split_document -> 向量化 -> Chroma + MySQL``
它产出的 ``page_no`` / ``encoding`` / ``warnings`` 会一路带到 ``chunks`` 表，
所以这里丢掉的元信息（尤其是页码）后面任何环节都补不回来。

**关键设计取舍**：
1. **编码嗅探按"确定性从高到低"排序**（BOM > 严格 UTF-8 > 探测器 > 中文编码兜底），
   而不是一上来就用统计探测器 —— 探测器对短文本会猜错，而 BOM 与严格 UTF-8
   的判定是**零误判**的。详见 :func:`_decode_bytes`。
2. **归一化统一在加载层做完**，切分器不再关心 ``\r\n`` / 零宽字符 / 软连字符。
   PDF 复制出来的文本里这些字符极多，放到切分层处理会让每个切分策略都要重复实现。
3. **各格式尽量保留结构语义**：PDF 逐页抽取保留页码（否则引用无法定位到页）、
   CSV 转成"列名: 值"的自然语言（否则列头与值的对应关系在向量化时丢失）、
   JSON 递归展平成"路径: 值"（否则嵌套结构变成一堆无意义括号）。
4. **单页失败不拖垮整篇**：PDF 某一页抽取抛异常时只清空该页文本并记 warning，
   因为"少一页"远好于"整篇上传失败"。
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import structlog
from pypdf import PdfReader

from knowflow.core.config import get_settings
from knowflow.core.exceptions import (
    DocumentEmptyError,
    DocumentParseError,
    UnsupportedFileTypeError,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "ParsedDocument",
    "ParsedPage",
    "detect_ext",
    "parse_bytes",
    "parse_path",
]

# --------------------------------------------------------------------------------------
# 常量：全部集中在这里，禁止函数体内散落魔法值
# --------------------------------------------------------------------------------------

#: 扩展名 -> 解析器名。paser 名会写进 `documents.parser` 字段，改这里等于改数据库枚举。
EXT_PARSERS: Final[dict[str, str]] = {
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "pdf": "pdf",
    "csv": "csv",
    "json": "json",
}

#: BOM 字节序列 -> 对应编码（**按字节前缀匹配，不是"试着 decode"**）。
#:
#: ⚠ 这里不能用"直接 `raw.decode("utf-16")` 成功就算 utf-16"的写法（第一版就是这么写的，
#: 被验证脚本抓出来了）：
#:
#: * ``utf-8-sig`` 解码**不带 BOM** 的 UTF-8 也完全成功，于是所有 UTF-8 文件都被标成
#:   ``utf-8-sig``，而 BOM 那一轮会直接吃掉严格 UTF-8 这一轮 —— 顺序设计形同虚设；
#: * ``utf-16`` / ``utf-32`` 在没有 BOM 时按本机字节序解码且**从不报错**，
#:   于是随机二进制会被"成功"解成一篇乱码汉字文本，永远走不到"无法识别编码"。
#:
#: 所以 BOM 检测必须落到字节层面：**先看字节前缀，再决定用哪个 codec**。
BOM_SIGNATURES: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

#: 无 BOM 时也绝不允许被 BOM 那一轮挑走的编码（防御性保留：接口自解释）。
BOM_ENCODINGS: Final[tuple[str, ...]] = tuple(encoding for _, encoding in BOM_SIGNATURES)

#: 探测器不可用时的中文编码兜底顺序。
#: **GB18030 必须排在 GBK 之前**：GB18030 是 GBK 的严格超集，
#: 反过来先试 GBK 时，生僻字会被解成"另一个合法汉字"而**不报错**，
#: 得到一篇看似正常、内容全错的文档（静默乱码，真实踩过的坑）。
CJK_FALLBACK_ENCODINGS: Final[tuple[str, ...]] = ("gb18030", "gbk", "gb2312", "big5")

#: PDF"没有可提取文本"的判定阈值：低于这个字符数基本等于扫描件。
PDF_MIN_EXTRACTED_CHARS: Final[int] = 20

#: JSON 展平的最大深度；超过则截断并记 warning，避免畸形文件把内存打满。
JSON_MAX_DEPTH: Final[int] = 20

#: CSV 自然语言行里列与值、列与列之间的分隔符。
#: 用 ` | ` 而不是 `,`：值本身可能含逗号，`|` 在中文语料里几乎不出现，可无歧义切回。
CSV_FIELD_SEP: Final[str] = " | "
CSV_KV_SEP: Final[str] = ": "

#: 伪标题判定的最大长度：超过这个长度的一行基本是正文而不是标题。
PSEUDO_HEADING_MAX_CHARS: Final[int] = 40

#: 句末/结构标点：以这些字符结尾的行是正文，不是标题。
SENTENCE_END_PUNCTUATION: Final[str] = "。！？；：，、.!?;:,—…）)】」』”\"'"

#: 零宽字符与软连字符：从 Word/PDF 复制文本时高频出现，
#: 它们会让同一个词在 BM25 里变成两个不同的 token（检索静默变差）。
_INVISIBLE_CHARS: Final[tuple[str, ...]] = ("\u200b", "\ufeff", "\u200e", "\u00ad")

#: 连续空行压缩：3 个以上换行 -> 2 个（保留段落边界，去掉大幅空白）。
_BLANK_LINES_RE = re.compile(r"\n{3,}")

#: Markdown 一级/二级标题（只用于判断"切分会不会退化成纯递归"）。
_MD_TOP_HEADING_RE = re.compile(r"^#{1,2}[ \t]+\S", re.MULTILINE)

#: Markdown 任意级别标题（用于判断是否值得走 MarkdownHeaderTextSplitter）。
_MD_ANY_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\S", re.MULTILINE)


@dataclass(slots=True)
class ParsedPage:
    """一页（或一个逻辑分区）的文本。

    ``page_no`` 为 ``None`` 表示该格式本身没有分页概念（md/txt/csv/json）。
    """

    page_no: int | None
    text: str


@dataclass(slots=True)
class ParsedDocument:
    """加载层的输出契约。

    ``text`` 是**归一化后的全文**（各页用 ``"\\n\\n"`` 连接），下游切分只认它；
    ``pages`` 保留页边界，是"引用能精确到页"的唯一来源。
    """

    filename: str
    ext: str
    parser: str
    text: str
    pages: list[ParsedPage]
    encoding: str | None
    line_count: int
    char_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        """逻辑分区数（PDF 即实际页数，可能大于"有文本的页数"）。"""
        return len(self.pages)


def detect_ext(filename: str) -> str:
    """从文件名取小写扩展名（不含点）；没有扩展名时返回空串。

    用 ``Path(name).suffix`` 而不是 ``rsplit(".")``：前者正确处理
    ``archive.tar.gz``、``.gitignore``（无扩展名）这类边界。
    """
    return Path(filename).suffix.lower().lstrip(".")


# ======================================================================================
# 编码嗅探
# ======================================================================================


def _detect_encoding(raw: bytes) -> str | None:
    """用统计探测器猜编码，让"严格 UTF-8 通过不了"的文件先被探测一轮。

    优先 `charset-normalizer`（比 chardet 新、对短文本更稳、chardet 已停止维护），
    缺失时退回 `chardet`；两个都没有则返回 ``None``，由调用方走 CJK 兜底顺序。
    """
    try:
        from charset_normalizer import from_bytes
    except ImportError:  # pragma: no cover - 取决于运行环境依赖
        try:
            import chardet
        except ImportError:  # pragma: no cover - 取决于运行环境依赖
            logger.warning("encoding_detector_unavailable", fallback="cjk_sequence")
            return None
        guess = chardet.detect(raw) or {}
        return guess.get("encoding") or None

    best = from_bytes(raw).best()
    return best.encoding if best is not None else None


def _decode_bytes(raw: bytes) -> tuple[str, str | None, list[str]]:
    """按"确定性从高到低"的顺序嗅探并解码，返回 ``(文本, 实际编码, 已尝试编码)``。

    **顺序就是本模块的设计核心，理由逐条如下**：

    1. **BOM 优先（按字节前缀判定，不是"试着 decode"）**：BOM 是文件自己声明的编码，
       是零误判的确定性信号，有 BOM 还去猜等于把确定信息丢掉换一个猜测。
       注意这一步必须看字节：第一版写成"逐个 codec 试着 decode，成功就用"，
       结果 ``utf-8-sig`` 对**不带 BOM** 的 UTF-8 也会成功，于是所有 UTF-8 文件都被
       判成 ``utf-8-sig``；更糟的是 ``utf-16``/``utf-32`` 在没有 BOM 时按本机字节序
       解码且从不报错，随机二进制会被"成功"解成一篇乱码，永远走不到"无法识别编码"。
       （这个 bug 是验证脚本第 1 项抓出来的，见 :data:`BOM_SIGNATURES`。）
    2. **严格 UTF-8**：中文文档（尤其 Markdown/JSON/CSV）现代基本都是 UTF-8。
       严格模式能过就一定是 UTF-8 —— UTF-8 的字节结构校验很强，
       一段 GBK 字节几乎不可能碰巧通过；
       **但是反过来不成立**：UTF-8 能解的字节序列被 GBK 解出来也是"合法汉字"，
       所以绝不能让 GBK 插到 UTF-8 前面。
    3. **统计探测器**：UTF-8 失败说明不是 UTF-8，此时让 `charset-normalizer`/`chardet`
       判断 GBK/GB18030/Big5/Shift-JIS……探测器依赖字符频率，长文本很准、极短文本会错，
       但作为第 3 顺位已经比"硬编码猜一个"好得多。
       探测器给出的名字可能系统不认识（LookupError）或解不对（UnicodeDecodeError），
       两种都吞掉继续往下走。
    4. **CJK 兜底顺序**：探测器不可用（没装依赖）或猜错时，按 GB18030 -> GBK -> GB2312 -> Big5 试。
       **GB18030 必须排在 GBK 前**：它是 GBK 的超集，先试 GBK 会把生僻字
       解成另一个合法汉字而不报错，得到静默乱码。GB2312 是 GBK 的子集，
       只在 GBK 也失败时才有意义；Big5 覆盖繁体中文（台/港的老资料）。
    5. **全失败** -> :class:`DocumentParseError`，把 ``tried`` 列表放进 detail，
       排查时能直接看到"到底试过什么"。
    """
    tried: list[str] = []

    # ---- 1. BOM：按字节前缀判定，命中才用对应 codec 解 ----
    bom_encoding = _match_bom(raw)
    if bom_encoding is not None:
        tried.append(bom_encoding)
        # utf-8-sig 自己会吞掉 BOM；utf-16-le/be 与 utf-32-le/be 不会，所以要切掉，
        # 否则 U+FEFF 会混进第一个 chunk（既污染检索，也会让文本首字符看起来是"乱码"）
        payload = raw[3:] if bom_encoding == "utf-8-sig" else raw
        return payload.decode(bom_encoding), bom_encoding, tried

    # ---- 2. 严格 UTF-8 ----
    tried.append("utf-8")
    try:
        return raw.decode("utf-8"), "utf-8", tried
    except UnicodeDecodeError:
        pass

    # ---- 3. 统计探测器 ----
    detected = _detect_encoding(raw)
    if detected:
        tried.append(detected)
        try:
            return raw.decode(detected), detected, tried
        except (LookupError, UnicodeDecodeError):
            # LookupError: 探测器返回了本机不认识的编码名
            # UnicodeDecodeError: 猜错了。两种都继续往下走，不要在这里中断。
            logger.debug("encoding_detector_missed", encoding=detected)

    # ---- 4. CJK 兜底顺序 ----
    for encoding in CJK_FALLBACK_ENCODINGS:
        try:
            decoded = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        tried.append(encoding)
        return decoded, encoding, tried

    # ---- 5. 全失败 ----
    raise DocumentParseError(
        "无法识别文件编码",
        detail={"tried": tried, "size_bytes": len(raw)},
    )


def _match_bom(raw: bytes) -> str | None:
    """按字节前缀匹配 BOM，返回应使用的编码（无 BOM 返回 ``None``）。

    长前缀优先（``FF FE 00 00`` 是 UTF-32-LE，必须在 ``FF FE``(UTF-16-LE) 之前判断，
    否则 UTF-32 的文件会被当 UTF-16 解出夹着 NUL 的乱码）。

    ``utf-16-le``/``utf-16-be``/``utf-32-le``/``utf-32-be`` 这些"指定字节序"的 codec
    **不会**吞掉 BOM，所以要在 :func:`_decode_bytes` 里切片去掉；
    而 ``utf-8-sig`` 会自己吞掉 BOM。
    """
    for signature, encoding in BOM_SIGNATURES:
        if raw.startswith(signature):
            return encoding
    return None


# ======================================================================================
# 文本归一化
# ======================================================================================


def _normalize(text: str) -> str:
    """归一化。**三步顺序有意义，不能交换**：

    1. **统一换行**（``\\r\\n`` / ``\\r`` -> ``\\n``）：必须最先做。
       否则后面按行/按段落处理时，``\\r\\n`` 会被当成两个字符，行尾多出空行，
       连"连续换行压缩"的计数都会算错。
    2. **删不可见字符**（零宽空格/零宽非连接符/LRM/软连字符）：这些字符
       不影响人眼阅读，但会让 BM25 把 ``住宿`` 和 ``住\\u200b宿`` 当成两个不同的词，
       检索命中率静默下降。必须放在换行压缩之前，避免它们把空行计数撑大。
    3. **压缩空行**（3 个以上换行 -> 2 个）：保留段落边界（Markdown 的段落语义），
       但去掉粘贴/转换产生的大片空白。

    **刻意不做** ``strip()`` 每一行：Markdown 的缩进有语义
    （代码块、列表续行、嵌套列表），strip 掉会把结构压平、让切分和展示都变形。
    这里只做末尾 ``rstrip()``（去掉文件尾部空白，它永远是噪声）。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    for invisible in _INVISIBLE_CHARS:
        if invisible in normalized:
            normalized = normalized.replace(invisible, "")

    normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.rstrip()


def _count_lines(text: str) -> int:
    """行数 = 换行符数 + 1（空字符串记 0 行）。"""
    if not text:
        return 0
    return text.count("\n") + 1


# ======================================================================================
# 各格式解析器
# ======================================================================================


def _parse_markdown(text: str) -> tuple[list[ParsedPage], list[str]]:
    """Markdown：整篇一页（``page_no=None``）。

    只做一件额外的事：检查是否存在一级/二级标题。
    没有标题时 ``MarkdownHeaderTextSplitter`` 无处可切，切分会退化成纯递归字符切分，
    ``section_path`` 全为 ``None`` —— 这不是错误，但会明显影响检索质量
    （"哪个小节的第几段"这个信息没了），所以在 ``warnings`` 里如实报出来。
    """
    warnings: list[str] = []
    if text and not _MD_TOP_HEADING_RE.search(text):
        warnings.append(
            "Markdown 中没有找到一级/二级标题，切分将退化为纯递归字符切分（无 section_path）"
        )
    return [ParsedPage(page_no=None, text=text)], warnings


def _parse_text(text: str) -> tuple[list[ParsedPage], list[str]]:
    """纯文本：整篇一页；行数由 :func:`parse_bytes` 统一按 ``\\n`` 统计。"""
    return [ParsedPage(page_no=None, text=text)], []


def _parse_pdf(raw: bytes) -> tuple[list[ParsedPage], list[str]]:
    """PDF：**逐页抽取并保留页码**（``page_no`` 从 1 开始）。

    页码是这一层最有价值的产出：没有它，回答里的引用只能定位到"某文档"，
    用户还得自己翻；有了它可以直接跳到第 N 页。

    三个刻意行为：

    * **加密 PDF 直接失败**：``PdfReader`` 对加密文件会在读取时才抛异常，
      与其让它抛一个用户看不懂的底层错误，不如在这里给出明确原因。
    * **单页失败不影响整篇**：``pypdf`` 对某些畸形页/不支持的过滤器会抛异常。
      把该页文本置空并记 warning，比整篇上传失败更符合工程直觉
      （用户还能拿到 90% 的内容，并且知道少了哪页）。
    * **全篇没有文本 -> 明确的"扫描件"错误**：扫描件的 PDF 页对象里只有图像，
      这不是"解析失败"而是"需要 OCR"，两者给用户的下一步动作完全不同，
      所以消息必须写清楚。
    """
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise DocumentParseError(
            "PDF 文件无法打开或已损坏",
            detail={"reason": type(exc).__name__},
        ) from exc

    if reader.is_encrypted:
        # 空口令的加密 PDF（只加"权限密码"）能自动解密，先试一次再判定失败
        try:
            if reader.decrypt("") == 0:
                raise DocumentParseError("PDF 已加密，无法解析")
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("PDF 已加密，无法解析") from exc

    warnings: list[str] = []
    pages: list[ParsedPage] = []

    for page_index, page in enumerate(reader.pages):
        page_no = page_index + 1  # 页码从 1 开始：用户看到的页码不是下标
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - 单页失败不拖垮整篇
            logger.warning("pdf_page_extract_failed", page_no=page_no, error=str(exc))
            warnings.append(f"第 {page_no} 页文本抽取失败，已跳过该页（{type(exc).__name__}）")
            page_text = ""
        pages.append(ParsedPage(page_no=page_no, text=page_text))

    total_chars = sum(len(page.text.strip()) for page in pages)
    if total_chars < PDF_MIN_EXTRACTED_CHARS:
        raise DocumentParseError(
            "PDF 中没有可提取的文本（可能是扫描件，需要 OCR）",
            detail={"page_count": len(pages), "extracted_chars": total_chars},
        )

    return pages, warnings


def _csv_delimiter(text: str) -> str:
    """猜 CSV 分隔符；采样失败一律退回 ``","``。

    * 采样只取前若干行：``csv.Sniffer`` 是纯 Python 启发式，文件大时它明显拖慢上传
      （上传是同步接口，慢就是用户等）。
    * 候选分隔符里带 ``:``：制度/配置类导出常见 ``姓名:张三`` 这种冒号分隔。
    * 嗅探失败（单列 CSV、只有一行、结构太规整）不报错，退回 ``","`` ——
      "猜不出分隔符"不等于"文件坏了"，单列 CSV 本来就该按整体处理。
    """
    sample_lines = text.split("\n")[:20]
    sample = "\n".join(sample_lines)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|:").delimiter
    except csv.Error:
        return ","


def _first_row_looks_like_header(rows: list[list[str]], full_text: str) -> bool:
    """判断第一行是否是表头。

    优先用 ``csv.Sniffer.has_header``（它比较两行的列"类型一致性"），
    但它在两种真实场景下会给出错误答案，所以必须自己兜底：

    * **中文表头猜不出来**：``姓名;城市`` 与 ``张三;北京`` 在它眼里"都是字符串"，判为无表头；
    * **样本太短直接抛 ``csv.Error``**（只有一行时）。

    兜底规则（两条都满足才算有表头，宁可少一行数据也不要让列名变成数据）：
    1. 第一行多数单元格"不含数字"——数据行一般含数字（金额、数量、日期）；
    2. 第一行与第二行至少有一列取值不同——完全相同说明两行是同类数据。

    代价说明：如果 CSV 数据行恰好全是纯文本（如 ``姓名,城市`` / ``张三,北京``），
    会把第一行数据当表头丢掉。这与"把表头当数据"相比危害小得多（后者会让每个
    数据行都缺少字段名，检索直接失效）。
    """
    try:
        if csv.Sniffer().has_header("\n".join(full_text.split("\n")[:20])):
            return True
    except csv.Error:
        pass

    if len(rows) < 2:
        # 只有一行：按"有表头"处理，最终会抛出"只有表头没有数据行"这种更准确的错误
        return True

    first, second = rows[0], rows[1]
    if not first:
        return True
    non_numeric = sum(1 for cell in first if not any(ch.isdigit() for ch in cell))
    if non_numeric * 2 <= len(first):  # 数字单元格占多数 -> 更像数据行
        return False
    return any(a != b for a, b in zip(first, second, strict=False))


def _parse_csv(text: str) -> tuple[list[ParsedPage], list[str]]:
    """CSV -> **自然语言行**，每行形如 ``姓名: 张三 | 部门: 研发部 | 住宿标准: 600``。

    **为什么必须转成自然语言（真实事故）**：
    CSV 直接当纯文本喂给向量模型时，表头在第一行、值在下面几百行，
    列的对应关系在切分后彻底丢失。用户问"谁的住宿标准是 600"，
    模型看到的上下文里只有一堆 ``张三 研发部 600``，没有任何线索说明
    600 是住宿标准 —— 模型会一本正经地答错，而且看起来很有依据。
    转成"列名: 值"后，每一行都自带完整的字段语义，切分到哪都自洽。

    其他约定：
    * 表头取第一行；数据行比表头长时，多余的列命名为 ``col_{i}``（i 从 0 计），
      避免掉数据；比表头短时缺的字段直接不输出（空值没有信息量）。
    * 首行是否表头由 ``csv.Sniffer.has_header()`` 判断，失败则按"有表头"处理
      —— 中文 CSV 绝大多数都带表头，猜错时"少一行数据"比"把数据当列名"更轻。
    * ``line_count`` 记**数据行数**（不含表头），与"文档有多少条记录"对齐。
    * 空 CSV -> :class:`DocumentEmptyError`（白名单/体积都过了但内容为空，属于用户错误）。
    """
    delimiter = _csv_delimiter(text)
    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        raise DocumentParseError("CSV 格式错误", detail={"reason": str(exc)}) from exc

    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise DocumentEmptyError("CSV 文件为空")

    has_header = _first_row_looks_like_header(rows, text)

    if has_header:
        header = [cell.strip() for cell in rows[0]]
        data_rows = rows[1:]
    else:
        header = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows

    lines: list[str] = []
    for row in data_rows:
        pairs: list[str] = []
        for column_index, value in enumerate(row):
            name = header[column_index] if column_index < len(header) else f"col_{column_index}"
            # 列名为空时退回 col_i，否则会生成 ": 值" 这种无法解释的行
            key = name or f"col_{column_index}"
            pairs.append(f"{key}{CSV_KV_SEP}{value.strip()}")
        if pairs:
            lines.append(CSV_FIELD_SEP.join(pairs))

    warnings: list[str] = []
    if not lines:
        # 只有表头没有数据行：**不抛错**，因为表头本身是有信息的文本
        # （"这份表有哪些字段"往往就是用户想问的），抛 DocumentEmptyError 会
        # 让"上传一份空模板"变成一次莫名其妙的失败。改成保留表头 + 记 warning，
        # 若最终文本为空会在 parse_bytes 第 4 步统一拦下。
        lines.append(CSV_FIELD_SEP.join(cell.strip() for cell in rows[0] if cell.strip()) or "空表")
        warnings.append("CSV 只有表头，没有数据行")

    return [ParsedPage(page_no=None, text="\n".join(lines))], warnings


def _flatten_json(
    node: Any,
    path: str,
    lines: list[str],
    warnings: list[str],
    depth: int,
    seen: set[int],
) -> None:
    """递归展平：``{"a": {"b": [1, 2]}}`` -> ``a.b[0]: 1`` / ``a.b[1]: 2``。

    数组用 ``[i]`` 下标（保留顺序信息，检索时能知道"这是第几个"），
    字典用 ``.`` 连接（路径可读）。
    """
    if depth > JSON_MAX_DEPTH:
        # 截断而不是继续递归：畸形/自引用结构会把内存打满。
        # 只加一条 warning（按第一次触发去重），不要每层都加一条刷屏。
        message = f"JSON 嵌套深度超过 {JSON_MAX_DEPTH} 层，超出的部分已截断"
        if message not in warnings:
            warnings.append(message)
        # **必须落一行占位**：如果整份文档深到"一层都没落下来"，
        # 产物就是空列表，上层会报"JSON 解析后没有任何字段" —— 但那不是事实，
        # 事实是"太深被截断了"。占位行 + warning 才是如实描述。
        lines.append(f"{path or '$'}: <截断：嵌套超过 {JSON_MAX_DEPTH} 层>")
        return

    if isinstance(node, dict):
        if id(node) in seen:
            warnings.append("JSON 中存在循环引用，已跳过该分支")
            return
        seen.add(id(node))
        if not node:
            lines.append(f"{path or '$'}: {{}}")
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            _flatten_json(value, child, lines, warnings, depth + 1, seen)
        seen.discard(id(node))
        return

    if isinstance(node, list):
        if id(node) in seen:
            warnings.append("JSON 中存在循环引用，已跳过该分支")
            return
        seen.add(id(node))
        if not node:
            lines.append(f"{path or '$'}: []")
        for index, value in enumerate(node):
            _flatten_json(value, f"{path}[{index}]", lines, warnings, depth + 1, seen)
        seen.discard(id(node))
        return

    # 叶子：顶层直接是标量时 path 为空，用 "$" 作为根占位符，避免出现 ": 值"
    lines.append(f"{path or '$'}{CSV_KV_SEP}{_json_scalar_to_text(node)}")


def _json_scalar_to_text(value: Any) -> str:
    """标量转字符串；``null`` 显式写出，避免"路径: "这种空值行被误读成缺字段。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_json(text: str) -> tuple[list[ParsedPage], list[str]]:
    """JSON -> ``路径: 值`` 行。

    顶层不是 dict/list 时按单值处理（``path`` 用 ``"$"``）。
    深度超过 :data:`JSON_MAX_DEPTH` 层时截断并加 warning，不抛错 ——
    截断的文档仍然可检索，只是少了最深的那部分。
    """
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentParseError(
            "JSON 格式错误",
            detail={"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
        ) from exc

    lines: list[str] = []
    warnings: list[str] = []
    _flatten_json(data, "", lines, warnings, depth=0, seen=set())

    if not lines:
        raise DocumentEmptyError("JSON 解析后没有任何字段")
    return [ParsedPage(page_no=None, text="\n".join(lines))], warnings


#: 解析器注册表：扩展名 -> 处理纯文本的函数。PDF 单独分派（它吃 bytes）。
_TEXT_PARSERS: Final[dict[str, Callable[[str], tuple[list[ParsedPage], list[str]]]]] = {
    "markdown": _parse_markdown,
    "text": _parse_text,
    "csv": _parse_csv,
    "json": _parse_json,
}


# ======================================================================================
# 统一入口
# ======================================================================================


def _resolve_allowed_extensions(allowed_extensions: Sequence[str] | None) -> list[str]:
    """白名单归一化：去点、小写、去空。

    ``None`` 表示"调用方没指定"，此时取 ``get_settings().allowed_extensions``
    （配置层已经做过同样的归一化，这里再兜一次是因为 API 层可能传进来
    未归一化的请求参数）。
    """
    raw = get_settings().allowed_extensions if allowed_extensions is None else allowed_extensions
    return sorted({ext.strip().lower().lstrip(".") for ext in raw if ext and ext.strip()})


def parse_bytes(
    data: bytes,
    filename: str,
    *,
    allowed_extensions: Sequence[str] | None = None,
) -> ParsedDocument:
    """解析入口：``bytes + 文件名 -> ParsedDocument``。

    **校验顺序（早失败早报错，每一步都比后一步便宜）**：

    1. **扩展名白名单**：纯字符串运算，最便宜，而且能挡掉"根本不该打开的格式"
       （比如用户传了 ``.exe``）。放在最前面可以避免为非法文件做任何解码工作。
    2. **零字节检查**：不做编码嗅探就能判定 —— 空字节序列走探测器会得到
       ``None`` 然后一路撞到"无法识别文件编码"，报错原因完全误导。
       （体积上限由 API 层在读取前用 ``Content-Length`` 拦截，这里不重复做。）
    3. **分派解析**：此时才会真正解码/解析，是最贵的一步。
    4. **解析后归一化 + 空文本检查**：有些文件"有字节但没内容"
       （全空白 txt、只有表头的 CSV、没有文本的 PDF、``[]`` 的 JSON），
       必须在入库前拦住，否则会往库里写一堆空 chunk。
    5. **组装 ``ParsedDocument``** 并填 ``char_count`` / ``line_count``。

    注意第 3 步抛出的 :class:`DocumentEmptyError`（如空 CSV）与第 4 步的
    "解析后为空"是两种语义：前者是"格式语义上就没有内容"，
    后者是"有字节但归一化后什么都没剩"，消息不同便于前端区分提示。
    """
    ext = detect_ext(filename)
    allowed = _resolve_allowed_extensions(allowed_extensions)

    # ---- 1. 白名单 ----
    if ext not in allowed:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型: {ext or '(无扩展名)'}",
            detail={"ext": ext, "allowed": allowed},
        )
    parser = EXT_PARSERS.get(ext)
    if parser is None:  # 白名单里配了但没实现（配置层与实现不同步时早失败）
        raise UnsupportedFileTypeError(
            f"暂不支持解析该类型: {ext}",
            detail={"ext": ext, "allowed": allowed},
        )

    # ---- 2. 零字节 ----
    if not data:
        raise DocumentEmptyError("文件内容为空")

    # ---- 3. 分派 ----
    encoding: str | None
    if parser == "pdf":
        pages, warnings = _parse_pdf(data)
        encoding = None  # PDF 内部字库自行编码，没有"文件编码"这个概念
    else:
        raw_text, encoding, _tried = _decode_bytes(data)
        pages, warnings = _TEXT_PARSERS[parser](raw_text)

    # ---- 4. 归一化 + 空文本检查 ----
    if parser == "pdf":
        # PDF 的每页文本也要过归一化，否则 PDF 抽取出的 \r、零宽字符会进入 BM25 词表
        pages = [ParsedPage(page_no=page.page_no, text=_normalize(page.text)) for page in pages]
        text = _normalize("\n\n".join(page.text for page in pages))
    else:
        # 文本类格式**必须归一化解析器改写过的文本**（CSV 的 "列名: 值"、
        # JSON 的展平行、Markdown 的原文），不能回头去归一化解码后的原始字节文本 ——
        # 第一版写成 `_normalize(raw_text)`，于是 CSV/JSON 的结构化改写整个丢失，
        # 用户拿到的还是原始 CSV 文本。这是验证脚本第 2、3 项抓出来的。
        text = _normalize("\n".join(page.text for page in pages))
        # 页内文本与全文保持一致（文本类格式固定一个逻辑分区，不存在"页"的概念）
        pages = [ParsedPage(page_no=page.page_no, text=text) for page in pages]

    if not text.strip():
        raise DocumentEmptyError("文档解析后没有有效文本")

    parsed = ParsedDocument(
        filename=filename,
        ext=ext,
        parser=parser,
        text=text,
        pages=pages,
        encoding=encoding,
        line_count=_count_lines(text),
        char_count=len(text),
        warnings=warnings,
    )
    logger.debug(
        "document_parsed",
        filename=filename,
        parser=parser,
        encoding=encoding,
        char_count=parsed.char_count,
        page_count=parsed.page_count,
        warnings=len(warnings),
    )
    return parsed


def parse_path(
    path: Path,
    *,
    allowed_extensions: Sequence[str] | None = None,
) -> ParsedDocument:
    """按路径解析。``filename`` 取 ``path.name``（数据库里存的是原始文件名，不是全路径）。"""
    resolved = Path(path)
    data = resolved.read_bytes()
    return parse_bytes(data, resolved.name, allowed_extensions=allowed_extensions)
