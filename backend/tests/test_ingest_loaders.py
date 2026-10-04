"""文档加载层：编码嗅探、CSV/JSON 结构化改写、PDF 逐页抽取、校验顺序。

这里钉的是**入库链路的入口契约**，错一步后面全废：

* **编码嗅探必须按"确定性从高到低"**：BOM（零误判）> 严格 UTF-8 > 统计探测器 > CJK 兜底。
  而且 BOM 必须**看字节前缀**，不能"逐个 codec 试着 decode"（`utf-16` 没有 BOM 时也从不报错，
  随机二进制会被"成功"解成一篇乱码，永远走不到"无法识别编码"）。
* **GB18030 必须排在 GBK 之前**：先试 GBK 会把生僻字解成另一个合法汉字而**不报错** → 静默乱码。
* **CSV 必须改写成"列名: 值"**：否则列头与值的对应关系在切分后丢失，模型会一本正经地答错。
* **PDF 必须逐页保留页码**：丢了页码，引用就只能定位到"某文档"，而且"少一页"远好于"整篇失败"。
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from knowflow.core.exceptions import (
    DocumentEmptyError,
    DocumentParseError,
    UnsupportedFileTypeError,
)
from knowflow.ingest import loaders
from knowflow.ingest.loaders import detect_ext, parse_bytes, parse_path

TEXT = "一线城市住宿标准为每晚六百元"


def make_pdf(pages: list[str]) -> bytes:
    """用 ``pypdf.PdfWriter`` 造一个多页 PDF（**不依赖 reportlab**）。

    标准 14 号字体（Helvetica）不需要嵌入字库，所以可以手工拼一个内容流；
    文本用 ASCII，保证 ``extract_text()`` 拿得到（中文需要 CID 字库，pypdf 造不出来）。
    """
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=400, height=300)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(content)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


PAGE_TEXTS = [
    "Page one talks about city hotel reimbursement AAA",
    "Page two talks about meal allowance BBB",
    "Page three talks about annual leave CCC",
]


# ======================================================================================
# 扩展名
# ======================================================================================
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.md", "md"),
        ("A.MD", "md"),
        ("report.tar.gz", "gz"),
        (".gitignore", ""),
        ("无扩展名", ""),
        ("dir/sub/b.pdf", "pdf"),
    ],
)
def test_扩展名从文件名末尾取且小写(filename: str, expected: str) -> None:
    """用 ``Path(name).suffix`` 而不是 ``rsplit(".")``：后者会把 ``.gitignore`` 当成扩展名。"""
    assert detect_ext(filename) == expected


# ======================================================================================
# 编码嗅探
# ======================================================================================
@pytest.mark.parametrize(
    ("label", "raw", "expected_encoding"),
    [
        ("UTF-8", TEXT.encode("utf-8"), "utf-8"),
        ("UTF-8-BOM", TEXT.encode("utf-8-sig"), "utf-8-sig"),
        ("UTF-16-LE-BOM", b"\xff\xfe" + TEXT.encode("utf-16-le"), "utf-16-le"),
        ("UTF-16-BE-BOM", b"\xfe\xff" + TEXT.encode("utf-16-be"), "utf-16-be"),
        ("UTF-32-LE-BOM", b"\xff\xfe\x00\x00" + TEXT.encode("utf-32-le"), "utf-32-le"),
        ("UTF-32-BE-BOM", b"\x00\x00\xfe\xff" + TEXT.encode("utf-32-be"), "utf-32-be"),
    ],
)
def test_各种编码都能被识别且内容无损(label: str, raw: bytes, expected_encoding: str) -> None:
    """BOM 必须按**字节前缀**判定：`utf-32-le` 的 `FF FE 00 00` 必须排在 `utf-16-le` 之前，
    否则 UTF-32 的文件会被当成 UTF-16 解出夹着 NUL 的乱码。"""
    parsed = parse_bytes(raw, "a.txt")
    assert parsed.encoding == expected_encoding
    assert parsed.text == TEXT, "BOM 字符不能留在正文里（会污染第一个 chunk）"


def test_GB18030排在GBK之前所以生僻字不会被解成另一个汉字() -> None:
    """GB18030 是 GBK 的超集；先试 GBK 时生僻字会**解码报错或解成别的字**，静默乱码。"""
    source = "员工𠮷野家的住宿标准为每晚六百元"
    raw = source.encode("gb18030")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("gbk")  # 前提：GBK 真的解不了这段字节

    parsed = parse_bytes(raw, "a.txt")
    assert parsed.encoding == "gb18030"
    assert "𠮷" in parsed.text
    assert parsed.text == source


def test_无法识别的编码抛出带排查信息的解析错误() -> None:
    """`tried` 列表要能直接回答"到底试过什么"，否则线上只能靠猜。"""
    with pytest.raises(DocumentParseError) as excinfo:
        parse_bytes(b"\x00\x01\x02\x03\xfe\xff\x00", "a.txt")
    assert excinfo.value.code == "DOCUMENT_PARSE_ERROR"
    assert excinfo.value.detail["tried"] == ["utf-8"]
    assert excinfo.value.detail["size_bytes"] == 7


def test_探测器不可用时全部兜底编码失败就报错(monkeypatch: pytest.MonkeyPatch) -> None:
    """把探测器打桩成 None，确定性地走「CJK 兜底顺序全失败」这条分支。"""
    monkeypatch.setattr(loaders, "_detect_encoding", lambda raw: None)
    with pytest.raises(DocumentParseError):
        parse_bytes(b"\xff\xff\xff", "a.txt")


# ======================================================================================
# 校验顺序
# ======================================================================================
def test_扩展名白名单先于解码被拦下() -> None:
    """非法文件不做任何解码工作（也避免拿 .exe 的二进制去喂编码探测器）。"""
    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        parse_bytes(b"\x00\x01\x02\x03\xfe\xff", "evil.exe")
    assert excinfo.value.detail["ext"] == "exe"
    assert "md" in excinfo.value.detail["allowed"]


def test_白名单之外的扩展名即使内容合法也被拦下() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        parse_bytes(TEXT.encode("utf-8"), "a.docx")


def test_不传白名单时用配置里的默认值() -> None:
    assert parse_bytes(TEXT.encode("utf-8"), "a.md").parser == "markdown"


def test_零字节文件报空而不是报编码错误() -> None:
    """空字节走探测器会得到 None，然后一路撞到"无法识别编码"，报错原因完全误导。"""
    with pytest.raises(DocumentEmptyError) as excinfo:
        parse_bytes(b"", "a.txt")
    assert excinfo.value.code == "DOCUMENT_EMPTY"


@pytest.mark.parametrize(("filename", "content"), [("a.txt", "   \n\n  "), ("a.md", "\n\n")])
def test_解析后没有有效文本时报文档为空(filename: str, content: str) -> None:
    with pytest.raises(DocumentEmptyError):
        parse_bytes(content.encode("utf-8"), filename)


def test_parse_path从路径取文件名(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "员工报销制度.md"
    path.write_bytes("# 标题\n\n正文内容。".encode())
    parsed = parse_path(path)
    assert parsed.filename == "员工报销制度.md"
    assert parsed.parser == "markdown"


# ======================================================================================
# Markdown / 文本
# ======================================================================================
def test_没有标题的markdown给出退化警告() -> None:
    """没有 `#` 标题时切分会退化成纯递归，section_path 全为 None —— 不是错误但必须说清。"""
    parsed = parse_bytes("没有任何标题的一段话。".encode(), "a.md")
    assert parsed.warnings
    assert "标题" in parsed.warnings[0]


def test_有标题的markdown不产生警告() -> None:
    # 注意：不能用 b"...非 ASCII..."，Python 的 bytes 字面量只允许 ASCII 字符。
    # 这份测试文件里其它地方都正确写成了 "...".encode("utf-8")，这里保持一致。
    parsed = parse_bytes("# 标题\n\n正文内容。".encode(), "a.md")
    assert parsed.warnings == []


def test_归一化会统一换行并去掉零宽字符与大片空行() -> None:
    parsed = parse_bytes("第一行\r\n\r\n\r\n\r\n第二行\u200b结束".encode("utf-8"), "a.txt")
    assert "\r" not in parsed.text
    assert "\u200b" not in parsed.text
    assert "\n\n\n" not in parsed.text


def test_行数按换行符统计() -> None:
    parsed = parse_bytes("第一行\n第二行\n第三行".encode(), "a.txt")
    assert parsed.line_count == 3
    assert parsed.char_count == len(parsed.text)


# ======================================================================================
# CSV
# ======================================================================================
def test_CSV被改写成列名冒号值的自然语言行() -> None:
    """不改写的话，切分之后"600"属于哪一列就丢了，模型会一本正经地答错。"""
    raw = "姓名,部门,住宿标准\n张三,研发部,600\n李四,市场部,500\n".encode()
    parsed = parse_bytes(raw, "a.csv")
    assert parsed.parser == "csv"
    assert parsed.text == (
        "姓名: 张三 | 部门: 研发部 | 住宿标准: 600\n姓名: 李四 | 部门: 市场部 | 住宿标准: 500"
    )


def test_CSV的行数只数数据行不含表头() -> None:
    parsed = parse_bytes("姓名,部门\n张三,研发部\n李四,市场部\n".encode(), "a.csv")
    assert parsed.line_count == 2


def test_只有表头的CSV保留表头并记警告() -> None:
    """ "这份表有哪些字段"本身往往就是用户想问的，抛错等于让上传空模板莫名失败。"""
    parsed = parse_bytes("姓名,部门\n".encode(), "a.csv")
    assert parsed.text == "姓名 | 部门"
    assert parsed.warnings == ["CSV 只有表头，没有数据行"]


def test_CSV数据行比表头长时多余的列命名为col_i() -> None:
    parsed = parse_bytes("姓名,部门\n张三,研发部,备注\n".encode(), "a.csv")
    assert "col_2: 备注" in parsed.text


def test_空CSV报文档为空() -> None:
    with pytest.raises(DocumentEmptyError):
        parse_bytes(b"\n\n", "a.csv")


def test_分号分隔的CSV也能识别() -> None:
    parsed = parse_bytes("姓名;城市\n张三;北京\n".encode(), "a.csv")
    assert parsed.text == "姓名: 张三 | 城市: 北京"


# ======================================================================================
# JSON
# ======================================================================================
def test_JSON被展平成路径冒号值() -> None:
    parsed = parse_bytes(b'{"a": {"b": [1, 2]}}', "a.json")
    assert parsed.parser == "json"
    assert parsed.text.splitlines() == ["a.b[0]: 1", "a.b[1]: 2"]


def test_JSON的null与布尔值被显式写出() -> None:
    """`路径: ` 这种空值行会被误读成缺字段，所以 null/true/false 都要有字面量。"""
    parsed = parse_bytes(b'{"c": true, "d": null, "e": false}', "a.json")
    assert parsed.text.splitlines() == ["c: true", "d: null", "e: false"]


def test_JSON顶层数组用下标路径() -> None:
    parsed = parse_bytes(b'[{"x": 1}, {"x": 2}]', "a.json")
    assert parsed.text.splitlines() == ["[0].x: 1", "[1].x: 2"]


@pytest.mark.parametrize("raw", [b"{}", b"[]"])
def test_空的JSON容器不会变成空文档(raw: bytes) -> None:
    parsed = parse_bytes(raw, "a.json")
    assert parsed.text in {"$: {}", "$: []"}


def test_JSON格式错误带上行列号() -> None:
    with pytest.raises(DocumentParseError) as excinfo:
        parse_bytes(b"{not json", "a.json")
    assert excinfo.value.detail["line"] == 1
    assert "reason" in excinfo.value.detail


def test_JSON嵌套过深会截断并记警告() -> None:
    """畸形/自引用结构不能把内存打满，但也要留下占位行说明"是太深被截断了"。"""
    raw = ('{"a":' * 30 + "1" + "}" * 30).encode("utf-8")
    parsed = parse_bytes(raw, "a.json")
    assert any("截断" in warning for warning in parsed.warnings)
    assert "截断" in parsed.text


# ======================================================================================
# PDF
# ======================================================================================
@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(PAGE_TEXTS)


def test_PDF逐页抽取并保留页码(pdf_bytes: bytes) -> None:
    """页码是这一层最有价值的产出：没有它，引用只能定位到"某文档"，用户还得自己翻。"""
    parsed = parse_bytes(pdf_bytes, "a.pdf")
    assert parsed.parser == "pdf"
    assert parsed.page_count == 3
    assert [page.page_no for page in parsed.pages] == [1, 2, 3]
    assert parsed.encoding is None, "PDF 内部字库自行编码，没有「文件编码」这个概念"


def test_PDF第二页的内容属于第二页(pdf_bytes: bytes) -> None:
    parsed = parse_bytes(pdf_bytes, "a.pdf")
    second = parsed.pages[1]
    assert second.page_no == 2
    assert "meal allowance" in second.text
    assert "city hotel" not in second.text


def test_PDF每页文本都会过归一化(pdf_bytes: bytes) -> None:
    parsed = parse_bytes(pdf_bytes, "a.pdf")
    assert "\r" not in parsed.text
    assert parsed.text.count("\n\n") >= 2


def test_没有文本的PDF报扫描件而不是解析失败() -> None:
    """ "需要 OCR"和"文件坏了"给用户的下一步动作完全不同，消息必须区分。"""
    blank = make_pdf(["", "", ""])
    with pytest.raises(DocumentParseError) as excinfo:
        parse_bytes(blank, "blank.pdf")
    assert "扫描件" in excinfo.value.message
    assert excinfo.value.detail["extracted_chars"] == 0


def test_损坏的PDF报解析失败() -> None:
    with pytest.raises(DocumentParseError):
        parse_bytes(b"%PDF-1.4 this is not really a pdf", "bad.pdf")


def test_单页抽取失败不影响整篇(monkeypatch: pytest.MonkeyPatch, pdf_bytes: bytes) -> None:
    """ "少一页"远好于"整篇上传失败"，但少了哪一页必须记在 warnings 里。"""

    class BrokenPage:
        def extract_text(self) -> str:
            raise RuntimeError("unsupported filter")

    original = loaders.PdfReader

    class ReaderProxy:
        def __init__(self, stream: object) -> None:
            self._reader = original(stream)

        @property
        def is_encrypted(self) -> bool:
            return bool(self._reader.is_encrypted)

        @property
        def pages(self) -> list[object]:
            pages = list(self._reader.pages)
            pages[1] = BrokenPage()
            return pages

    monkeypatch.setattr(loaders, "PdfReader", ReaderProxy)
    parsed = parse_bytes(pdf_bytes, "a.pdf")
    assert parsed.page_count == 3
    assert any("第 2 页" in warning for warning in parsed.warnings)
