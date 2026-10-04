from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowflow.retrieval.text import coverage, query_terms, tokenize_zh  # noqa: E402


def show(label, value):
    print(f"{label}: {value!r}")


show("tok 一线城市住宿标准", tokenize_zh("一线城市住宿标准"))
show("tok 报销，标准", tokenize_zh("报销，标准"))
show("tok 错误码 PAYLOAD_TOO_LARGE", tokenize_zh("错误码 PAYLOAD_TOO_LARGE"))
show("tok 空", tokenize_zh(""))
show("tok 纯标点 !!!", tokenize_zh("!!!"))
show("tok 标点混合 ，。、", tokenize_zh("，。、"))
show("query_terms 一线城市住宿标准", query_terms("一线城市住宿标准"))
show("query_terms !!!", query_terms("!!!"))
show("query_terms 的了吗", query_terms("的了吗"))
show("cov 空query", coverage("", "任意文本"))
show("cov 纯标点query", coverage("!!!", "任意文本"))
show("cov 纯标点text", coverage("住宿", "!!!"))
show("cov 全命中", coverage("住宿标准", "一线城市住宿标准为600元"))
show("cov 部分", coverage("住宿标准600", "一线城市住宿标准为每晚600元"))
show("cov query在text", coverage("住宿标准", "住宿标准"))

# ---- encoding ----
from knowflow.ingest.loaders import _decode_bytes, detect_ext, parse_bytes  # noqa: E402

cases = {
    "utf8": "一线城市住宿标准".encode("utf-8"),
    "utf8-bom": "一线城市住宿标准".encode("utf-8-sig"),
    "utf16-le-bom": "一线城市住宿标准".encode("utf-16"),
    "utf16-be-bom": "一线城市住宿标准".encode("utf-16-be"),
    "gb18030-rare": "𠮷野家".encode("gb18030"),
    "gb18030-normal": "一线城市住宿标准".encode("gb18030"),
    "gbk-normal": "一线城市住宿标准".encode("gbk"),
}
for name, raw in cases.items():
    try:
        text, enc, tried = _decode_bytes(raw)
        print(f"decode {name}: enc={enc!r} tried={tried!r} text={text[:20]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"decode {name}: {type(exc).__name__}: {exc}")

for name, raw in {
    "ff-ff-ff": b"\xff\xff\xff",
    "random": bytes(range(0x80, 0xA0)),
    "utf16-forced": b"\x00\x01\x02\x03\xfe\xff\x00",
}.items():
    try:
        text, enc, tried = _decode_bytes(raw)
        print(f"invalid? {name}: enc={enc!r} tried={tried!r} text={text[:20]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"invalid? {name}: {type(exc).__name__}: {exc}")

show("detect_ext a.tar.gz", detect_ext("a.tar.gz"))
show("detect_ext .gitignore", detect_ext(".gitignore"))
show("detect_ext A.MD", detect_ext("A.MD"))

# CSV / JSON
csv_doc = parse_bytes("姓名,部门,住宿标准\n张三,研发部,600\n李四,市场部,500\n".encode(), "a.csv")
show("csv parser", csv_doc.parser)
show("csv text", csv_doc.text)
show("csv line_count", csv_doc.line_count)
show("csv warnings", csv_doc.warnings)

json_raw = '{"a": {"b": [1, 2]}, "c": true, "d": null, "e": []}'
json_doc = parse_bytes(json_raw.encode(), "a.json")
show("json text", json_doc.text)

# PDF
from pypdf import PdfWriter  # noqa: E402
from pypdf.generic import DictionaryObject, NameObject  # noqa: E402


def make_pdf(pages_text):
    writer = PdfWriter()
    for i, text in enumerate(pages_text):
        page = writer.add_blank_page(width=300, height=200)
        stream = writer._add_object(
            DictionaryObject()  # placeholder
        )
        from pypdf.generic import DecodedStreamObject

        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode("latin-1"))
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
    import io

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


pdf_bytes = make_pdf(
    [
        "Page one has enough extractable text AAA",
        "Page two has enough extractable text BBB",
        "Page three has enough extractable text CCC",
    ]
)
from pypdf import PdfReader  # noqa: E402

r = PdfReader(__import__("io").BytesIO(pdf_bytes))
print("pdf raw pages:", len(r.pages), [p.extract_text() for p in r.pages])
pdf_doc = parse_bytes(pdf_bytes, "a.pdf")
show("pdf page_count", pdf_doc.page_count)
show("pdf page_no list", [p.page_no for p in pdf_doc.pages])
show("pdf text", pdf_doc.text[:120])
show("pdf encoding", pdf_doc.encoding)

blank = make_pdf(["", "", ""])
try:
    parse_bytes(blank, "blank.pdf")
except Exception as exc:  # noqa: BLE001
    print("blank pdf:", type(exc).__name__, exc, getattr(exc, "detail", None))

# subprocess pipes
import subprocess  # noqa: E402

try:
    out = subprocess.run(
        [sys.executable, "-c", "print('hello-subprocess')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("subprocess:", out.returncode, out.stdout.strip(), out.stderr[:200])
except Exception as exc:  # noqa: BLE001
    print("subprocess failed:", type(exc).__name__, exc)
