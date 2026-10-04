from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowflow.ingest.loaders import parse_bytes  # noqa: E402

TEXT = "一线城市住宿标准为每晚六百元"

cases = {
    "utf8": TEXT.encode("utf-8"),
    "utf8-sig": TEXT.encode("utf-8-sig"),
    "utf16-le-bom": b"\xff\xfe" + TEXT.encode("utf-16-le"),
    "utf16-be-bom": b"\xfe\xff" + TEXT.encode("utf-16-be"),
    "utf32-le-bom": b"\xff\xfe\x00\x00" + TEXT.encode("utf-32-le"),
    "utf32-be-bom": b"\x00\x00\xfe\xff" + TEXT.encode("utf-32-be"),
}
for name, raw in cases.items():
    doc = parse_bytes(raw, "a.txt")
    print(f"{name}: encoding={doc.encoding!r} text={doc.text!r} ok={doc.text == TEXT}")

for label, text_value in {
    "rare-3": "𠮷野家",
    "rare-long": "𠮷野家的住宿标准是每晚六百元",
    "rare-with-latin": "员工𠮷野家的住宿标准 600 元",
}.items():
    raw = text_value.encode("gb18030")
    doc = parse_bytes(raw, "a.txt")
    print(f"gb18030 {label}: encoding={doc.encoding!r} text={doc.text!r} equal={doc.text == text_value}")
    try:
        raw.decode("gbk")
        print("   gbk decode: OK")
    except UnicodeDecodeError as exc:
        print("   gbk decode: UnicodeDecodeError", str(exc)[:60])

for name, raw in {
    "ctrl": b"\x00\x01\x02\x03\xfe\xff\x00",
    "ff": b"\xff\xff\xff",
    "high": bytes(range(0x80, 0xA0)),
}.items():
    try:
        doc = parse_bytes(raw, "a.txt")
        print(f"invalid {name}: NO ERROR encoding={doc.encoding!r} text={doc.text[:20]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"invalid {name}: {type(exc).__name__} {exc} detail={getattr(exc, 'detail', None)}")

# 只有空白 / 空 CSV / 坏 JSON
for name, raw, fname in [
    ("blank-txt", b"   \n\n  ", "a.txt"),
    ("empty-csv", b"\n\n", "a.csv"),
    ("header-only-csv", "姓名,部门\n".encode(), "a.csv"),
    ("bad-json", b"{not json", "a.json"),
    ("empty-json-obj", b"{}", "a.json"),
    ("empty-json-arr", b"[]", "a.json"),
    ("no-heading-md", "没有任何标题的一段话。".encode(), "a.md"),
]:
    try:
        doc = parse_bytes(raw, fname)
        print(f"{name}: ok parser={doc.parser} text={doc.text!r} warnings={doc.warnings}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: {type(exc).__name__} code={getattr(exc, 'code', None)} {exc} detail={getattr(exc, 'detail', None)}")
