"""检查并（可选）修复 markdown 目录里的锚点是否指向真实标题。

**为什么需要**：手册有 9,500 行、几十个目录条目。一旦改了某个标题的文字，
**锚点就悄悄失效了** —— 在编辑器里看不出来，只有点的时候才发现跳不过去。
本项目就发生过：把标题里的"204 个用例、4.32 秒"改成"220 个用例、4.22 秒"之后，
目录的链接文字同步变了，但**锚点仍是旧的** `#27-测试204-个用例432-秒...`。

用 GitHub 的 slug 规则生成期望锚点再比对：
  - 转小写；去掉除 `-` `_` 与中日韩文字、字母、数字以外的字符；空格转 `-`。

跑法：
    .venv\\Scripts\\python.exe backend\\scripts\\fix_doc_anchors.py            # 只检查
    .venv\\Scripts\\python.exe backend\\scripts\\fix_doc_anchors.py --fix      # 自动改锚点
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = ["知识库学习手册.md", "README.md"]

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
_TOC_LINK = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
# GitHub 锚点里保留的字符：字母、数字、`-`、`_`、以及 CJK
_KEEP = re.compile(r"[^\w\-\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af ]", re.UNICODE)


def slugify(text: str) -> str:
    """按 GitHub 的规则把标题转成锚点。

    细节：GitHub 会**丢掉**大多数标点（中英文标点都丢），空格变 `-`，
    并且在 `#` 里会去掉行首的序号点号（`27.` -> `27`）。
    """
    text = text.strip()
    # 去掉 markdown 行内标记（`code`、**bold**、[link](x)）
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = _KEEP.sub("", text)
    return text.replace(" ", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="自动把失效锚点改成正确值")
    args = parser.parse_args()

    total_broken = 0
    for rel in DOCS:
        path = REPO / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        anchors = {slugify(m.group(2)) for m in _HEADING.finditer(text)}

        broken: list[tuple[str, str, str]] = []
        for match in _TOC_LINK.finditer(text):
            label, anchor = match.group(1), match.group(2)
            if anchor in anchors:
                continue
            # 用链接文字推导期望锚点
            expected = slugify(label)
            if expected in anchors:
                broken.append((label, anchor, expected))

        print(f"--- {rel} ---")
        print(
            f"  标题 {len(anchors)} 个，目录链接 {len(_TOC_LINK.findall(text))} 个，"
            f"失效 {len(broken)} 个"
        )
        for label, anchor, expected in broken[:12]:
            print(f"  ✗ [{label[:44]}...]")
            print(f"      现在 -> #{anchor}")
            print(f"      应为 -> #{expected}")
        total_broken += len(broken)

        if args.fix and broken:
            for _label, anchor, expected in broken:
                text = text.replace(f"](#{anchor})", f"](#{expected})")
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"  ✓ 已修复 {len(broken)} 个锚点")

    print(f"\nRESULT: {'OK' if not total_broken else f'{total_broken} 个失效锚点'}")
    return 0 if not total_broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
