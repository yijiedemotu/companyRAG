"""文档质量门：检查三份大文档与教程的结构、自包含性与"数字可复核"。

**为什么文档也需要门禁**：文档的错误不会被测试发现，但会**直接误导读者**。
最危险的三类：

1. **编造的数字**。文档里的"recall@5 从 0.71 提到 0.85"如果没人复核，
   读者（和面试官）无法分辨它是不是真的跑出来的。本项目的纪律是：
   **所有数字必须能在 `docs/03-实测证据.md` 里找到**。
2. **失效的交叉引用**。手册指向"第 25 章"、HTML 目录指向一个不存在的锚点 ——
   读者点不过去就只能放弃。
3. **不再自包含的 HTML**。哪天有人顺手加了个 CDN 引用，离线打开就白屏。

做四件事：
  (a) 教程每个阶段文件必须含 8 个标准小节（结构统一，读者才知道去哪儿找）；
  (b) HTML 必须自包含（无外部 script/link/img）+ id 无重复 + 内部锚点无失效；
  (c) 手册 / HTML / README 里出现的关键数字必须能在 `docs/03-实测证据.md` 找到；
  (d) 文档里引用的仓库路径必须真实存在。

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\check_docs.py
退出码：0 = 全部通过；1 = 有问题
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
EVIDENCE = DOCS / "03-实测证据.md"

# 教程每个阶段文件必须有的小节（按标题关键字匹配，容忍编号差异）
TUTORIAL_SECTIONS = [
    "为什么先做",
    "要解决的问题",
    "动手实现",
    "怎么验证",
    "常见错误",
    "自测问题",
    "本阶段小结",
    "延伸阅读",
]

# 必须能追溯到 docs/03 的关键数字（正则）。每条都是简历/文档里真正在用的结论。
TRACEABLE = [
    (r"222", "pytest 用例数"),
    (r"PASS = 41", "链路冒烟断言数"),
    (r"PASS = 70", "HTTP 冒烟断言数"),
    (r"34 / 34 / 34", "三方一致性"),
    (r"0\.6755", "正样本相似度均值"),
    (r"0\.5704", "负样本相似度均值"),
    (r"1\.79", "两后端 score 差"),
    (r"0\.849", "hybrid_rerank recall@5"),
    (r"0\.721", "bm25 recall@5"),
    (r"p\s*=\s*0\.010", "消融显著性 p 值"),
]

problems: list[str] = []


def read(path: Path) -> str:
    return open(path, encoding="utf-8", errors="ignore").read()


def check_tutorial() -> None:
    folder = DOCS / "教程"
    print("--- (a) 教程结构 ---")
    if not folder.exists():
        problems.append("docs/教程/ 不存在")
        print("  ✗ docs/教程/ 不存在")
        return
    files = sorted(folder.glob("*.md"))
    print(f"  共 {len(files)} 个文件")
    for path in files:
        text = read(path)
        heads = re.findall(r"^#{1,3}\s*(.+)$", text, re.M)
        missing = [s for s in TUTORIAL_SECTIONS if not any(s in h for h in heads)]
        lines = text.count("\n") + 1
        if path.name == "README.md":
            print(f"  [--] {path.name:34s} {lines:5d} 行（总览，不套模板）")
            continue
        if "附录" in path.name:
            # 附录结构本来就与"阶段"不同（按错误信息索引 / 按问题分组），不套七节模板
            print(f"  [--] {path.name:34s} {lines:5d} 行（附录，不套模板）")
            continue
        if missing:
            problems.append(f"{path.name} 缺少小节: {missing}")
            print(f"  [✗] {path.name:34s} {lines:5d} 行  缺少: {missing}")
        else:
            print(f"  [✓] {path.name:34s} {lines:5d} 行")


def check_html() -> None:
    print("\n--- (b) HTML 自包含性与锚点 ---")
    path = REPO / "03-knowflow-项目实现详解.html"
    if not path.exists():
        problems.append("HTML 文档不存在")
        print("  ✗ 文件不存在")
        return
    try:
        from lxml import html as lxml_html
    except ImportError:
        print("  (跳过：未安装 lxml)")
        return

    doc = lxml_html.parse(str(path)).getroot()
    size = path.stat().st_size

    external = [
        e.get("src") or e.get("href")
        for e in doc.xpath("//script[@src] | //link[@href] | //img[@src]")
    ]
    external = [u for u in external if u and not u.startswith("#")]
    ids = doc.xpath("//*[@id]/@id")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    internal = sorted(
        a[1:] for a in set(doc.xpath("//a/@href")) if a.startswith("#") and len(a) > 1
    )
    broken = [a for a in internal if a not in set(ids)]
    charset = bool(doc.xpath("//meta[@charset]"))
    viewport = bool(doc.xpath("//meta[contains(@name,'viewport')]"))

    print(f"  字节数        : {size:,}")
    print(
        f"  h1/h2/h3      : {len(doc.xpath('//h1'))} / {len(doc.xpath('//h2'))} / {len(doc.xpath('//h3'))}"
    )
    print(
        f"  svg/表格/代码 : {len(doc.xpath('//svg'))} / {len(doc.xpath('//table'))} / {len(doc.xpath('//pre'))}"
    )
    print(f"  charset/viewport: {charset} / {viewport}")
    print(f"  外部资源引用  : {external or '无（自包含 ✓）'}")
    print(f"  重复 id       : {dup or '无 ✓'}")
    print(f"  内部锚点      : {len(internal)} 个，失效 {broken or '无 ✓'}")

    if external:
        problems.append(f"HTML 引用了外部资源: {external[:3]}")
    if dup:
        problems.append(f"HTML 存在重复 id: {dup[:3]}")
    if broken:
        problems.append(f"HTML 存在失效锚点: {broken[:3]}")
    if not (charset and viewport):
        problems.append("HTML 缺少 charset 或 viewport")


def check_numbers() -> None:
    print("\n--- (c) 数字可复核性 ---")
    if not EVIDENCE.exists():
        problems.append("docs/03-实测证据.md 不存在")
        print("  ✗ 证据文件不存在")
        return
    evidence = read(EVIDENCE)
    targets = [
        REPO / "知识库学习手册.md",
        REPO / "03-knowflow-项目实现详解.html",
        REPO / "README.md",
        REPO / "简历-大模型应用开发实习生-KnowFlow.md",
    ]
    for target in targets:
        if not target.exists():
            continue
        text = read(target)
        unverified: list[str] = []
        used = 0
        for pattern, label in TRACEABLE:
            if re.search(pattern, text):
                used += 1
                if not re.search(pattern, evidence):
                    unverified.append(f"{label}({pattern})")
        status = "✓" if not unverified else "✗"
        print(
            f"  [{status}] {target.name:36s} 命中 {used}/{len(TRACEABLE)} 项，"
            f"无法追溯: {unverified or '无'}"
        )
        if unverified:
            problems.append(f"{target.name} 里有数字无法在 docs/03 追溯: {unverified}")


#: 运行时才生成、或由使用者自己创建的路径（不该算"文档引用了不存在的文件"）
_RUNTIME_PATHS = {
    "backend/.env",
    "data/_smoke",
    "data/_smoke_hash",
    "data/chroma",
    "data/uploads",
}

#: 出现这些词的行，说明作者**正是在说"这个文件不存在"**，属于正确用法而非错误引用
_MISSING_MARKERS = (
    "不存在",
    "已被删除",
    "未实测",
    "新建",
    "建议",
    "改造",
    "运行时",
    "临时",
)


def _is_allowed_missing(text: str, ref: str) -> bool:
    """判断一个"不存在的路径"是否属于合理引用。

    三道放行，都是为了**避免门禁被假警淹没**（假警会让门禁失去意义）：

    1. 运行时才生成的路径（`data/_smoke`、用户自己复制的 `backend/.env`）；
    2. 无扩展名的简写（`docs/00` 在正文里是"那一章"的代称，不是文件路径）；
    3. 引用所在行带"不存在 / 已被删除 / 未实测 / 新建"等语义标记 ——
       作者**本来就在说它不存在**（例如附录里的"已知缺口"表与"改造练习"）。

    真正要抓的是第 4 类：作者**当成存在**来引用的路径（那才是会让读者找不到东西的错误）。
    """
    if ref in _RUNTIME_PATHS:
        return True
    last = ref.rsplit("/", 1)[-1]
    if "." not in last:
        return True
    for line in text.splitlines():
        if ref in line and any(marker in line for marker in _MISSING_MARKERS):
            return True
    return False


def check_markdown_anchors() -> None:
    """markdown 目录里的锚点必须指向真实标题。

    这一条**是补出来的**：改标题文字时锚点会悄悄失效（编辑器里看不出来，
    只有点的时候才发现跳不过去）。本项目真实发生过 —— 把标题里的
    "204 个用例、4.32 秒"改成"220 个用例、4.22 秒"后，目录的**链接文字**同步变了，
    但**锚点仍是旧的**。9,500 行的手册有 37 个目录链接，靠人眼看不可能不漏。
    """
    print("\n--- (e) markdown 目录锚点 ---")
    script = Path(__file__).with_name("fix_doc_anchors.py")
    if not script.exists():
        print("  (跳过：fix_doc_anchors.py 不存在)")
        return
    import subprocess

    result = subprocess.run(  # noqa: S603 - 固定调用本仓库脚本，无外部输入
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    out = result.stdout or ""
    for line in out.strip().splitlines():
        if line.strip().startswith(("---", "标题", "RESULT", "  ✗", "      现在", "      应为")):
            print(f"  {line.strip()}")
    if result.returncode != 0:
        problems.append("markdown 目录里有失效锚点（跑 fix_doc_anchors.py --fix 修复）")


def check_paths() -> None:
    print("\n--- (d) 文档引用的仓库路径是否存在 ---")
    targets = [
        REPO / "知识库学习手册.md",
        REPO / "README.md",
        REPO / "简历-大模型应用开发实习生-KnowFlow.md",
    ]
    pattern = re.compile(r"`((?:backend|frontend|docs|data)/[A-Za-z0-9_\-./]+)`")
    for target in targets:
        if not target.exists():
            continue
        text = read(target)
        refs = set(pattern.findall(text))
        missing = []
        for ref in refs:
            clean = ref.rstrip(".,;:)")
            if (REPO / clean).exists():
                continue
            # 允许"目录/前"这种泛指，以及形如 `data/samples/*.md` 的通配
            if "*" in clean or clean.endswith("/"):
                continue
            # 允许这四类"本来就不该存在"的引用（否则门禁会被假警淹没）：
            #   1) 运行时才生成的（data/_smoke、backend/.env 由用户复制而来）
            #   2) 文档明确标注为"不存在 / 已被删除 / 未实测"的历史引用
            #   3) 改造练习里"建议新建"的文件
            #   4) 无扩展名的简写（`docs/00` 这种在正文里指代文档的写法）
            if _is_allowed_missing(text, clean):
                continue
            missing.append(clean)
        status = "✓" if not missing else "✗"
        print(
            f"  [{status}] {target.name:36s} 引用 {len(refs)} 个路径，不存在: "
            f"{sorted(missing)[:4] or '无'}"
        )
        if missing:
            problems.append(f"{target.name} 引用了不存在的路径: {sorted(missing)[:5]}")


def main() -> int:
    print("=" * 78)
    print("文档质量门")
    print("=" * 78)
    check_tutorial()
    check_html()
    check_numbers()
    check_paths()
    check_markdown_anchors()

    print("\n" + "=" * 78)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for item in problems:
            print(f"  - {item}")
    print("RESULT:", "OK" if not problems else f"{len(problems)} 个问题")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
