from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowflow.core.config import Settings  # noqa: E402

CFG = Settings(
    env="test",
    debug=True,
    database_url="sqlite+pysqlite:///:memory:",
    data_dir=Path("data") / "_probe",
    embedding_provider="hash",
    vector_backend="memory",
    openai_api_key="",
    agent_checkpoint_backend="memory",
    log_level="WARNING",
)

from knowflow.ingest.chunkers import DEFAULT_SEPARATORS, _validate_separators, split_document  # noqa: E402
from knowflow.ingest.loaders import parse_bytes  # noqa: E402

md = """# 员工报销制度

## 差旅报销标准

一线城市住宿标准为每晚 600 元，二线城市为 400 元。报销需提供发票。

## 餐饮报销标准

餐饮补贴为每天 100 元。需要提供发票与审批单。

# 年假制度

员工入职满一年后有 5 天年假。
"""
doc = parse_bytes(md.encode(), "a.md")
chunks = split_document(doc, chunk_size=80, chunk_overlap=20, min_chars=20, parent_chunk_size=200)
print("md chunks:", len(chunks))
for c in chunks:
    print("  ", c.index, c.page_no, repr(c.section_path), len(c.content), "in parent:", c.content in c.parent_content)
print("parents distinct:", len({c.parent_content for c in chunks}))

# 碎片合并 / 纯文本伪标题
txt = "\n".join(f"第 {i} 条 报销需提供凭证" for i in range(1, 41))
doc2 = parse_bytes(txt.encode(), "b.txt")
chunks2 = split_document(doc2, chunk_size=100, chunk_overlap=20, min_chars=80, parent_chunk_size=300)
print("txt chunks:", len(chunks2), [len(c.content) for c in chunks2][:10])
print("txt min len:", min(len(c.content) for c in chunks2))
print("txt section paths:", sorted({c.section_path for c in chunks2})[:3])

# 空文档
empty = parse_bytes(b"", "x.txt") if False else None
from knowflow.ingest.loaders import ParsedDocument, ParsedPage  # noqa: E402

empty_doc = ParsedDocument(
    filename="e.txt", ext="txt", parser="text", text="", pages=[], encoding="utf-8",
    line_count=0, char_count=0,
)
print("empty split:", split_document(empty_doc, chunk_size=100, chunk_overlap=10, min_chars=10, parent_chunk_size=200))

# separators 防护
try:
    _validate_separators(("\n\n", "", "\n"))
except Exception as exc:  # noqa: BLE001
    print("sep guard:", type(exc).__name__, str(exc)[:80])
print("default tail:", DEFAULT_SEPARATORS[-1] == "")

# PDF 不跨页
import io  # noqa: E402

from pypdf import PdfWriter  # noqa: E402
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject  # noqa: E402


def make_pdf(pages_text):
    writer = PdfWriter()
    for text in pages_text:
        page = writer.add_blank_page(width=300, height=200)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 20 150 Td ({text}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(content)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


pdf_doc = parse_bytes(
    make_pdf(["AAA first page content here", "BBB second page content here"]), "p.pdf"
)
pchunks = split_document(pdf_doc, chunk_size=20, chunk_overlap=5, min_chars=5, parent_chunk_size=60)
print("pdf chunks:", [(c.index, c.page_no, c.content[:20]) for c in pchunks])
print("pdf pages per chunk:", sorted({c.page_no for c in pchunks}))

# ---- fusion / autocut / gate / trim ----
from knowflow.retrieval.autocut import autocut, evaluate_gate, trim_to_budget  # noqa: E402
from knowflow.retrieval.fusion import minmax_normalize, rank_of, rrf_fuse, weighted_fuse  # noqa: E402
from knowflow.retrieval.types import Candidate  # noqa: E402

print("minmax same:", minmax_normalize({"a": 0.5, "b": 0.5}))
print("minmax one:", minmax_normalize({"a": 3.0}))
print("minmax empty:", minmax_normalize({}))
print("minmax diff:", minmax_normalize({"a": 0.0, "b": 1.0, "c": 0.5}))
print("rrf:", rrf_fuse([["a", "b"]], k=60))
print("rrf rank1 weight:", 1.0 / 61)
print("rank_of:", rank_of({"a": 0.1, "b": 0.9}))
print("weighted:", weighted_fuse({"a": 0.9, "b": 0.1}, {"b": 5.0, "c": 1.0}, alpha=0.5))


def cand(vid, score, vs=None, cov=0.0, text="x" * 10):
    return Candidate(
        vector_id=vid, doc_id=1, doc_name="d.md", content=text, context_text=text,
        vector_score=vs, keyword_coverage=cov, score=score,
    )


cs = [cand("1:0", 1.0), cand("1:1", 0.5), cand("1:2", 0.1)]
print("autocut ratio .35:", [c.vector_id for c in autocut(cs, ratio=0.35, min_keep=1)])
print("autocut ratio 0:", [c.vector_id for c in autocut(cs, ratio=0.0, min_keep=1, max_keep=2)])
print("autocut empty:", autocut([], ratio=0.5))
print("autocut min_keep 3:", [c.vector_id for c in autocut(cs, ratio=0.9, min_keep=3)])
print("autocut negative top:", [c.vector_id for c in autocut([cand("1:0", -1.0), cand("1:1", -2.0)], ratio=0.5)])

g = evaluate_gate([cand("1:0", 0.5, vs=0.7), cand("1:1", 0.4, vs=None, cov=0.5)], vector_threshold=0.6, keyword_threshold=0.4)
print("gate:", g)
g2 = evaluate_gate([cand("1:0", 0.5, vs=None, cov=0.1)], vector_threshold=0.6, keyword_threshold=0.4)
print("gate bm25-only:", g2, "best_vector:", g2.best_vector_score)
g3 = evaluate_gate([], vector_threshold=0.6, keyword_threshold=0.4)
print("gate none:", g3.reason)

many = [cand(str(i), 1.0 - i * 0.01, text="y" * 100) for i in range(10)]
kept, dropped = trim_to_budget(many, max_chars=250)
print("trim:", len(kept), dropped, [len(c.context_text) for c in kept])
kept2, dropped2 = trim_to_budget(many, max_chars=1)
print("trim tiny(min_keep=1):", len(kept2), dropped2)
kept3, dropped3 = trim_to_budget(many, max_chars=1, min_keep=3)
print("trim tiny(min_keep=3):", len(kept3), dropped3)

# ---- BM25 ----
from knowflow.retrieval.bm25 import BM25Index, BM25Registry  # noqa: E402

idx = BM25Index()
idx.add("d1", "一线城市住宿标准", {"content": "一线城市住宿标准"})
idx.add("d2", "二线城市住宿标准", {"content": "二线城市住宿标准"})
idx.add("d3", "餐饮补贴标准", {"content": "餐饮补贴标准"})
print("avgdl:", idx.avgdl, "docs:", idx.doc_count, "terms:", idx.term_count)
res = idx.search("住宿标准", top_k=5)
print("bm25 search:", [(m.doc_key, round(m.score, 6)) for m in res])

hi = BM25Index()
for i in range(4):
    hi.add(f"h{i}", "公司 公司 公司 差旅", {"content": ""})
print("idf 高频词:", hi._idf("公"), "df:", hi._df.get("公"), "n:", len(hi._payload))
print("hi search:", [(m.doc_key, round(m.score, 6)) for m in hi.search("公司", top_k=4)])

reg = BM25Registry()
reg.add_chunks(1, [{"vector_id": "1:0", "content": "住宿标准 六百元", "doc_id": 1, "doc_name": "a.md", "chunk_index": 0, "page_no": None, "section_path": None}])
reg.add_chunks(2, [{"vector_id": "2:0", "content": "餐饮补贴 一百元", "doc_id": 2, "doc_name": "b.md", "chunk_index": 0, "page_no": None, "section_path": None}])
print("kb1 search:", reg.search(kb_id=1, query="住宿标准", top_k=5))
print("kb2 search:", reg.search(kb_id=2, query="住宿标准", top_k=5))
print("all search:", reg.search(kb_id=None, query="标准", top_k=5))
print("doc_count_of:", reg.doc_count_of(1), reg.doc_count_of(99), reg.doc_count)

# 幂等
i2 = BM25Index()
i2.add("k", "住宿标准", {"content": "住宿标准"})
i2.add("k", "住宿标准", {"content": "住宿标准"})
print("idempotent:", i2.doc_count, i2._df.get("住"), i2.avgdl)

# ---- evaluation ----
from knowflow.evaluation.metrics import mrr, ndcg_at_k, recall_at_k, tokenize  # noqa: E402
from knowflow.evaluation.stats import bootstrap_ci, paired_bootstrap_test  # noqa: E402

print("recall:", recall_at_k(["a", "b", "c"], {"a", "d"}, 3), recall_at_k(["a"], {"a"}, 0), recall_at_k(["a"], set(), 3))
print("recall dup:", recall_at_k(["a", "a", "b"], {"a", "b"}, 2))
print("mrr:", mrr(["x", "a"], {"a"}), mrr(["x"], {"a"}), mrr(["a"], set()))
print("ndcg:", ndcg_at_k(["a", "b"], {"a", "b"}, 2), ndcg_at_k(["b", "a"], {"a"}, 2), ndcg_at_k(["a"], {"a"}, 5))
print("ndcg 3 relevant:", ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3))
print("eval tokenize:", tokenize("一线城市住宿标准"), tokenize("报销，标准"))
ci = bootstrap_ci([1.0, 0.5, 0.0], n_boot=200, seed=7)
ci2 = bootstrap_ci([1.0, 0.5, 0.0], n_boot=200, seed=7)
print("bootstrap:", ci, ci == ci2)
print("bootstrap empty:", bootstrap_ci([]))
print("paired big:", paired_bootstrap_test([1.0] * 20, [0.0] * 20, n_boot=200))
print("paired same:", paired_bootstrap_test([0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8], n_boot=200))
try:
    paired_bootstrap_test([1.0], [1.0, 2.0])
except Exception as exc:  # noqa: BLE001
    print("paired len:", type(exc).__name__)
