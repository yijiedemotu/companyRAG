from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from knowflow.agent.graph import build_agent_graph, build_fixed_chain  # noqa: E402
from knowflow.agent.nodes import AgentDeps  # noqa: E402
from knowflow.agent.runner import AgentRunner  # noqa: E402
from knowflow.agent.tools import ToolRegistry  # noqa: E402
from knowflow.core.config import Settings  # noqa: E402
from knowflow.db.models import Base, Document, TraceSpan  # noqa: E402
from knowflow.db.session import create_engine_from_settings  # noqa: E402
from knowflow.embeddings.hash_embedder import HashEmbedder  # noqa: E402
from knowflow.llm.mock import MockChatModel  # noqa: E402
from knowflow.retrieval.bm25 import BM25Registry  # noqa: E402
from knowflow.retrieval.engine import HybridRetriever  # noqa: E402
from knowflow.services.auth import AuthService  # noqa: E402
from knowflow.services.chat import ChatService  # noqa: E402
from knowflow.services.chunks import SqlChunkEnricher  # noqa: E402
from knowflow.services.document import DocumentService  # noqa: E402
from knowflow.services.knowledge_base import KBService  # noqa: E402
from knowflow.vectorstore.memory_store import InMemoryVectorStore  # noqa: E402

DATA = Path("data") / "_probe3"
CFG = Settings(
    env="test", debug=True, database_url="sqlite+pysqlite:///:memory:", data_dir=DATA,
    embedding_provider="hash", vector_backend="memory", openai_api_key="",
    agent_checkpoint_backend="memory", log_level="WARNING",
)

engine = create_engine_from_settings(CFG)
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
session: Session = factory()

embedder = HashEmbedder(CFG)
vector_store = InMemoryVectorStore(CFG)
bm25 = BM25Registry()
chat_model = MockChatModel(CFG)
judge = MockChatModel(CFG)

kb_service = KBService(session=session, vector_store=vector_store, bm25=bm25, embedder=embedder, settings=CFG)
document_service = DocumentService(session=session, kb_service=kb_service, vector_store=vector_store, bm25=bm25, embedder=embedder, settings=CFG)
retriever = HybridRetriever(embedder=embedder, vector_store=vector_store, bm25=bm25, chat_model=judge, enricher=SqlChunkEnricher(session), settings=CFG)
tools = ToolRegistry(retriever=retriever)
deps = AgentDeps(settings=CFG, chat_model=chat_model, judge_model=judge, retriever=retriever, tools=tools)
agent_graph = build_agent_graph(deps, checkpointer=None)
fixed = build_fixed_chain(deps)
runner = AgentRunner(agent_graph=agent_graph, fixed_graph=fixed, settings=CFG, model_name=chat_model.model)
chat_service = ChatService(session=session, retriever=retriever, runner=runner, kb_service=kb_service, document_service=document_service, chat_model=chat_model, settings=CFG)

auth = AuthService(session=session, settings=CFG)
user, token, exp = auth.register(username="probe", password="passw0rd")
session.commit()
print("user:", user.id, user.role)

kb = kb_service.create(name="探针库", description=None, owner_id=user.id)
session.commit()
print("kb:", kb.id, kb.embedding_provider, kb.embedding_dim)

md = """# 员工报销制度

## 差旅报销标准

一线城市住宿标准为每晚 600 元，二线城市为 400 元。住宿费需提供发票。

## 餐饮报销标准

餐饮补贴为每天 100 元，需要提供发票与审批单。
"""
doc = document_service.ingest_upload(kb_id=kb.id, filename="员工报销制度.md", data=md.encode())
print("doc:", doc.id, doc.status, doc.chunk_count, doc.token_count, doc.ingest_ms, doc.parser, doc.page_count)
stats = kb_service.stats(kb.id)
print("stats:", {k: stats[k] for k in ("doc_count", "ready_doc_count", "chunk_count", "vector_count", "bm25_doc_count", "consistent")})

d2 = document_service.ingest_upload(
    kb_id=kb.id,
    filename="错误码.md",
    data="# 错误码\n\nPAYLOAD_TOO_LARGE 表示请求体超过服务端限制，需要拆分请求。\n".encode(),
)
print("doc2:", d2.id, d2.status, d2.chunk_count)

# 重复上传
from knowflow.core.exceptions import DocumentDuplicateError  # noqa: E402

try:
    document_service.ingest_upload(kb_id=kb.id, filename="copy.md", data=md.encode())
except Exception as exc:  # noqa: BLE001
    print("dup:", type(exc).__name__, getattr(exc, "code", None))

# 解析失败
try:
    document_service.ingest_upload(kb_id=kb.id, filename="bad.md", data=b"   \n  \n")
except Exception as exc:  # noqa: BLE001
    print("bad upload:", type(exc).__name__, getattr(exc, "code", None), str(exc)[:60])
failed = session.execute(select(Document).where(Document.status == "FAILED")).scalars().all()
print("failed rows:", [(f.id, f.filename, f.status, f.error_code, (f.error_message or "")[:40]) for f in failed])

# 检索
for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
    res = chat_service.search(kb_id=kb.id, query="一线城市住宿标准是多少", mode=mode, top_k=5)
    print(f"search {mode}: hits={len(res.candidates)} gate={res.gate.passed}/{res.gate.reason} debug={res.debug.to_dict()}")

res = chat_service.search(kb_id=kb.id, query="量子计算机的制冷机型号是什么", mode="hybrid_rerank", top_k=5)
print("unrelated:", res.gate.passed, res.gate.reason, res.gate.best_vector_score, res.gate.best_keyword_coverage)

# 问答
ans = chat_service.answer(user_id=user.id, question="一线城市住宿标准是多少", kb_id=kb.id, mode="agent")
print("answer:", repr(ans.outcome.answer[:160]))
print("  refusal:", ans.outcome.refusal, "sources:", len(ans.outcome.sources), "rounds:", ans.outcome.retrieval_rounds, "reflect:", ans.outcome.reflect_passed, "usage:", ans.outcome.usage)
print("  cost:", ans.cost_usd, "trace:", ans.trace_id)

ref = chat_service.answer(user_id=user.id, question="量子计算机的制冷机型号是什么", kb_id=kb.id, mode="agent")
print("refusal:", ref.outcome.refusal, repr(ref.outcome.answer[:60]), "sources:", len(ref.outcome.sources))
span_names = [n for n in session.execute(select(TraceSpan.name).where(TraceSpan.trace_id == ref.trace_id)).scalars()]
print("  spans:", span_names)

rag = chat_service.answer(user_id=user.id, question="员工年假怎么算", kb_id=kb.id, mode="rag")
print("rag:", repr(rag.outcome.answer[:80]), rag.outcome.retrieval_rounds)

# 流式
events = []
tokens = []
for frame in chat_service.answer_stream(user_id=user.id, question="住宿标准是多少", kb_id=kb.id, mode="agent", use_memory=False):
    events.append(frame.get("event"))
    if frame.get("event") == "token":
        tokens.append(frame["text"])
print("stream events:", events[:6], "...", events[-3:], "ntokens:", len(tokens))
print("stream text:", repr("".join(tokens)[:80]))

# memory
from knowflow.agent.memory import append_message, load_history, truncate_history  # noqa: E402
from knowflow.llm.base import ChatMessage  # noqa: E402

msgs = [ChatMessage(role="user", content="a" * 10), ChatMessage(role="assistant", content="b" * 10)] * 5
print("truncate turns=2:", [(m.role, len(m.content)) for m in truncate_history(msgs, max_turns=2, max_chars=1000)])
print("truncate chars=15:", [(m.role, len(m.content)) for m in truncate_history(msgs, max_turns=10, max_chars=15)])
print("truncate empty:", truncate_history([], max_turns=3, max_chars=100))
print("truncate only-long:", [(m.role, len(m.content)) for m in truncate_history([ChatMessage(role="user", content="x" * 100)], max_turns=1, max_chars=5)])

# observability
from knowflow.observability.tracing import TraceRecorder, persist, use_recorder  # noqa: E402

rec = TraceRecorder("t" * 32, "chat", settings=CFG)
with use_recorder(rec):
    with rec.span("retrieve", span_type="retrieval", input={"q": "x"}) as sp:
        sp.set_output({"n": 3})
    try:
        with rec.span("grade"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    rec.add_span("generate.llm", span_type="llm", duration_ms=12)
print("summary:", {k: v for k, v in rec.summary().items() if k != "spans"})
persist(rec, session, name="chat", mode="agent", user_id=user.id, latency_ms=42, usage={"prompt_tokens": 10, "completion_tokens": 5, "model": "offline-mock"}, extra={"source_count": 2, "refusal": False})
session.commit()
tid = persist(rec, session, name="chat", latency_ms=43)
session.commit()
spans = session.execute(select(TraceSpan).where(TraceSpan.trace_id == "t" * 32)).scalars().all()
print("persisted spans:", len(spans), [(s.seq, s.name, s.status, s.duration_ms) for s in spans])

from knowflow.observability.metrics import collect_quality, collect_stats  # noqa: E402

s = collect_stats(session, hours=24, settings=CFG)
print("collect_stats:", {k: s[k] for k in ("requests", "errors", "total_tokens", "cost_usd", "latency_ms") if k in s})
print("by_mode:", s["by_mode"])
print("by_model:", s["by_model"])
print("series len:", len(s["series"]))
print("quality:", collect_quality(session, hours=24))

# 删除文档
before = kb_service.stats(kb.id)
out = document_service.delete_document(d2.id)
after = kb_service.stats(kb.id)
print("delete:", out, before["chunk_count"], after["chunk_count"], after["vector_count"], after["bm25_doc_count"], after["consistent"])
