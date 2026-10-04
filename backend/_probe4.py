from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jwt  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from knowflow.core.config import Settings, _mask_dsn  # noqa: E402

print("offline ' ':", Settings(openai_api_key=" ", _env_file=None).is_offline)
print("mask:", _mask_dsn("mysql+pymysql://root:1234@127.0.0.1:3306/knowflow?charset=utf8mb4"))
print("mask no password:", _mask_dsn("postgresql://host/db"))
print("prefix:", Settings(api_prefix="api/v1/", _env_file=None).api_prefix,
      Settings(api_prefix="/api/v1", _env_file=None).api_prefix,
      Settings(api_prefix="/", _env_file=None).api_prefix)

os.environ["CORS_ORIGINS"] = "http://a,http://b"
os.environ["ALLOWED_EXTENSIONS"] = "md, .TXT ,pdf"
from knowflow.core.config import get_settings  # noqa: E402

get_settings.cache_clear()
s = Settings(_env_file=None)
print("cors comma:", s.cors_origins)
print("ext normalize:", s.allowed_extensions)
os.environ["CORS_ORIGINS"] = '["http://x"]'
get_settings.cache_clear()
print("cors json:", Settings(_env_file=None).cors_origins)
del os.environ["CORS_ORIGINS"]
del os.environ["ALLOWED_EXTENSIONS"]
get_settings.cache_clear()
print("direct comma:", Settings(cors_origins="a, b", _env_file=None).cors_origins)
print("direct json:", Settings(cors_origins='["a"]', _env_file=None).cors_origins)
print("empty:", Settings(cors_origins="", _env_file=None).cors_origins)

for kwargs in (
    {"chunk_size": 100, "chunk_overlap": 100},
    {"chunk_size": 800, "parent_chunk_size": 700},
    {"rerank_top_n": 20, "fetch_k": 10},
):
    try:
        Settings(_env_file=None, **kwargs)
        print("NO ERROR", kwargs)
    except ValidationError as exc:
        print("ValidationError:", kwargs, str(exc).split("\n")[2][:80])

# security
from knowflow.core.exceptions import UnauthorizedError  # noqa: E402
from knowflow.core.security import (  # noqa: E402
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

SEC = Settings(_env_file=None, jwt_secret="x" * 40)
h1 = hash_password("passw0rd")
h2 = hash_password("passw0rd")
print("salt:", h1 != h2, verify_password("passw0rd", h1), verify_password("passw0rd", h2), verify_password("nope", h1))
long_pw = "汉" * 50
print("long pw:", len(long_pw.encode()), hash_password(long_pw) != hash_password(long_pw), verify_password(long_pw, hash_password(long_pw)))
print("long pw cross:", verify_password("汉" * 49, hash_password(long_pw)))
emoji = "🔒" * 30
print("emoji pw:", len(emoji.encode()), verify_password(emoji, hash_password(emoji)))
print("72 exactly:", verify_password("a" * 72, hash_password("a" * 72)))
print("prepared:", len(__import__("knowflow.core.security", fromlist=["_prepare_password"])._prepare_password(long_pw)))

tok, exp = create_access_token(subject=7, extra={"role": "admin"}, settings=SEC)
print("token:", decode_access_token(tok, settings=SEC)["sub"], exp)
print("tampered:", end=" ")
bad = tok[:-1] + ("a" if tok[-1] != "a" else "b")
try:
    decode_access_token(bad, settings=SEC)
except UnauthorizedError as exc:
    print(type(exc).__name__, exc.code)
past = {"sub": "1", "iat": int(time.time()) - 100, "exp": int(time.time()) - 10, "iss": SEC.app_name}
try:
    decode_access_token(jwt.encode(past, SEC.jwt_secret, algorithm="HS256"), settings=SEC)
except UnauthorizedError as exc:
    print("expired:", exc.code, exc.message)
forged = jwt.encode({**past, "exp": int(time.time()) + 100, "iss": "evil"}, SEC.jwt_secret, algorithm="HS256")
try:
    decode_access_token(forged, settings=SEC)
except UnauthorizedError as exc:
    print("issuer:", exc.code, exc.message)
missing = jwt.encode({"exp": int(time.time()) + 100, "iss": SEC.app_name}, SEC.jwt_secret, algorithm="HS256")
try:
    decode_access_token(missing, settings=SEC)
except UnauthorizedError as exc:
    print("missing sub:", exc.code)

# chroma
import tempfile  # noqa: E402

from knowflow.embeddings.hash_embedder import HashEmbedder  # noqa: E402
from knowflow.vectorstore.base import VectorItem  # noqa: E402
from knowflow.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402
from knowflow.vectorstore.memory_store import InMemoryVectorStore  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="kf-chroma-"))
CS = Settings(_env_file=None, embedding_dim=64, chroma_dir=tmp / "chroma", data_dir=tmp)
emb = HashEmbedder(CS)
mem = InMemoryVectorStore(CS)
t0 = time.perf_counter()
chroma = ChromaVectorStore(CS)
print("chroma init ms:", int((time.perf_counter() - t0) * 1000))

texts = ["一线城市住宿标准为每晚六百元", "餐饮补贴每天一百元", "PAYLOAD_TOO_LARGE 错误码说明"]
items = [
    VectorItem(
        id=f"1:{i}",
        vector=emb.embed_documents([t])[0],
        content=t,
        metadata={"doc_id": 1, "doc_name": "a.md", "chunk_index": i, "page_no": None, "section_path": None, "ext": "md"},
    )
    for i, t in enumerate(texts)
]
t0 = time.perf_counter()
print("mem upsert:", mem.upsert(kb_id=7, items=items))
print("chroma upsert:", chroma.upsert(kb_id=7, items=items))
print("upsert ms:", int((time.perf_counter() - t0) * 1000))
q = emb.embed_query("一线城市住宿标准")
mq = mem.query(kb_id=7, vector=q, top_k=3)
cq = chroma.query(kb_id=7, vector=q, top_k=3)
print("mem:", [(h.id, round(h.score, 6)) for h in mq])
print("chroma:", [(h.id, round(h.score, 6)) for h in cq])
print("same top1:", mq[0].id == cq[0].id, "score diff:", abs(mq[0].score - cq[0].score))
print("chroma metadata:", cq[0].metadata)
print("counts:", mem.count(kb_id=7), chroma.count(kb_id=7), chroma.count())
print("list ids:", mem.list_vector_ids(kb_id=7), chroma.list_vector_ids(kb_id=7))
print("delete doc:", mem.delete(kb_id=7, doc_id=1), chroma.delete(kb_id=7, doc_id=1))
print("after delete:", mem.count(kb_id=7), chroma.count(kb_id=7))
print("query after delete:", len(mem.query(kb_id=7, vector=q, top_k=3)), len(chroma.query(kb_id=7, vector=q, top_k=3)))
print("health:", chroma.health())
chroma.reset(kb_id=7)
print("after reset:", chroma.count(kb_id=7), chroma.health())
# 空集查询不应建出集合
print("empty query:", chroma.query(kb_id=99, vector=q, top_k=3), chroma.count())


# 惰性导入
import importlib  # noqa: E402

for mod in ("torch", "sentence_transformers"):
    sys.modules.pop(mod, None)
importlib.import_module("knowflow.embeddings")
print("lazy:", "torch" in sys.modules, "sentence_transformers" in sys.modules)
