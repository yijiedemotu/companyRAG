"""集成自检：逐层导入，报告缺失/报错的模块。"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

MODULES = [
    "knowflow",
    "knowflow.core.config",
    "knowflow.core.exceptions",
    "knowflow.core.logging",
    "knowflow.core.security",
    "knowflow.db.base",
    "knowflow.db.types",
    "knowflow.db.session",
    "knowflow.db.models",
    "knowflow.llm.tokenizer",
    "knowflow.llm.base",
    "knowflow.llm.prompts",
    "knowflow.llm.parsing",
    "knowflow.llm.mock",
    "knowflow.llm.openai_compat",
    "knowflow.llm.factory",
    "knowflow.retrieval.text",
    "knowflow.retrieval.bm25",
    "knowflow.retrieval.types",
    "knowflow.retrieval.fusion",
    "knowflow.retrieval.autocut",
    "knowflow.retrieval.rerank",
    "knowflow.retrieval.engine",
    "knowflow.retrieval",
    "knowflow.embeddings.base",
    "knowflow.embeddings.factory",
    "knowflow.vectorstore.base",
    "knowflow.vectorstore.factory",
    "knowflow.ingest.loaders",
    "knowflow.ingest.chunkers",
    "knowflow.ingest.pipeline",
    "knowflow.observability.tracing",
    "knowflow.observability.metrics",
    "knowflow.observability.pricing",
    "knowflow.evaluation.metrics",
    "knowflow.agent.state",
    "knowflow.agent.tools",
    "knowflow.agent.memory",
    "knowflow.agent.nodes",
    "knowflow.agent.graph",
    "knowflow.agent.runner",
    "knowflow.agent",
    "knowflow.services.chunks",
    "knowflow.services.knowledge_base",
    "knowflow.services.document",
    "knowflow.services.chat",
    "knowflow.services.obs_bridge",
    "knowflow.services.auth",
    "knowflow.schemas",
    "knowflow.middleware.context",
    "knowflow.container",
]

ok, failed = [], []
for name in MODULES:
    try:
        __import__(name)
        ok.append(name)
        print(f"  OK   {name}")
    except Exception as exc:  # noqa: BLE001
        failed.append((name, exc))
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")

print()
print(f"OK={len(ok)}  FAIL={len(failed)}")
if failed:
    print("\n--- 失败详情 ---")
    for name, exc in failed:
        print(f"\n### {name}")
        traceback.print_exception(type(exc), exc, exc.__traceback__, limit=6)
