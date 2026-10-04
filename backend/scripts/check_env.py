"""环境自检：确认所有依赖都能导入，并打印关键版本。"""

import importlib

MODS = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
    "alembic",
    "pymysql",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "langgraph.checkpoint.sqlite",
    "chromadb",
    "sentence_transformers",
    "numpy",
    "pypdf",
    "structlog",
    "jwt",
    "bcrypt",
    "httpx",
    "sse_starlette",
    "aiosqlite",
    "pytest",
    "torch",
    "transformers",
]

failed = 0
for name in MODS:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "")
        print(f"  OK   {name:34s} {ver}")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"  FAIL {name:34s} {type(exc).__name__}: {exc}")

print()
print("RESULT:", "ALL OK" if failed == 0 else f"{failed} FAILED")
