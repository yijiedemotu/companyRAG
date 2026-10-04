"""Alembic 运行环境。

三个关键点：

1. **`target_metadata` 必须包含全部模型**。它来自 `knowflow.db.models`，
   而那个 `__init__.py` 会 import 所有模型类。**漏掉任何一个模型，
   autogenerate 都会生成一条 DROP TABLE** —— 这是最危险的迁移事故，
   所以 `models/__init__.py` 里有一条硬约定要求新增模型必须在那里注册。
2. **URL 只从 settings 取**，不写死在 `alembic.ini`（避免密码进 git、避免双份配置）。
3. **`compare_type=True`**：不然 MySQL 上把 `VARCHAR(64)` 改成 `VARCHAR(128)`
   时 autogenerate 检测不到，迁移里就不会有 ALTER。

用法（在 `backend/` 目录下）：
    ..\\.venv\\Scripts\\python.exe -m alembic revision --autogenerate -m "xxx"
    ..\\.venv\\Scripts\\python.exe -m alembic upgrade head
    ..\\.venv\\Scripts\\python.exe -m alembic downgrade -1
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# 让 alembic 能 import 到 knowflow（源码在 src/ 下，不是默认包路径）
BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knowflow.core.config import get_settings  # noqa: E402
from knowflow.db.models import Base  # noqa: E402  ← 导入所有模型，注册进 metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()


def get_url() -> str:
    """命令行可用 `-x db_url=...` 覆盖（CI 与测试要用不同的库）。"""
    override = context.get_x_argument(as_dictionary=True).get("db_url")
    return override or settings.database_url


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 文本，不连库（用于 review 或交给 DBA 执行）。"""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        # 让迁移文件里的时间戳/注释保持中文可读
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连库执行。"""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # 迁移是一次性任务，不需要连接池
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
