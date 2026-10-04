"""模型自检：在真实 MySQL 上建表，确认 14 张表都能建出来。

为什么要单独跑这一步：SQLAlchemy 模型在**导入阶段**不会暴露大部分错误，
只有真正 `CREATE TABLE` 时才会发现
（比如 MySQL 不接受某个类型、外键类型不匹配、索引名过长）。
在写 Alembic 迁移之前先确认模型本身是对的，能把两类问题分开定位。

用法：
    .venv\\Scripts\\python.exe backend\\scripts\\_check_models.py [database_url]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sqlalchemy import inspect, text  # noqa: E402

from knowflow.core.config import Settings  # noqa: E402
from knowflow.db.models import Base  # noqa: E402
from knowflow.db.session import create_engine_from_settings  # noqa: E402

EXPECTED_TABLES = {
    "users",
    "knowledge_bases",
    "documents",
    "chunks",
    "conversations",
    "messages",
    "message_citations",
    "feedback",
    "traces",
    "trace_spans",
    "eval_datasets",
    "eval_cases",
    "eval_runs",
    "eval_case_results",
}


def main() -> int:
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("mysql+pymysql://root:1234@127.0.0.1:3306/knowflow_test?charset=utf8mb4")
    )
    settings = Settings(database_url=url, env="test")
    engine = create_engine_from_settings(settings)

    print(f"目标库: {engine.url.render_as_string(hide_password=True)}")
    print(f"方言  : {engine.dialect.name} / {engine.dialect.driver}")

    print("\n[1/4] drop_all（清掉上次残留）")
    Base.metadata.drop_all(engine)

    print("[2/4] create_all")
    Base.metadata.create_all(engine)

    print("[3/4] 校验表清单")
    inspector = inspect(engine)
    actual = set(inspector.get_table_names())
    print(f"      建出 {len(actual)} 张表")
    missing = EXPECTED_TABLES - actual
    extra = actual - EXPECTED_TABLES
    if missing:
        print(f"      ✗ 缺失: {sorted(missing)}")
    if extra:
        print(f"      ? 多出: {sorted(extra)}")
    if not missing and not extra:
        print("      ✓ 与契约中的 14 张表完全一致")

    print("[4/4] 逐个表打印列数/索引数/外键数")
    ok = True
    for table in sorted(EXPECTED_TABLES & actual):
        cols = inspector.get_columns(table)
        idx = inspector.get_indexes(table)
        fks = inspector.get_foreign_keys(table)
        uniq = inspector.get_unique_constraints(table)
        pk = inspector.get_pk_constraint(table)
        print(
            f"      {table:20s} 列={len(cols):2d} 主键={pk.get('constrained_columns')} "
            f"索引={len(idx)} 唯一={len(uniq)} 外键={len(fks)}"
        )
        if not pk.get("constrained_columns"):
            print(f"        ✗ {table} 没有主键")
            ok = False

    # 额外验证：utf8mb4 能否存 emoji（这是 MySQL 最常见的字符集坑）
    print("\n[附加] 字符集验证：写入含 emoji 与生僻字的数据")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS _charset_probe"))
        conn.execute(
            text(
                "CREATE TABLE _charset_probe (id INT PRIMARY KEY, v VARCHAR(64)) "
                "DEFAULT CHARSET=utf8mb4"
            )
        )
        probe = "一线城市🏨住宿标准"
        conn.execute(text("INSERT INTO _charset_probe VALUES (1, :v)"), {"v": probe})
        got = conn.execute(text("SELECT v FROM _charset_probe WHERE id=1")).scalar()
        conn.execute(text("DROP TABLE _charset_probe"))
        print(f"      写入 {probe!r} -> 读回 {got!r}  {'✓' if got == probe else '✗ 不一致'}")
        ok = ok and got == probe

    print("\n[清理] drop_all")
    Base.metadata.drop_all(engine)
    engine.dispose()

    print("\nRESULT:", "OK" if (ok and not missing) else "FAILED")
    return 0 if (ok and not missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
