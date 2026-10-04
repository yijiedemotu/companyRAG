"""一键准备可演示环境：管理员账号 + 示例知识库 + 6 份文档 + 评测数据集。

**为什么需要这个脚本**：`clone → 跑起来 → 看到效果` 之间的路程越短越好。
新手最常卡在"要先手动建库、再手动注册、再手动上传文件"，
任何一步出错就放弃了。这个脚本把它变成一条命令，并且**可重复执行**（幂等）。

做四件事：
  1. 确保表结构存在（用 `Base.metadata.create_all`，生产请用 `alembic upgrade head`）；
  2. 建管理员账号（默认 admin / admin123，**仅用于本地演示**）；
  3. 建示例知识库并把 `data/samples/` 下 6 份文档全部入库（按 sha256 去重）；
  4. 把 `data/eval_cases.jsonl` 的 43 条标注问答导入成评测数据集。

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\seed_demo.py
    .venv\\Scripts\\python.exe backend\\scripts\\seed_demo.py --reset   # 先清空演示知识库
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

from knowflow.container import build_container  # noqa: E402
from knowflow.core.config import get_settings  # noqa: E402
from knowflow.core.logging import configure_logging, get_logger  # noqa: E402
from knowflow.db.models import Base  # noqa: E402
from knowflow.db.session import create_engine_from_settings  # noqa: E402

logger = get_logger(__name__)

DEMO_KB_NAME = "KnowFlow 演示知识库"
DEMO_DATASET_NAME = "knowflow-golden-v1"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 KnowFlow 演示环境")
    parser.add_argument("--reset", action="store_true", help="先删除同名演示知识库再重建")
    parser.add_argument("--admin-password", default=ADMIN_PASSWORD)
    parser.add_argument("--no-eval", action="store_true", help="跳过评测数据集导入")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    settings.ensure_dirs()

    print("=" * 78)
    print("KnowFlow 演示环境准备")
    print("=" * 78)
    print(f"  数据库   : {settings.database_url.split('@')[-1]}")
    print(f"  数据目录 : {settings.data_dir}")
    print(f"  向量化   : {settings.embedding_provider}   向量库: {settings.vector_backend}")
    print(f"  离线模式 : {settings.is_offline}")

    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    engine.dispose()
    print("\n[1/4] 表结构就绪")

    # 容器在这里才建（它会预热 embedding 模型 + 重建 BM25 索引）
    print("[2/4] 装配容器（首次会加载向量模型，请稍等）...")
    started = time.perf_counter()
    container = build_container(settings)
    print(
        f"      完成，耗时 {time.perf_counter() - started:.1f}s；"
        f"向量化模式 = {container.embedding_mode()}"
    )

    from sqlalchemy import select

    from knowflow.db.models import KnowledgeBase, User
    from knowflow.evaluation.datasets import (
        case_from_dict,
        load_jsonl,
        seed_dataset,
    )
    from knowflow.services.auth import AuthService

    # ---------- 管理员 ----------
    with container.session_factory() as session:
        auth = AuthService(session=session, settings=settings)
        user = session.execute(
            select(User).where(User.username == ADMIN_USERNAME)
        ).scalar_one_or_none()
        if user is None:
            user, _token, _exp = auth.register(
                username=ADMIN_USERNAME,
                password=args.admin_password,
                display_name="管理员",
            )
            session.commit()
            print(
                f"[3/4] 已创建管理员账号：{ADMIN_USERNAME} / {args.admin_password}"
                f"（role={user.role}，仅本地演示用）"
            )
        else:
            print(f"[3/4] 管理员账号已存在：{ADMIN_USERNAME}（role={user.role}）")
        admin_id = user.id

    # ---------- 知识库 ----------
    with container.session_factory() as session:
        services = container.build_request_services(session)
        existing = session.execute(
            select(KnowledgeBase).where(KnowledgeBase.name == DEMO_KB_NAME)
        ).scalar_one_or_none()

        if existing is not None and args.reset:
            print(f"      --reset：删除旧知识库 id={existing.id}")
            services.kb_service.delete(existing.id)
            session.commit()
            existing = None

        if existing is None:
            kb = services.kb_service.create(
                name=DEMO_KB_NAME,
                description="由 seed_demo.py 生成的演示知识库（6 份样例文档）",
                owner_id=admin_id,
            )
            session.commit()
            print(f"      创建知识库 id={kb.id} name={kb.name}")
        else:
            kb = existing
            print(f"      复用已有知识库 id={kb.id}")
        kb_id = kb.id

    # ---------- 文档 ----------
    samples = sorted((settings.data_dir / "samples").glob("*"))
    print(f"\n[4/4] 入库 {len(samples)} 份样例文档")
    ok, skipped, failed = 0, 0, 0
    for path in samples:
        data = path.read_bytes()
        with container.session_factory() as session:
            services = container.build_request_services(session)
            try:
                doc = services.document_service.ingest_upload(
                    kb_id=kb_id, filename=path.name, data=data
                )
                ok += 1
                print(
                    f"      {path.name:28s} {doc.status:6s} chunks={doc.chunk_count:3d} "
                    f"{doc.ingest_ms:5d}ms"
                )
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                if name == "DocumentDuplicateError":
                    skipped += 1
                    print(f"      {path.name:28s} 已存在，跳过")
                else:
                    failed += 1
                    print(f"      {path.name:28s} 失败：{name}: {exc}")

    # ---------- 评测数据集 ----------
    if not args.no_eval:
        case_path = settings.data_dir / "eval_cases.jsonl"
        if case_path.exists():
            raw_cases = load_jsonl(case_path)
            cases = [case_from_dict(row) for row in raw_cases]
            with container.session_factory() as session:
                dataset_id, case_count = seed_dataset(
                    session,
                    name=DEMO_DATASET_NAME,
                    description="KnowFlow 标注评测集：事实型 / 同义改写 / 专有名词 / 多跳",
                    cases=cases,
                )
                session.commit()
            print(
                f"\n      评测数据集 id={dataset_id} name={DEMO_DATASET_NAME} 用例 {case_count} 条"
            )
        else:
            print(f"\n      ⚠ 未找到 {case_path}，跳过评测数据集")

    # ---------- 汇总 ----------
    with container.session_factory() as session:
        services = container.build_request_services(session)
        stats = services.kb_service.stats(kb_id)

    container.close()

    print("\n" + "=" * 78)
    print("准备完成")
    print("=" * 78)
    print(f"  管理员      : {ADMIN_USERNAME} / {args.admin_password}")
    print(f"  知识库      : id={kb_id}  「{DEMO_KB_NAME}」")
    print(f"  文档        : {stats['doc_count']} 篇（就绪 {stats['ready_doc_count']}）")
    print(
        f"  切片 / 向量 / BM25 : {stats['chunk_count']} / "
        f"{stats['vector_count']} / {stats['bm25_doc_count']}  "
        f"一致性={'✓' if stats['consistent'] else '✗'}"
    )
    print(f"  总字数/token: {stats['total_chars']} / {stats['total_tokens']}")
    print(
        f"  向量化模式  : {stats['embedding_provider']}/{stats['embedding_model']}"
        f"/{stats['embedding_dim']}d"
    )
    print("\n  下一步：")
    print(
        "    1) 起后端  .venv\\Scripts\\python.exe -m uvicorn knowflow.main:app "
        "--app-dir backend/src --port 8000"
    )
    print("    2) 起前端  cd frontend && pnpm dev")
    print("    3) 打开    http://127.0.0.1:5173  用上面的管理员账号登录")
    if failed:
        print(f"\n  ⚠ 有 {failed} 份文档入库失败，请检查上面的错误信息")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
