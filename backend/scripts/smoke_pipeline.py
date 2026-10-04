"""端到端链路自检（不经过 HTTP，直接调 service 层）。

这一份脚本的价值：在写 API 与前端之前，先把**业务链路**验证到底。
它覆盖：

    注册 → 建知识库 → 上传 6 份不同格式文档 → 一致性校验
      → 四种检索模式对比（消融）
      → 非流式问答（agent / rag 两种模式）
      → 流式问答（校验 SSE 事件顺序）
      → 拒答路径（无召回时必须不调用生成模型）
      → 会话记忆（多轮指代）
      → 可观测（trace 是否落库）

**完全隔离**：跑在独立的数据库（`knowflow_test`）与独立的 `DATA_DIR`
（`data/_smoke`，每次运行前清空），所以它**不会碰你的开发数据**。

为什么必须隔离：`DATA_DIR`（含 `chroma/`、`uploads/`）与 `DATABASE_URL` 是**一一对应**的。
两者不匹配时，向量库和 MySQL 会串库——这是本项目的已知边界之一
（见 `docs/00-架构与技术选型.md` 第 8 节）。测试更不能踩这个坑。

跑法（在仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\smoke_pipeline.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND / "src"))

# 独立数据库 + 独立数据目录（必须在 import settings 之前设好）
os.environ.setdefault(
    "DATABASE_URL",
    "mysql+pymysql://root:1234@127.0.0.1:3306/knowflow_test?charset=utf8mb4",
)
os.environ["DATA_DIR"] = str(REPO_ROOT / "data" / "_smoke")

from knowflow.container import build_container  # noqa: E402
from knowflow.core.config import get_settings  # noqa: E402
from knowflow.db.models import Base  # noqa: E402
from knowflow.llm.prompts import REFUSAL_ANSWER  # noqa: E402
from knowflow.services.auth import AuthService  # noqa: E402

get_settings.cache_clear()

PASS, FAIL = [], []


def check(label: str, condition: bool, detail: str = "") -> bool:
    (PASS if condition else FAIL).append(label)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  → {detail}" if detail else ""))
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def reset_state(settings) -> None:  # noqa: ANN001
    """清空数据库表 + 隔离的数据目录，并把样例资产复制进隔离目录。"""
    from knowflow.db.session import create_engine_from_settings

    engine = create_engine_from_settings(settings)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()

    # 隔离目录整个重建，保证每次运行起点一致
    if settings.data_dir.exists():
        shutil.rmtree(settings.data_dir, ignore_errors=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # 样例与评测集是"仓库资产"，从主 data 目录复制过来
    source = REPO_ROOT / "data"
    shutil.copytree(source / "samples", settings.sample_dir)
    for asset in ("eval_cases.jsonl", "threshold_calibration.json"):
        src = source / asset
        if src.exists():
            shutil.copy2(src, settings.data_dir / asset)


def main() -> int:
    settings = get_settings()
    print(f"数据库     : {settings.database_url.split('@')[-1]}")
    print(f"数据目录   : {settings.data_dir}  （隔离，不碰开发数据）")
    print(f"离线模式   : {settings.is_offline}")
    print(f"向量后端   : {settings.vector_backend}")

    section("[0] 重置状态（独立测试库 + 独立数据目录）")
    reset_state(settings)
    print("      数据库表已重建；data/_smoke 已重建并复制样例资产")

    section("[1] 装配容器")
    t0 = time.perf_counter()
    container = build_container(settings)
    print(f"      启动耗时 {container.startup_ms} ms")
    print(f"      向量化模式 : {container.embedding_mode()}")
    print(f"      LLM 模式   : {settings.llm_mode} ({container.chat_model.model})")
    print(f"      工具支持   : {container.chat_model.supports_tools()}")
    for warning in container.warnings:
        print(f"      ⚠ {warning}")
    check("容器装配成功", True)
    check(
        "BM25 索引初始化（空库为 0）",
        container.bm25.doc_count >= 0,
        f"{container.bm25.doc_count} 条",
    )

    section("[2] 注册用户 + 建知识库")
    with container.session_factory() as session:
        auth = AuthService(session=session, settings=settings)
        username = f"smoke_{int(time.time()) % 100000}"
        user, token, expires_in = auth.register(
            username=username, password="passw0rd123", display_name="冒烟测试"
        )
        session.commit()
        user_id = user.id
        print(f"      用户 id={user_id} username={user.username} role={user.role}")
        print(f"      token 前 24 位: {token[:24]}… 有效期 {expires_in}s")
        check("注册成功且首个用户是 admin", user.is_admin, user.role)
        check("JWT 可解析", bool(token))

    from sqlalchemy import func, select

    from knowflow.db.models import Chunk, Document, Message, Trace

    with container.session_factory() as session:
        services = container.build_request_services(session)
        kb = services.kb_service.create(
            name=f"冒烟知识库-{int(time.time()) % 100000}",
            description="端到端自检用",
            owner_id=user_id,
        )
        session.commit()
        kb_id = kb.id
        print(f"      知识库 id={kb_id} name={kb.name}")
        print(f"      向量化快照: {kb.embedding_provider}/{kb.embedding_model}/{kb.embedding_dim}d")
        check("知识库创建成功", kb_id > 0)
        check(
            "embedding 维度快照 = 1024",
            kb.embedding_dim == settings.embedding_dim,
            str(kb.embedding_dim),
        )

    section("[3] 上传 6 份不同格式的示例文档")
    samples = sorted((settings.data_dir / "samples").glob("*"))
    print(f"      找到 {len(samples)} 份样例")
    ingest_results = []
    for path in samples:
        data = path.read_bytes()
        with container.session_factory() as session:
            services = container.build_request_services(session)
            t = time.perf_counter()
            try:
                doc = services.document_service.ingest_upload(
                    kb_id=kb_id, filename=path.name, data=data
                )
                elapsed = int((time.perf_counter() - t) * 1000)
                ingest_results.append(
                    (path.name, doc.status, doc.chunk_count, doc.token_count, elapsed)
                )
                print(
                    f"      {path.name:32s} status={doc.status:6s} "
                    f"chunks={doc.chunk_count:3d} tokens={doc.token_count:5d} {elapsed:5d}ms"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"      {path.name:32s} ✗ {type(exc).__name__}: {exc}")
                ingest_results.append((path.name, "FAILED", 0, 0, 0))

    ok_ingest = [r for r in ingest_results if r[1] == "READY"]
    check(
        f"全部 {len(samples)} 份文档入库成功",
        len(ok_ingest) == len(samples),
        f"{len(ok_ingest)}/{len(samples)}",
    )
    total_chunks = sum(r[2] for r in ok_ingest)
    check("切片总数 > 0", total_chunks > 0, f"{total_chunks} 个切片")

    section("[4] 一致性校验：MySQL chunks vs 向量库 vs BM25")
    with container.session_factory() as session:
        services = container.build_request_services(session)
        stats = services.kb_service.stats(kb_id)
        for key in (
            "doc_count",
            "ready_doc_count",
            "chunk_count",
            "vector_count",
            "bm25_doc_count",
            "total_chars",
            "total_tokens",
        ):
            print(f"      {key:18s} = {stats[key]}")
        print(f"      consistent         = {stats['consistent']}")
        check(
            "chunk 数 == 向量库数",
            stats["chunk_count"] == stats["vector_count"],
            f"{stats['chunk_count']} vs {stats['vector_count']}",
        )
        check(
            "chunk 数 == BM25 索引数",
            stats["chunk_count"] == stats["bm25_doc_count"],
            f"{stats['chunk_count']} vs {stats['bm25_doc_count']}",
        )
        check("三者一致（consistent）", bool(stats["consistent"]))
        doc_count = int(
            session.execute(select(func.count(Document.id)).where(Document.kb_id == kb_id)).scalar()
            or 0
        )
        check("documents 表行数 == 上传数", doc_count == len(samples), f"{doc_count}")

    section("[5] 四种检索模式消融对比（同一组问题）")
    probe_questions = [
        ("一线城市住宿标准是多少", "同义/事实"),
        ("PAYLOAD_TOO_LARGE 是什么错误", "专有名词"),
        ("出差住房费用最多能报多少", "同义改写"),
        ("杭州的住宿标准是多少", "跨文档多跳"),
    ]
    modes = ["vector", "bm25", "hybrid", "hybrid_rerank"]
    print(f"      {'问题':28s} " + " ".join(f"{m:>14s}" for m in modes))
    ablation: dict[str, list[int]] = {m: [] for m in modes}
    for question, tag in probe_questions:
        row = []
        for mode in modes:
            with container.session_factory() as session:
                services = container.build_request_services(session)
                result = services.chat_service.search(
                    kb_id=kb_id, query=question, mode=mode, top_k=5
                )
                hits = len(result.candidates)
                ablation[mode].append(hits)
                row.append(f"{hits:>14d}")
        print(f"      {question[:26]:28s} " + " ".join(row))
    for mode in modes:
        print(f"      平均命中 {mode:16s} = {sum(ablation[mode]) / len(ablation[mode]):.2f}")

    section("[6] 检索闸门 + 调试信息（/search 的完整返回）")
    with container.session_factory() as session:
        services = container.build_request_services(session)
        result = services.chat_service.search(
            kb_id=kb_id, query="一线城市住宿标准是多少", mode="hybrid_rerank", top_k=3
        )
        print(
            f"      mode={result.mode} 耗时={result.total_ms}ms "
            f"(embed={result.embedding_ms} vector={result.vector_ms} "
            f"bm25={result.bm25_ms} rerank={result.rerank_ms})"
        )
        print(
            f"      gate: passed={result.gate.passed} reason={result.gate.reason} "
            f"best_vec={result.gate.best_vector_score:.4f} "
            f"best_cov={result.gate.best_keyword_coverage:.4f}"
        )
        print(f"      debug: {result.debug.to_dict()}")
        for i, c in enumerate(result.candidates, start=1):
            print(
                f"      [{i}] score={c.score:.4f} vec={c.vector_score} bm25={c.bm25_score} "
                f"cov={c.keyword_coverage:.2f} {c.source_label}"
            )
            print(f"          {c.snippet[:80]}")
        check("检索有命中", len(result.candidates) > 0, f"{len(result.candidates)} 条")
        check("闸门通过", result.gate.passed, result.gate.reason)
        check("引用带出处（文件名）", all(c.doc_name for c in result.candidates))

    section("[7] 拒答路径：问一个知识库里没有的问题")
    with container.session_factory() as session:
        services = container.build_request_services(session)
        result = services.chat_service.search(
            kb_id=kb_id, query="量子计算机的制冷机型号是什么", mode="hybrid_rerank", top_k=5
        )
        print(f"      gate: passed={result.gate.passed} reason={result.gate.reason}")
        print(
            f"      best_vec={result.gate.best_vector_score:.4f} "
            f"best_cov={result.gate.best_keyword_coverage:.4f}"
        )
        check("无关问题被闸门拦下", not result.gate.passed, result.gate.reason)

        answer = services.chat_service.answer(
            user_id=user_id, question="量子计算机的制冷机型号是什么", kb_id=kb_id, mode="agent"
        )
        print(f"      answer: {answer.outcome.answer[:120]}")
        print(f"      refusal={answer.outcome.refusal} tokens={answer.outcome.usage}")
        check("无召回时直接拒答", answer.outcome.refusal)
        check(
            "拒答用的是固定话术（说明是短路，不是模型生成的）",
            answer.outcome.answer == REFUSAL_ANSWER,
        )
        check("拒答时不产生引用", len(answer.outcome.sources) == 0)
        # 注意口径：Agent 仍然会花一次「便宜的规划调用」（analyze 节点判断要不要检索），
        # 所以 prompt_tokens 不会是 0。真正要保证的是**不调用生成模型**——
        # 下面用 trace 里有没有 `generate.llm` span 来证明。
        print(
            f"      （analyze 规划调用的 token 成本：{answer.outcome.usage.get('prompt_tokens', 0)}，"
            "这是 Agent 形态的固有开销）"
        )
        with container.session_factory() as s3:
            from knowflow.db.models import TraceSpan

            generate_spans = int(
                s3.execute(
                    select(func.count(TraceSpan.id))
                    .where(TraceSpan.trace_id == answer.trace_id)
                    .where(TraceSpan.name == "generate.llm")
                ).scalar()
                or 0
            )
            all_spans = [
                row
                for row in s3.execute(
                    select(TraceSpan.name).where(TraceSpan.trace_id == answer.trace_id)
                ).scalars()
            ]
        print(f"      该 trace 的 span: {all_spans}")
        check(
            "拒答路径下没有调用生成模型（无 generate.llm span）",
            generate_spans == 0,
            f"generate.llm span = {generate_spans}",
        )

    section("[8] 非流式问答（agent 模式）")
    with container.session_factory() as session:
        services = container.build_request_services(session)
        answer = services.chat_service.answer(
            user_id=user_id,
            question="一线城市住宿标准是多少",
            kb_id=kb_id,
            mode="agent",
        )
        o = answer.outcome
        print(f"      答案: {o.answer[:200]}")
        print(f"      conversation_id={answer.conversation_id} message_id={answer.message_id}")
        print(
            f"      refusal={o.refusal} retrieval_rounds={o.retrieval_rounds} "
            f"reflect_passed={o.reflect_passed}"
        )
        print(f"      usage={o.usage} cost=${answer.cost_usd} latency={answer.latency_ms}ms")
        print(f"      引用 {len(o.sources)} 条:")
        for s in o.sources:
            print(
                f"        [{s['rank']}] {s['doc_name']} | {s['section_path']} | score={s['score']}"
            )
        check("agent 模式产出答案", len(o.answer) > 0)
        check("agent 模式有引用", len(o.sources) > 0)
        check("答案里有 [n] 引用编号", "[1]" in o.answer or "[2]" in o.answer, o.answer[:60])
        check("trace_id 已生成", len(answer.trace_id) == 32)
        conv_id = answer.conversation_id

    section("[9] 会话记忆：多轮指代（第二轮问'那二线城市呢'）")
    with container.session_factory() as session:
        services = container.build_request_services(session)
        answer = services.chat_service.answer(
            user_id=user_id,
            question="那二线城市呢？",
            kb_id=kb_id,
            conversation_id=conv_id,
            mode="agent",
        )
        print(f"      答案: {answer.outcome.answer[:200]}")
        with container.session_factory() as s2:
            from knowflow.agent.memory import load_history

            history = load_history(s2, conv_id, settings=settings)
            print(f"      记忆条数 = {len(history)}")
            for m in history:
                print(f"        {m.role}: {m.content[:50]}")
        check("第二轮复用了同一会话", answer.conversation_id == conv_id)
        check("记忆里有历史消息", len(history) >= 3, f"{len(history)} 条")

    section("[10] 流式问答：校验 SSE 事件顺序")
    events: list[str] = []
    tokens: list[str] = []
    with container.session_factory() as session:
        services = container.build_request_services(session)
        stream = services.chat_service.answer_stream(
            user_id=user_id, question="年假有几天", kb_id=kb_id, mode="agent", use_memory=False
        )
        for frame in stream:
            name = frame.get("event", "?")
            events.append(name)
            if name == "token":
                tokens.append(str(frame.get("text", "")))
            elif name == "done":
                print(f"      done: answer={str(frame.get('answer'))[:80]}…")
                print(
                    f"            usage={frame.get('usage')} latency={frame.get('latency_ms')}ms "
                    f"message_id={frame.get('message_id')}"
                )

    print(f"      事件序列: {events}")
    print(f"      token 帧数 = {len(tokens)}，拼接长度 = {len(''.join(tokens))}")
    check("首帧是 meta", events and events[0] == "meta", events[0] if events else "empty")
    check("末帧是 end", events and events[-1] == "end", events[-1] if events else "empty")
    check("有 done 帧", "done" in events)
    check("token 帧数量 > 3（确实在流式）", len(tokens) > 3, str(len(tokens)))
    check(
        "sources 在 token 之前",
        "sources" in events
        and "token" in events
        and events.index("sources") < events.index("token"),
    )
    order_ok = True
    seen_meta, seen_done, seen_end = False, False, False
    for name in events:
        if name == "meta":
            seen_meta = True
        elif name == "done":
            if not seen_meta:
                order_ok = False
            seen_done = True
        elif name == "end":
            if not seen_done:
                order_ok = False
            seen_end = True
        elif seen_done and name != "end":
            order_ok = False
    check("事件顺序 meta → … → done → end", order_ok and seen_meta and seen_end)

    section("[11] rag 快路径 vs agent 全图（延迟对比）")
    timings: dict[str, int] = {}
    for mode in ("rag", "agent"):
        with container.session_factory() as session:
            services = container.build_request_services(session)
            t = time.perf_counter()
            answer = services.chat_service.answer(
                user_id=user_id, question="员工年假怎么算", kb_id=kb_id, mode=mode, use_memory=False
            )
            elapsed = int((time.perf_counter() - t) * 1000)
            timings[mode] = elapsed
            print(
                f"      mode={mode:6s} 总耗时={elapsed:5d}ms "
                f"retrieval_rounds={answer.outcome.retrieval_rounds} "
                f"reflect={answer.outcome.reflect_passed} "
                f"tokens={answer.outcome.usage.get('prompt_tokens', 0)}+"
                f"{answer.outcome.usage.get('completion_tokens', 0)}"
            )
    check("两种模式都能出答案", all(v > 0 for v in timings.values()), str(timings))

    section("[12] 可观测：trace / message / citation 是否落库")
    with container.session_factory() as session:
        trace_count = int(session.execute(select(func.count(Trace.id))).scalar() or 0)
        message_count = int(session.execute(select(func.count(Message.id))).scalar() or 0)
        chunk_count = int(session.execute(select(func.count(Chunk.id))).scalar() or 0)
        print(f"      traces   = {trace_count}")
        print(f"      messages = {message_count}")
        print(f"      chunks   = {chunk_count}")
        check("trace 已落库", trace_count > 0, f"{trace_count} 条")
        check("message 已落库", message_count > 0, f"{message_count} 条")

        if trace_count:
            latest = session.execute(select(Trace).order_by(Trace.id.desc()).limit(1)).scalar_one()
            from knowflow.db.models import TraceSpan

            span_count = int(
                session.execute(
                    select(func.count(TraceSpan.id)).where(TraceSpan.trace_id == latest.trace_id)
                ).scalar()
                or 0
            )
            print(
                f"      最新 trace: id={latest.trace_id[:12]}… name={latest.name} "
                f"mode={latest.mode} latency={latest.latency_ms}ms "
                f"tokens={latest.prompt_tokens}+{latest.completion_tokens} "
                f"cost=${latest.cost_usd} spans={span_count}"
            )
            check("trace 有 span 树", span_count > 0, f"{span_count} 个 span")

    section("[13] 可观测聚合（/obs/stats 的能力）")
    try:
        from knowflow.observability.metrics import collect_quality, collect_stats

        with container.session_factory() as session:
            stats = collect_stats(session, hours=24, settings=settings)
            quality = collect_quality(session, hours=24)
            print(f"      stats.requests = {stats.get('requests')}")
            print(f"      stats.tokens   = {stats.get('tokens')}")
            print(f"      stats.cost_usd = {stats.get('cost_usd')} (¥{stats.get('cost_cny')})")
            print(f"      stats.latency  = {stats.get('latency')}")
            print(f"      by_mode        = {stats.get('by_mode')}")
            print(f"      quality        = {quality}")
            check("可观测聚合可运行", stats.get("requests", 0) > 0, str(stats.get("requests")))
    except Exception as exc:  # noqa: BLE001
        print(f"      ✗ {type(exc).__name__}: {exc}")
        check("可观测聚合可运行", False, str(exc)[:120])

    section("[14] 删除文档：向量与 BM25 是否同步清理")
    with container.session_factory() as session:
        services = container.build_request_services(session)
        docs, _ = services.document_service.list_documents(kb_id=kb_id, page=1, size=100)
        victim = docs[0]
        before = services.kb_service.stats(kb_id)
        print(f"      删除 {victim.filename}（{victim.chunk_count} 切片）")
        outcome = services.document_service.delete_document(victim.id)
        after = services.kb_service.stats(kb_id)
        print(f"      删除结果: {outcome}")
        print(
            f"      chunk={before['chunk_count']}→{after['chunk_count']} "
            f"vector={before['vector_count']}→{after['vector_count']} "
            f"bm25={before['bm25_doc_count']}→{after['bm25_doc_count']}"
        )
        check("chunk 数下降", after["chunk_count"] < before["chunk_count"])
        check("向量数下降", after["vector_count"] < before["vector_count"])
        check("BM25 索引同步下降", after["bm25_doc_count"] < before["bm25_doc_count"])
        check(
            "删除后仍保持一致",
            bool(after["consistent"]),
            f"chunk={after['chunk_count']} vec={after['vector_count']} bm25={after['bm25_doc_count']}",
        )

    container.close()

    section("汇总")
    print(f"  PASS = {len(PASS)}")
    print(f"  FAIL = {len(FAIL)}")
    for name in FAIL:
        print(f"    ✗ {name}")
    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")
    print("RESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
