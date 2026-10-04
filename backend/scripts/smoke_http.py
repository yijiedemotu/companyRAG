"""HTTP 层端到端冒烟：真起 uvicorn，真打接口，真解析 SSE。

**为什么必须有这一层**：`pytest` 打的是 ASGI（`httpx.ASGITransport`），
它跳过了真实服务器、真实端口、真实中间件栈与**真实的流式输出**。
而本项目最容易坏的地方恰好都在这一层：

- SSE 被中间件缓冲住（`BaseHTTPMiddleware` 的经典问题）；
- CORS 的 `expose_headers` 没配，前端读不到 `X-Request-ID`；
- 异常处理器没覆盖到，返回的 500 不是统一错误信封；
- 路由前缀 / 依赖注入在真实 `lifespan` 下才暴露问题。

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\smoke_http.py
    .venv\\Scripts\\python.exe backend\\scripts\\smoke_http.py --hash   # 用哈希向量，启动快
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND / "src"))

PASS, FAIL = [], []
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"


def check(label: str, condition: bool, detail: str = "") -> bool:
    (PASS if condition else FAIL).append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  → {detail}" if detail else ""))
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def is_envelope(body: Any) -> bool:
    """统一错误信封：`{"error": {"code", "message", "detail", "request_id", "timestamp"}}`。"""
    if not isinstance(body, dict) or "error" not in body:
        return False
    err = body["error"]
    return isinstance(err, dict) and {"code", "message", "request_id", "timestamp"} <= set(err)


def parse_sse(raw: str) -> list[dict[str, Any]]:
    """把 SSE 原始文本按 `\\n\\n` 分帧，解析成 `[{"event": ..., "data": ...}]`。

    手写解析是刻意的：**前端的 `sse-parser.ts` 做的就是这件事**，
    用第三方库解析会掩盖"帧格式到底对不对"这个问题。
    """
    frames: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event, data_lines = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip(" "))
        if event is None:
            continue
        payload: Any = "\n".join(data_lines)
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            pass
        frames.append({"event": event, "data": payload})
    return frames


def main() -> int:
    # 注意：`global` 必须是函数体的**第一条语句**。
    # 写在 `parser.add_argument(default=PORT)` 之后会报
    # SyntaxError: name 'PORT' is used prior to global declaration。
    global PORT, BASE

    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="用哈希向量（不加载 2GB 模型）")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    PORT = args.port
    BASE = f"http://127.0.0.1:{PORT}"

    if args.hash:
        os.environ["EMBEDDING_PROVIDER"] = "hash"
    os.environ.setdefault("LOG_LEVEL", "WARNING")

    import httpx
    import uvicorn

    from knowflow.core.config import get_settings
    from knowflow.core.logging import configure_logging
    from knowflow.main import create_app

    settings = get_settings()
    configure_logging(level="WARNING", json_output=False)
    print("=" * 78)
    print("KnowFlow HTTP 冒烟测试")
    print("=" * 78)
    print(f"  监听       : {BASE}")
    print(f"  数据库     : {settings.database_url.split('@')[-1]}")
    print(f"  数据目录   : {settings.data_dir}")
    print(f"  向量化     : {settings.embedding_provider}")
    print(f"  API 前缀   : {settings.api_prefix}")

    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    section("[0] 等待服务就绪（会跑 lifespan：建容器 + 重建 BM25）")
    started = time.perf_counter()
    ready = False
    for _ in range(600):  # 最多等 120 秒（本地模型首次加载可能十几秒）
        if server.started:
            ready = True
            break
        time.sleep(0.2)
    print(f"      服务启动耗时 {time.perf_counter() - started:.1f}s，started={ready}")
    check("uvicorn 起来了", ready)

    token: str | None = None
    kb_id: int | None = None
    conversation_id: int | None = None
    message_id: int | None = None

    try:
        with httpx.Client(base_url=BASE, timeout=120.0) as client:
            section("[1] 运维接口（不需要鉴权，且在根路径）")
            r = client.get("/health")
            body = r.json()
            print(f"      GET /health -> {r.status_code}")
            for key in (
                "status",
                "offline",
                "llm_mode",
                "embedding_mode",
                "vector_backend",
                "bm25_doc_count",
                "warnings",
            ):
                print(f"        {key:18s} = {body.get(key)}")
            check("/health 返回 200", r.status_code == 200)
            check("/health 有 offline 字段", "offline" in body)
            check("/health 如实上报 embedding_mode", bool(body.get("embedding_mode")))
            check("/health 返回 bm25_doc_count", "bm25_doc_count" in body)
            db = body.get("db") or {}
            check("/health 报告数据库可用", bool(db.get("ok")), str(db.get("version")))

            r = client.get("/ready")
            print(f"      GET /ready -> {r.status_code}")
            check("/ready 返回 200（已就绪）", r.status_code == 200)

            r = client.get("/metrics")
            print(f"      GET /metrics -> {r.status_code}, {len(r.text)} 字节")
            print(f"        首行: {r.text.splitlines()[0] if r.text else '(空)'}")
            check("/metrics 返回 200", r.status_code == 200)

            section("[2] 鉴权：未带 token 必须 401 且是统一错误信封")
            r = client.get(f"{settings.api_prefix}/kbs")
            print(f"      GET /kbs（无 token）-> {r.status_code} {r.text[:160]}")
            check("无 token 返回 401", r.status_code == 401)
            check("401 是统一错误信封", is_envelope(r.json()))
            check(
                "401 的 code 是 UNAUTHORIZED",
                r.json().get("error", {}).get("code") == "UNAUTHORIZED",
            )

            section("[3] 登录（用 seed_demo 建的管理员）")
            r = client.post(
                f"{settings.api_prefix}/auth/login",
                json={"username": "admin", "password": "admin123"},
            )
            print(f"      POST /auth/login -> {r.status_code}")
            check("登录成功", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                payload = r.json()
                token = payload.get("access_token")
                print(
                    f"        token_type={payload.get('token_type')} "
                    f"expires_in={payload.get('expires_in')} user={payload.get('user', {}).get('username')}"
                )
                check("返回 access_token", bool(token))
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            r = client.post(
                f"{settings.api_prefix}/auth/login",
                json={"username": "admin", "password": "wrong-password"},
            )
            print(f"      POST /auth/login（错密码）-> {r.status_code}")
            check("错密码返回 401", r.status_code == 401)
            check(
                "错密码的 code 是 BAD_CREDENTIALS",
                r.json().get("error", {}).get("code") == "BAD_CREDENTIALS",
            )

            r = client.get(f"{settings.api_prefix}/auth/me", headers=headers)
            print(f"      GET /auth/me -> {r.status_code} {r.text[:120]}")
            check("/auth/me 成功", r.status_code == 200)

            section("[4] 知识库列表（带 token）")
            r = client.get(f"{settings.api_prefix}/kbs", headers=headers)
            print(f"      GET /kbs -> {r.status_code}")
            check("带 token 能列知识库", r.status_code == 200, r.text[:200])
            items = (r.json() or {}).get("items") or []
            print(f"        共 {r.json().get('total')} 个知识库")
            for item in items[:5]:
                print(
                    f"        id={item['id']} name={item['name']!r} "
                    f"docs={item.get('doc_count')} chunks={item.get('chunk_count')}"
                )
            check(
                "Page 结构含 items/total/page/size/pages",
                {"items", "total", "page", "size", "pages"} <= set(r.json()),
            )
            if items:
                kb_id = items[0]["id"]

            section("[5] 422 校验失败也是统一信封")
            r = client.post(f"{settings.api_prefix}/chat", headers=headers, json={"question": ""})
            print(f"      POST /chat（question 为空）-> {r.status_code} {r.text[:200]}")
            check("空 question 返回 422", r.status_code == 422)
            check("422 也是统一错误信封", is_envelope(r.json()))
            check(
                "422 的 code 是 REQUEST_VALIDATION_ERROR",
                r.json().get("error", {}).get("code") == "REQUEST_VALIDATION_ERROR",
            )

            section("[6] 404 也是统一信封")
            r = client.get(f"{settings.api_prefix}/this-path-does-not-exist", headers=headers)
            print(f"      GET /this-path-does-not-exist -> {r.status_code} {r.text[:160]}")
            check("不存在的路径返回 404", r.status_code == 404)
            check("404 也是统一错误信封", is_envelope(r.json()))

            if kb_id is None:
                check("有知识库可用于后续测试", False, "知识库列表为空，请先跑 seed_demo.py")
                raise SystemExit(1)

            section("[7] 知识库统计（三方一致性）")
            r = client.get(f"{settings.api_prefix}/kbs/{kb_id}/stats", headers=headers)
            print(f"      GET /kbs/{kb_id}/stats -> {r.status_code}")
            stats = r.json()
            for key in (
                "doc_count",
                "ready_doc_count",
                "chunk_count",
                "vector_count",
                "bm25_doc_count",
                "total_chars",
                "total_tokens",
                "consistent",
            ):
                print(f"        {key:18s} = {stats.get(key)}")
            check("统计接口成功", r.status_code == 200)
            check(
                "三方一致（consistent）",
                bool(stats.get("consistent")),
                f"chunk={stats.get('chunk_count')} vec={stats.get('vector_count')} "
                f"bm25={stats.get('bm25_doc_count')}",
            )

            section("[8] 检索调试接口（不调用大模型，不花钱）")
            r = client.post(
                f"{settings.api_prefix}/kbs/{kb_id}/search",
                headers=headers,
                json={
                    "query": "一线城市住宿标准是多少",
                    "mode": "hybrid_rerank",
                    "top_k": 3,
                    "include_debug": True,
                },
            )
            print(f"      POST /kbs/{kb_id}/search -> {r.status_code}")
            check("检索接口成功", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                search = r.json()
                print(f"        mode={search.get('mode')} latency={search.get('latency_ms')}ms")
                gate = search.get("gate") or {}
                print(
                    f"        gate: passed={gate.get('passed')} reason={gate.get('reason')} "
                    f"best_vec={gate.get('best_vector_score')} best_cov={gate.get('best_keyword_coverage')}"
                )
                print(f"        debug={search.get('debug')}")
                hits = search.get("hits") or []
                print(f"        命中 {len(hits)} 条：")
                for hit in hits[:3]:
                    print(
                        f"          [{hit['rank']}] {hit['doc_name']} | {hit['section_path']} "
                        f"| score={hit['score']} vec={hit['vector_score']} bm25={hit['bm25_score']}"
                    )
                check("有命中结果", len(hits) > 0)
                check("命中带文件名出处", all(h.get("doc_name") for h in hits))
                check(
                    "debug 含 dropped_orphans（新增字段没被丢掉）",
                    "dropped_orphans" in (search.get("debug") or {}),
                )

            section("[9] 非流式问答")
            r = client.post(
                f"{settings.api_prefix}/chat",
                headers=headers,
                json={"question": "一线城市住宿标准是多少", "kb_id": kb_id, "mode": "agent"},
            )
            print(f"      POST /chat -> {r.status_code}")
            check("问答成功", r.status_code == 200, r.text[:300])
            if r.status_code == 200:
                answer = r.json()
                conversation_id = answer.get("conversation_id")
                message_id = answer.get("message_id")
                print(f"        答案: {str(answer.get('answer'))[:150]}")
                print(
                    f"        conversation_id={conversation_id} message_id={message_id} "
                    f"trace_id={str(answer.get('trace_id'))[:12]}…"
                )
                print(
                    f"        refusal={answer.get('refusal')} "
                    f"retrieval_rounds={answer.get('retrieval_rounds')} "
                    f"reflect_passed={answer.get('reflect_passed')}"
                )
                print(
                    f"        usage={answer.get('usage')} cost_usd={answer.get('cost_usd')} "
                    f"latency_ms={answer.get('latency_ms')}"
                )
                sources = answer.get("sources") or []
                print(f"        引用 {len(sources)} 条：")
                for src in sources[:3]:
                    print(
                        f"          [{src['rank']}] {src['doc_name']} | {src['section_path']} "
                        f"| score={src['score']}"
                    )
                check("有答案正文", len(str(answer.get("answer") or "")) > 0)
                check("有引用来源", len(sources) > 0)
                check("引用的 doc_name 非空", all(s.get("doc_name") for s in sources))
                check(
                    "引用的 rank 从 1 连续",
                    [s.get("rank") for s in sources] == list(range(1, len(sources) + 1)),
                )
                check("响应含 trace_id", bool(answer.get("trace_id")))

            section("[10] 流式问答：校验 SSE 帧格式与事件顺序")
            raw = ""
            with client.stream(
                "POST",
                f"{settings.api_prefix}/chat/stream",
                headers=headers,
                json={
                    "question": "年假有几天",
                    "kb_id": kb_id,
                    "mode": "agent",
                    "use_memory": False,
                },
            ) as response:
                print(
                    f"      POST /chat/stream -> {response.status_code} "
                    f"content-type={response.headers.get('content-type')}"
                )
                check("流式接口返回 200", response.status_code == 200)
                check(
                    "Content-Type 是 text/event-stream",
                    "text/event-stream" in (response.headers.get("content-type") or ""),
                )
                check(
                    "带了 X-Request-ID 响应头",
                    bool(response.headers.get("x-request-id")),
                    str(response.headers.get("x-request-id")),
                )
                chunks = []
                for text in response.iter_text():
                    chunks.append(text)
                raw = "".join(chunks)

            frames = parse_sse(raw)
            events = [f["event"] for f in frames]
            print(f"      收到 {len(frames)} 帧，事件序列：")
            print(f"        {events}")
            for frame in frames:
                if frame["event"] == "error":
                    print(f"      ⚠ error 帧内容: {json.dumps(frame['data'], ensure_ascii=False)}")
            check(
                "没有 error 帧（流正常收尾）",
                "error" not in events,
                str(next((f["data"] for f in frames if f["event"] == "error"), "")),
            )
            check("解析出 SSE 帧", len(frames) > 0)
            check(
                "每帧 data 都能解析成 JSON",
                all(isinstance(f["data"], (dict, list)) for f in frames),
            )
            check("首帧是 meta", events[:1] == ["meta"], str(events[:1]))
            check("末帧是 end", events[-1:] == ["end"], str(events[-1:]))
            check("有 done 帧", "done" in events)
            check(
                "有 token 帧（确实在流式）", events.count("token") > 3, str(events.count("token"))
            )
            check(
                "sources 在 token 之前",
                "sources" in events
                and "token" in events
                and events.index("sources") < events.index("token"),
            )
            meta = next((f["data"] for f in frames if f["event"] == "meta"), {})
            print(
                f"        meta: conversation_id={meta.get('conversation_id')} "
                f"mode={meta.get('mode')} model={meta.get('model')} offline={meta.get('offline')}"
            )
            check("meta 含 conversation_id", bool(meta.get("conversation_id")))
            done = next((f["data"] for f in frames if f["event"] == "done"), {})
            if isinstance(done, dict):
                print(
                    f"        done: message_id={done.get('message_id')} "
                    f"usage={done.get('usage')} latency_ms={done.get('latency_ms')}"
                )
                check("done 含 message_id（已落库）", bool(done.get("message_id")))
                check("done 含 usage", bool(done.get("usage")))
                conversation_id = conversation_id or meta.get("conversation_id")

            section("[11] 会话与消息（校验引用补齐了 doc_name）")
            r = client.get(f"{settings.api_prefix}/conversations", headers=headers)
            print(f"      GET /conversations -> {r.status_code}")
            check("会话列表成功", r.status_code == 200)
            if r.status_code == 200:
                convs = (r.json() or {}).get("items") or []
                print(f"        共 {r.json().get('total')} 个会话")
                for conv in convs[:3]:
                    print(
                        f"        id={conv['id']} title={conv['title']!r} "
                        f"msg_count={conv.get('message_count')} mode={conv.get('mode')}"
                    )
                if convs and conversation_id is None:
                    conversation_id = convs[0]["id"]

            if conversation_id is not None:
                r = client.get(
                    f"{settings.api_prefix}/conversations/{conversation_id}/messages",
                    headers=headers,
                )
                print(f"      GET /conversations/{conversation_id}/messages -> {r.status_code}")
                check("消息列表成功", r.status_code == 200, r.text[:200])
                if r.status_code == 200:
                    msgs = (r.json() or {}).get("items") or []
                    print(f"        共 {r.json().get('total')} 条消息")
                    citation_seen = 0
                    for msg in msgs[-4:]:
                        print(
                            f"        [{msg.get('role')}] {str(msg.get('content'))[:60]!r} "
                            f"citations={len(msg.get('citations') or [])}"
                        )
                        for cite in (msg.get("citations") or [])[:2]:
                            citation_seen += 1
                            print(
                                f"            rank={cite.get('rank')} doc={cite.get('doc_name')} "
                                f"page={cite.get('page_no')} section={cite.get('section_path')} "
                                f"score={cite.get('score')}"
                            )
                    check(
                        "消息里有 assistant 消息", any(m.get("role") == "assistant" for m in msgs)
                    )
                    if citation_seen:
                        has_doc = any(
                            c.get("doc_name") for m in msgs for c in (m.get("citations") or [])
                        )
                        check("历史消息的引用补齐了 doc_name（前端可直接显示文件名）", has_doc)

            section("[12] 可观测接口")
            r = client.get(
                f"{settings.api_prefix}/obs/stats", headers=headers, params={"hours": 24}
            )
            print(f"      GET /obs/stats -> {r.status_code}")
            check("可观测统计成功", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                obs = r.json()
                # ⚠ 注意：这里读的是 `ObsStatsOut`（接口契约模型）的字段，
                # 不是 `collect_stats()` 的原始 dict —— 路由会用 `_to_stats_out`
                # 重塑一次（`by_mode` 从 dict 变成 list，`errors`/`error_rate`
                # 不在契约模型里）。按 collector 的形状写断言会 KeyError/AttributeError。
                print(f"        hours={obs.get('hours')} requests={obs.get('requests')}")
                print(f"        tokens={obs.get('tokens')}")
                print(f"        cost_usd={obs.get('cost_usd')} cost_cny={obs.get('cost_cny')}")
                print(f"        latency={obs.get('latency')}")
                by_mode = obs.get("by_mode") or []
                print(f"        by_mode: {len(by_mode)} 组")
                for item in by_mode[:3]:
                    print(f"          {item}")
                by_model = obs.get("by_model") or []
                print(f"        by_model: {len(by_model)} 组")
                for item in by_model[:3]:
                    print(f"          {item}")
                timeline = obs.get("timeline") or []
                print(f"        timeline 点数 = {len(timeline)}")
                check("统计里有请求数", int(obs.get("requests") or 0) > 0, str(obs.get("requests")))
                check(
                    "tokens 是嵌套结构（含 prompt/completion/total）",
                    isinstance(obs.get("tokens"), dict)
                    and {"prompt", "completion", "total"} <= set(obs["tokens"]),
                )
                check(
                    "latency 是嵌套结构（含 p50/p95/p99）",
                    isinstance(obs.get("latency"), dict)
                    and {"p50", "p95", "p99"} <= set(obs["latency"]),
                )
                check("timeline 是列表（前端画折线图用）", isinstance(timeline, list))

            r = client.get(
                f"{settings.api_prefix}/obs/traces", headers=headers, params={"page": 1, "size": 5}
            )
            print(f"      GET /obs/traces -> {r.status_code}")
            check("trace 列表成功", r.status_code == 200, r.text[:200])
            trace_id = None
            if r.status_code == 200:
                traces = (r.json() or {}).get("items") or []
                print(f"        共 {r.json().get('total')} 条 trace")
                for tr in traces[:3]:
                    print(
                        f"        trace_id={str(tr.get('trace_id'))[:12]}… name={tr.get('name')} "
                        f"mode={tr.get('mode')} latency={tr.get('latency_ms')}ms "
                        f"tokens={tr.get('prompt_tokens')}+{tr.get('completion_tokens')} "
                        f"spans={tr.get('span_count')}"
                    )
                if traces:
                    trace_id = traces[0].get("trace_id")
                check("trace 已落库", len(traces) > 0)

            if trace_id:
                r = client.get(f"{settings.api_prefix}/obs/traces/{trace_id}", headers=headers)
                print(f"      GET /obs/traces/{str(trace_id)[:12]}… -> {r.status_code}")
                check("trace 详情成功", r.status_code == 200, r.text[:200])
                if r.status_code == 200:
                    detail = r.json()
                    spans = detail.get("spans") or []
                    print(f"        span 数={len(spans)}")
                    for span in spans[:8]:
                        print(
                            f"          seq={span.get('seq')} {span.get('name'):14s} "
                            f"type={span.get('span_type'):10s} "
                            f"+{span.get('start_offset_ms')}ms {span.get('duration_ms')}ms "
                            f"{span.get('status')}"
                        )
                    check("trace 有 span 树", len(spans) > 0)
                    check(
                        "span 的 seq 连续",
                        [s.get("seq") for s in spans] == list(range(1, len(spans) + 1)),
                        str([s.get("seq") for s in spans]),
                    )

            r = client.get(
                f"{settings.api_prefix}/obs/quality", headers=headers, params={"hours": 24}
            )
            print(f"      GET /obs/quality -> {r.status_code}")
            check("质量指标成功", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                print(f"        {r.json()}")

            section("[13] 评测接口")
            r = client.get(f"{settings.api_prefix}/eval/datasets", headers=headers)
            print(f"      GET /eval/datasets -> {r.status_code}")
            check("数据集列表成功", r.status_code == 200, r.text[:200])
            if r.status_code == 200:
                datasets = r.json()
                print(
                    f"        返回类型={'数组' if isinstance(datasets, list) else type(datasets).__name__}，"
                    f"共 {len(datasets)} 个数据集"
                )
                for ds in (datasets if isinstance(datasets, list) else [])[:3]:
                    print(
                        f"        id={ds.get('id')} name={ds.get('name')} cases={ds.get('case_count')}"
                    )

            r = client.get(f"{settings.api_prefix}/eval/runs", headers=headers)
            print(f"      GET /eval/runs -> {r.status_code}")
            check("评测运行列表成功", r.status_code == 200, r.text[:200])

            section("[14] 响应头（CORS 暴露 + request-id）")
            r = client.get(f"{settings.api_prefix}/kbs", headers=headers)
            print(f"      X-Request-ID       = {r.headers.get('x-request-id')}")
            print(f"      X-Process-Time-Ms  = {r.headers.get('x-process-time-ms')}")
            check("响应带 X-Request-ID", bool(r.headers.get("x-request-id")))
            check("响应带 X-Process-Time-Ms", bool(r.headers.get("x-process-time-ms")))
            try:
                float(r.headers.get("x-process-time-ms", ""))
                check("X-Process-Time-Ms 可解析为浮点数", True)
            except ValueError:
                check(
                    "X-Process-Time-Ms 可解析为浮点数",
                    False,
                    str(r.headers.get("x-process-time-ms")),
                )

            r = client.get(
                f"{settings.api_prefix}/kbs", headers=headers, params={"page": 1, "size": 999}
            )
            print(f"      GET /kbs?size=999 -> {r.status_code}（分页上限校验）")
            check("size 超过上限被拒（422）", r.status_code == 422)

    finally:
        section("关闭服务")
        server.should_exit = True
        thread.join(timeout=15)
        print(f"      已停止（thread alive={thread.is_alive()}）")

    section("汇总")
    print(f"  PASS = {len(PASS)}")
    print(f"  FAIL = {len(FAIL)}")
    for name in FAIL:
        print(f"    ✗ {name}")
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
