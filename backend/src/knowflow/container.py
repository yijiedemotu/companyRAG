"""依赖容器：**全项目唯一的装配入口**。

**为什么必须有这一层**（面试常问）：

1. **启动即失败**：所有重组件（配置校验、数据库连通、向量库目录、embedding 模型）
   在 `build_container()` 里一次性初始化。出问题进程直接起不来，
   而不是"服务起来了，第一个请求才 500" —— 后者在容器化环境里最难查。
2. **装配入口唯一**：全项目只有这里 `new` 向量库/模型/图。
   如果散落在各处，测试里就没法替换成假实现，也没人能说清"到底装了些什么"。
3. **请求级隔离**：`build_request_services(session)` 为每个请求组装一套
   带 session 的服务（检索器、工具、图运行器）。
   **每次请求重建而不是共享一个带 session 的实例** —— 共享就会串号，
   这是"用户 A 的检索用了用户 B 的数据库会话"这类事故的根源。
   代价是每次请求多几毫秒的对象构造，换来的是不可能出错。

生命周期：
    build_container()          进程启动一次（lifespan）
      └─ build_request_services()  每个请求一次（api/deps.py）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from knowflow.agent.graph import build_agent_graph, build_checkpointer, build_fixed_chain
from knowflow.agent.nodes import AgentDeps, RecorderBox
from knowflow.agent.runner import AgentRunner
from knowflow.agent.tools import ToolRegistry
from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.db.session import create_engine_from_settings, get_session_factory
from knowflow.embeddings.base import Embedder, warmup_embedder
from knowflow.embeddings.factory import build_embedder, get_embedder_status
from knowflow.llm.base import ChatModel
from knowflow.llm.factory import build_chat_model, build_judge_model
from knowflow.retrieval.bm25 import BM25Registry
from knowflow.retrieval.engine import HybridRetriever
from knowflow.services.chat import ChatService
from knowflow.services.document import DocumentService
from knowflow.services.knowledge_base import KBService
from knowflow.vectorstore.base import VectorStore
from knowflow.vectorstore.factory import build_vector_store

logger = get_logger(__name__)


@dataclass(slots=True)
class RequestServices:
    """一个请求内可用的全部服务。由 `api/deps.py` 组装并注入路由。"""

    session: Session
    kb_service: KBService
    document_service: DocumentService
    chat_service: ChatService
    retriever: HybridRetriever
    tool_registry: ToolRegistry
    runner: AgentRunner
    #: 本次请求的 trace 持有者。`ChatService` 创建 recorder 后写进来，图里的节点从它读。
    #: **显式传递而不是 contextvar**：流式路径的生成器会被 Starlette 用
    #: `iterate_in_threadpool` 在不同线程上分次驱动，`ContextVar.set()` 返回的 token
    #: 无法在另一个 context 里 `reset()`（实测报
    #: `ValueError: Token ... was created in a different Context`）。
    #: 详见 `agent/nodes.py::RecorderBox`。
    recorder_box: RecorderBox


@dataclass(slots=True)
class Container:
    """进程级单例集合。"""

    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    embedder: Embedder
    vector_store: VectorStore
    bm25: BM25Registry
    chat_model: ChatModel
    judge_model: ChatModel
    checkpointer: Any | None
    startup_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    # BM25 重建结果（/health 要报出来：为 0 就说明关键词检索是瞎的）
    bm25_rebuilt_chunks: int = 0

    # ------------------------------------------------------------------ 状态
    def embedding_mode(self) -> str:
        """当前**真实生效**的向量化模式，例如 `local` 或 `hash(fallback:load_failed)`。

        `get_embedder_status(...).mode` 是唯一数据来源 —— 不要在这里自己拼字符串，
        否则 `/health` 报的与工厂内部状态可能不一致（那正是"静默降级"）。
        """
        try:
            return get_embedder_status(self.embedder).mode
        except Exception:  # noqa: BLE001
            return getattr(self.embedder, "provider", "unknown")

    def health(self) -> dict[str, Any]:
        from knowflow.db.session import check_database

        db_info = check_database(self.settings)
        vector_info: dict[str, Any] = {}
        try:
            vector_info = self.vector_store.health()
        except Exception as exc:  # noqa: BLE001
            vector_info = {"error": f"{type(exc).__name__}: {exc}"[:200]}

        warnings = list(self.warnings)
        if self.settings.is_offline:
            warnings.append("未配置 OPENAI_API_KEY，当前为离线模式（抽取式回答）")
        if "fallback" in self.embedding_mode():
            warnings.append(f"向量化已降级：{self.embedding_mode()}")
        if self.bm25_rebuilt_chunks == 0:
            warnings.append("BM25 索引为空，关键词检索不可用（可能知识库为空或重建失败）")
        if not db_info.get("ok"):
            warnings.append(f"数据库不可用：{db_info.get('error')}")

        return {
            "status": "ok" if db_info.get("ok") else "degraded",
            "app": self.settings.app_name,
            "version": self.settings.app_version,
            "env": self.settings.env,
            "offline": self.settings.is_offline,
            "llm_mode": self.settings.llm_mode,
            "llm_model": self.settings.openai_model,
            "embedding_mode": self.embedding_mode(),
            "embedding_model": self.embedder.model,
            "embedding_dim": self.embedder.dim,
            "vector_backend": self.vector_store.backend,
            "vector": vector_info,
            "bm25_doc_count": self.bm25.doc_count,
            "bm25_rebuilt_chunks": self.bm25_rebuilt_chunks,
            "db": db_info,
            "agent_checkpoint_backend": self.settings.agent_checkpoint_backend,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ 请求级
    def build_request_services(self, session: Session) -> RequestServices:
        """为一次请求组装服务。**每次调用都返回全新实例**（见模块 docstring）。"""
        kb_service = KBService(
            session=session,
            vector_store=self.vector_store,
            bm25=self.bm25,
            embedder=self.embedder,
            settings=self.settings,
        )
        document_service = DocumentService(
            session=session,
            kb_service=kb_service,
            vector_store=self.vector_store,
            bm25=self.bm25,
            embedder=self.embedder,
            settings=self.settings,
        )

        # enricher 绑定到本请求的 session：检索命中后回 MySQL 取权威正文
        from knowflow.services.chunks import SqlChunkEnricher

        retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25=self.bm25,
            chat_model=self.judge_model,
            enricher=SqlChunkEnricher(session),
            settings=self.settings,
        )
        chat_service_placeholder: dict[str, ChatService] = {}

        def _kb_stats() -> dict[str, Any]:
            return chat_service_placeholder["svc"].kb_stats_for_tool(None)

        def _list_documents(keyword: str | None) -> list[dict[str, Any]]:
            return chat_service_placeholder["svc"].list_documents_for_tool(
                kb_id=None, keyword=keyword
            )

        tool_registry = ToolRegistry(
            retriever=retriever,
            kb_stats=_kb_stats,
            list_documents=_list_documents,
        )

        # 本次请求的 trace 持有者：显式传给节点（见 RequestServices.recorder_box 的说明）
        recorder_box = RecorderBox()

        deps = AgentDeps(
            settings=self.settings,
            chat_model=self.chat_model,
            judge_model=self.judge_model,
            retriever=retriever,
            tools=tool_registry,
            recorder_box=recorder_box,
        )
        agent_graph = build_agent_graph(deps, checkpointer=self.checkpointer)
        fixed_graph = build_fixed_chain(deps)
        runner = AgentRunner(
            agent_graph=agent_graph,
            fixed_graph=fixed_graph,
            settings=self.settings,
            model_name=self.chat_model.model,
        )

        chat_service = ChatService(
            session=session,
            retriever=retriever,
            runner=runner,
            kb_service=kb_service,
            document_service=document_service,
            chat_model=self.chat_model,
            settings=self.settings,
            recorder_box=recorder_box,
        )
        chat_service_placeholder["svc"] = chat_service

        return RequestServices(
            session=session,
            kb_service=kb_service,
            document_service=document_service,
            chat_service=chat_service,
            retriever=retriever,
            tool_registry=tool_registry,
            runner=runner,
            recorder_box=recorder_box,
        )

    # ------------------------------------------------------------------ 关闭
    def close(self) -> None:
        try:
            self.engine.dispose()
        except Exception as exc:  # noqa: BLE001
            logger.warning("container.engine_dispose_failed", error=str(exc)[:200])


def build_container(settings: Settings | None = None) -> Container:
    """装配全项目依赖。**启动期任何失败都要抛出去**，让进程起不来。"""
    started = time.perf_counter()
    cfg = settings or get_settings()

    cfg.apply_runtime_env()
    created = cfg.ensure_dirs()
    logger.info(
        "container.dirs",
        dirs=[str(p) for p in created],
        data_dir=str(cfg.data_dir),
    )

    engine = create_engine_from_settings(cfg)
    session_factory = get_session_factory(cfg)

    warnings: list[str] = []

    embedder = build_embedder(cfg)

    # 启动时预热向量化模型。**这不是性能优化，是"不让降级静默"**：
    # 本地模型是惰性加载的，如果不预热，`/health` 会一直报 `local`，
    # 直到第一次上传文档才暴露"模型加载失败、已降级成哈希向量"——
    # 而那时用户已经以为系统在正常工作。预热让降级在启动日志与 /health 里立刻可见。
    # 代价：启动多约 5~8 秒 + 模型常驻内存。
    if cfg.embedding_provider == "local":
        try:
            warmup_embedder(embedder)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "container.embedding_warmup_failed", error=f"{type(exc).__name__}: {exc}"[:200]
            )

    status = get_embedder_status(embedder)
    if status.degraded:
        warnings.append(f"向量化降级：{status.degrade_reason}")
        logger.warning("container.embedding_degraded", reason=str(status.degrade_reason)[:200])

    vector_store = build_vector_store(cfg)
    bm25 = BM25Registry(k1=1.5, b=0.75)
    chat_model = build_chat_model(cfg)
    judge_model = build_judge_model(cfg)
    checkpointer = build_checkpointer(cfg)

    # 启动时重建 BM25：不做这一步，关键词检索会**静默失效**
    bm25_rebuilt = 0
    try:
        with session_factory() as session:
            from knowflow.services.knowledge_base import KBService as _KBService

            service = _KBService(
                session=session,
                vector_store=vector_store,
                bm25=bm25,
                embedder=embedder,
                settings=cfg,
            )
            bm25_rebuilt = service.rebuild_bm25_index()
    except Exception as exc:  # noqa: BLE001
        # 数据库还没建表时（首次启动、迁移未跑）不该让服务起不来，
        # 但必须把这件事报出来 —— 否则就是又一次静默降级。
        message = f"BM25 索引重建失败（关键词检索当前不可用）：{type(exc).__name__}: {exc}"
        warnings.append(message)
        logger.warning("container.bm25_rebuild_failed", error=str(exc)[:300])

    container = Container(
        settings=cfg,
        engine=engine,
        session_factory=session_factory,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        chat_model=chat_model,
        judge_model=judge_model,
        checkpointer=checkpointer,
        startup_ms=int((time.perf_counter() - started) * 1000),
        warnings=warnings,
        bm25_rebuilt_chunks=bm25_rebuilt,
    )

    logger.info(
        "container.ready",
        startup_ms=container.startup_ms,
        offline=cfg.is_offline,
        embedding=container.embedding_mode(),
        vector_backend=vector_store.backend,
        bm25_chunks=bm25_rebuilt,
        warnings=len(warnings),
    )
    return container


__all__ = ["Container", "RequestServices", "build_container"]
