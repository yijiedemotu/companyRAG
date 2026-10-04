"""pytest 夹具：全部离线、零成本、可重复。

这个文件承担三件事，缺一样测试就跑不起来：

1. **把 ``backend/src`` 放进 ``sys.path``**：仓库没有装成可编辑包，测试直接从源码导入。
   放在 conftest 的**最顶部**（在 import knowflow 之前），否则导入会失败。
2. **关掉 Chroma 的匿名遥测**：``ANONYMIZED_TELEMETRY=False``。遥测失败本身不影响功能，
   但「全套测试零网络」是硬承诺，不能靠"它自己会静默失败"来实现。
3. **提供一套测试专用 Settings**：SQLite 内存库 + hash 向量化 + 内存向量库 + 空 API Key
   （空 Key ⇒ ``is_offline`` ⇒ 全部走 ``MockChatModel``）。**不读 ``.env``**（``_env_file=None``），
   否则别人本地的一份 ``backend/.env`` 就能让断言飘。

关于异步：``pytest-asyncio`` 装在 strict 模式，所以异步测试一律显式写
``@pytest.mark.asyncio``，不需要（也不应该）去改 ``pyproject.toml`` 的 asyncio_mode。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# 1. 源码路径（必须在任何 knowflow 导入之前）
# --------------------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --------------------------------------------------------------------------------------
# 2. 离线承诺：Chroma 的遥测必须显式关掉
# --------------------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pytest  # noqa: E402
from sqlalchemy import Engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

import knowflow.db.models  # noqa: E402,F401  必须先导入全部模型，Base.metadata 才是完整的
from knowflow.agent.graph import build_agent_graph, build_fixed_chain  # noqa: E402
from knowflow.agent.nodes import AgentDeps  # noqa: E402
from knowflow.agent.runner import AgentRunner  # noqa: E402
from knowflow.agent.tools import ToolRegistry  # noqa: E402
from knowflow.container import RequestServices  # noqa: E402
from knowflow.core.config import Settings  # noqa: E402
from knowflow.db.base import Base  # noqa: E402
from knowflow.db.models.identity import User  # noqa: E402
from knowflow.db.models.knowledge import Document, KnowledgeBase  # noqa: E402
from knowflow.db.session import create_engine_from_settings  # noqa: E402
from knowflow.embeddings.hash_embedder import HashEmbedder  # noqa: E402
from knowflow.llm.base import ChatModel  # noqa: E402
from knowflow.llm.mock import MockChatModel  # noqa: E402
from knowflow.retrieval.bm25 import BM25Registry  # noqa: E402
from knowflow.retrieval.engine import HybridRetriever  # noqa: E402
from knowflow.services.chat import ChatService  # noqa: E402
from knowflow.services.chunks import SqlChunkEnricher  # noqa: E402
from knowflow.services.document import DocumentService  # noqa: E402
from knowflow.services.knowledge_base import KBService  # noqa: E402
from knowflow.vectorstore.memory_store import InMemoryVectorStore  # noqa: E402

#: 测试里反复用到的一小段 Markdown（有标题、有数字、两小节）。
SAMPLE_MD = """# 员工报销制度

## 差旅报销标准

一线城市住宿标准为每晚 600 元，二线城市为每小时 400 元。住宿费需提供发票。

## 餐饮报销标准

餐饮补贴为每天 100 元，需要提供发票与审批单。
"""

SAMPLE_ERROR_CODE_MD = """# 错误码说明

PAYLOAD_TOO_LARGE 表示请求体超过服务端限制，需要拆分请求后重试。
"""


# ======================================================================================
# 配置与数据库
# ======================================================================================
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """一套测试专用配置：离线 + SQLite 内存 + hash 向量 + 内存向量库。

    - ``_env_file=None``：不读 ``backend/.env`` / 仓库根 ``.env``，测试结论只由这里的入参决定；
    - ``jwt_secret`` 给足 32 字节：pyjwt 对短 HMAC key 会打 ``InsecureKeyLengthWarning``，
      测试输出不需要这种噪声；
    - ``agent_checkpoint_backend="memory"``：不落 checkpointer 文件（SQLite saver 会建文件）。
    """
    return Settings(
        _env_file=None,
        env="test",
        debug=True,
        database_url="sqlite+pysqlite:///:memory:",
        data_dir=tmp_path,
        embedding_provider="hash",
        vector_backend="memory",
        openai_api_key="",
        agent_checkpoint_backend="memory",
        jwt_secret="test-secret-0123456789abcdef01234567",
        log_level="WARNING",
    )


@pytest.fixture
def db_engine(settings: Settings) -> Iterator[Engine]:
    """SQLite 内存引擎 + 建表。

    **复用 ``create_engine_from_settings``**：它已经处理了 ``:memory:`` 必须配 ``StaticPool``
    这件事（否则每个连接都是一个新空库，表现为"表不存在"）。
    """
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session_factory(db_engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=db_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """一个请求级会话（与 ``api/deps.get_db`` 同口径：只负责关，不负责 commit）。"""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


# ======================================================================================
# 组件层（全部离线实现）
# ======================================================================================
@pytest.fixture
def embedder(settings: Settings) -> HashEmbedder:
    return HashEmbedder(settings)


@pytest.fixture
def vector_store(settings: Settings) -> InMemoryVectorStore:
    return InMemoryVectorStore(settings)


@pytest.fixture
def bm25() -> BM25Registry:
    return BM25Registry()


@pytest.fixture
def chat_model(settings: Settings) -> MockChatModel:
    """主对话模型。离线 mock 会按 system prompt 扮演六种角色并返回合法 JSON，
    所以图上的每一条条件边都能被真实触发。"""
    return MockChatModel(settings)


@pytest.fixture
def judge_model(settings: Settings) -> MockChatModel:
    return MockChatModel(settings)


# ======================================================================================
# 服务层
# ======================================================================================
def build_services(
    settings: Settings,
    session: Session,
    *,
    embedder: Any,
    vector_store: Any,
    bm25: BM25Registry,
    chat_model: ChatModel,
    judge_model: ChatModel,
    checkpointer: Any | None = None,
) -> RequestServices:
    """按 ``Container.build_request_services`` 的同一套接线，手工装配一套服务。

    为什么不直接调 ``build_container()``：那会去连数据库、建目录、**预热向量模型**
    （离线 CI 不该加载 2GB 权重）。测试要的是同一套装配逻辑，但不是同一套 IO。
    """
    kb_service = KBService(
        session=session,
        vector_store=vector_store,
        bm25=bm25,
        embedder=embedder,
        settings=settings,
    )
    document_service = DocumentService(
        session=session,
        kb_service=kb_service,
        vector_store=vector_store,
        bm25=bm25,
        embedder=embedder,
        settings=settings,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        chat_model=judge_model,
        enricher=SqlChunkEnricher(session),
        settings=settings,
    )

    # 与容器一致：kb_stats / list_documents 两个工具要回指到 ChatService（避免循环依赖）
    placeholder: dict[str, ChatService] = {}

    def _kb_stats() -> dict[str, Any]:
        return placeholder["svc"].kb_stats_for_tool(None)

    def _list_documents(keyword: str | None) -> list[dict[str, Any]]:
        return placeholder["svc"].list_documents_for_tool(kb_id=None, keyword=keyword)

    tool_registry = ToolRegistry(
        retriever=retriever, kb_stats=_kb_stats, list_documents=_list_documents
    )
    deps = AgentDeps(
        settings=settings,
        chat_model=chat_model,
        judge_model=judge_model,
        retriever=retriever,
        tools=tool_registry,
    )
    runner = AgentRunner(
        agent_graph=build_agent_graph(deps, checkpointer=checkpointer),
        fixed_graph=build_fixed_chain(deps),
        settings=settings,
        model_name=chat_model.model,
    )
    chat_service = ChatService(
        session=session,
        retriever=retriever,
        runner=runner,
        kb_service=kb_service,
        document_service=document_service,
        chat_model=chat_model,
        settings=settings,
    )
    placeholder["svc"] = chat_service

    return RequestServices(
        session=session,
        kb_service=kb_service,
        document_service=document_service,
        chat_service=chat_service,
        retriever=retriever,
        tool_registry=tool_registry,
        runner=runner,
    )


@pytest.fixture
def services(
    settings: Settings,
    db_session: Session,
    embedder: HashEmbedder,
    vector_store: InMemoryVectorStore,
    bm25: BM25Registry,
    chat_model: MockChatModel,
    judge_model: MockChatModel,
) -> RequestServices:
    """一套完整的请求级服务（检索 / 工具 / 图 / 会话落库），跑在 SQLite 内存库上。"""
    return build_services(
        settings,
        db_session,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        chat_model=chat_model,
        judge_model=judge_model,
    )


@pytest.fixture
def user(services: RequestServices, db_session: Session) -> User:
    """第一个注册的用户（按契约自动是 admin）。"""
    registered, _token, _expires = _register(services)
    db_session.commit()
    return registered


def _register(services: RequestServices) -> tuple[User, str, int]:
    from knowflow.services.auth import AuthService

    auth = AuthService(session=services.session, settings=services.kb_service.settings)
    return auth.register(username="tester", password="passw0rd123", display_name="测试用户")


@pytest.fixture
def kb(services: RequestServices, db_session: Session, user: User) -> KnowledgeBase:
    kb = services.kb_service.create(name="测试知识库", description=None, owner_id=user.id)
    db_session.commit()
    return kb


@pytest.fixture
def ingest(services: RequestServices):
    """便捷入库：``ingest(kb_id, filename, content) -> Document``。"""

    def _ingest(kb_id: int, filename: str, content: str) -> Document:
        return services.document_service.ingest_upload(
            kb_id=kb_id, filename=filename, data=content.encode("utf-8")
        )

    return _ingest


@pytest.fixture
def kb_with_docs(kb: KnowledgeBase, ingest) -> KnowledgeBase:
    """已入库两份文档的知识库：一份制度（含住宿标准），一份错误码（专有名词）。"""
    ingest(kb.id, "员工报销制度.md", SAMPLE_MD)
    ingest(kb.id, "错误码说明.md", SAMPLE_ERROR_CODE_MD)
    return kb
