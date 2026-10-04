"""数据库引擎与会话。

三个必须处理的现实问题：

1. **MySQL 连接会"悄悄死掉"**：默认 `wait_timeout` 8 小时，隔夜后连接池里的
   连接已被服务端关闭，第一个请求会拿到 `Lost connection`。
   `pool_pre_ping=True`（借出前 ping 一次）是标准解法，代价是每次多一个往返。
2. **测试要用 SQLite 且要快**：内存 SQLite 必须用 `StaticPool`，
   否则每个连接都是一个新的空库（"表不存在"的经典翻车点）。
   同时 SQLite 默认禁止跨线程使用连接，FastAPI 的线程池会踩到，要关掉检查。
3. **会话必须显式关闭**：不关会耗尽连接池，表现为服务"越跑越慢然后全线 500"。
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def create_engine_from_settings(settings: Settings) -> Engine:
    """按配置创建引擎（不做全局缓存，测试要能造多个）。"""
    url = settings.database_url
    kwargs: dict[str, Any] = {"echo": settings.db_echo, "future": True}

    if _is_sqlite(url):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url.endswith("sqlite://"):
            # 内存库必须共享同一个连接，否则每次 connect 都是一个新空库
            kwargs["poolclass"] = StaticPool
    else:
        kwargs.update(
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_recycle=settings.db_pool_recycle,
        )

    engine = create_engine(url, **kwargs)

    if not _is_sqlite(url):
        _install_mysql_session_settings(engine)
    return engine


def _install_mysql_session_settings(engine: Engine) -> None:
    """把 MySQL 会话设成严格模式 + utf8mb4。

    - `STRICT_TRANS_TABLES`：类型不符/超长直接报错而不是静默截断
      （MySQL 非严格模式下会"帮你"截断字符串，数据悄悄坏掉）；
    - `utf8mb4`：能存 emoji 与生僻字（`utf8` 在 MySQL 里其实只有 3 字节）。
    """

    @event.listens_for(engine, "connect")
    def _set_session(  # pragma: no cover - 需要真实 MySQL 才能触发
        dbapi_connection: Any, _connection_record: Any
    ) -> None:
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET SESSION sql_mode='STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION'")
            cursor.execute("SET NAMES utf8mb4")


def get_engine(settings: Settings | None = None) -> Engine:
    """进程级单例引擎。"""
    global _engine
    if _engine is None:
        cfg = settings or get_settings()
        _engine = create_engine_from_settings(cfg)
        logger.debug("db.engine.created", dialect=_engine.dialect.name)
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(settings),
            autoflush=False,  # 查询前不自动 flush，避免"查着查着把半成品写进库"
            autocommit=False,
            expire_on_commit=False,  # commit 后对象仍可读，否则序列化响应会触发额外查询
            future=True,
        )
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """给脚本/后台任务用的会话上下文：自动 commit，异常自动 rollback。"""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个会话。

    注意 **不在这里 commit**：事务边界由 service 层决定（一个请求可能要做
    "写消息 + 写引用 + 写 trace" 这一组要么全成要么全败的操作）。
    依赖只负责"一定关闭"。
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def check_database(settings: Settings | None = None) -> dict[str, Any]:
    """`/health` 用的数据库探活。绝不抛异常——探活失败本身就是返回值。"""
    cfg = settings or get_settings()
    info: dict[str, Any] = {"ok": False, "dialect": None, "version": None, "error": None}
    try:
        engine = get_engine(cfg)
        info["dialect"] = engine.dialect.name
        with engine.connect() as conn:
            row = conn.execute(text("SELECT VERSION()")).scalar()
            info["version"] = str(row) if row is not None else None
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 - 探活失败要如实上报，不能让它 500
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return info


def reset_engine_state() -> None:
    """测试用：清掉单例，让下一处调用重新按新配置建引擎。"""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


__all__ = [
    "check_database",
    "create_engine_from_settings",
    "get_db",
    "get_engine",
    "get_session_factory",
    "reset_engine_state",
    "session_scope",
]
