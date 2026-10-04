"""HTTP 层的依赖注入：容器、会话、当前用户、请求级服务。

**一句话定位**：把「进程级单例容器」与「请求级 session」缝合成路由函数能直接用的对象。

**在链路中的位置**：
``main.create_app()`` 的 lifespan -> ``app.state.container`` -> **本模块** -> 各路由函数。

**关键设计取舍**：

1. **容器从 `request.app.state` 取，而不是模块级全局变量**。
   全局变量在测试里没法替换（两个 `create_app()` 会互相踩），而
   `app.state` 天然一进程一实例、测试可以随便造。取不到就抛 `NotReadyError`（503）：
   "服务起来了但容器没装配好"是一种真实存在的状态，必须能被区分出来。
2. **每个请求新建一整套服务**（`container.build_request_services(session)`）。
   见 `container.py` 的 docstring：共享带 session 的服务会串号。
3. **`finally` 里必须 `close()` session**。FastAPI 的依赖生成器在响应**产生后**才收尾，
   漏掉 close 会耗尽连接池，表现为服务"越跑越慢然后全线 500"。
4. **不在这里 commit**。事务边界属于 service 层（一次问答要"写消息 + 写引用 + 写 trace"
   要么全成要么全败），依赖只负责"一定关闭"。
5. **不用 `OAuth2PasswordBearer`**：它会在 `/docs` 上渲染出一个 form 登录框，
   而本项目的登录接口是 JSON（`POST /auth/login`）。用错了会让文档误导调用方。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from knowflow.container import Container, RequestServices
from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import NotReadyError, UnauthorizedError
from knowflow.core.security import decode_access_token
from knowflow.db.models.identity import User
from knowflow.services.auth import AuthService

__all__ = [
    "AdminUser",
    "ContainerDep",
    "CurrentUser",
    "RequestId",
    "Services",
    "SessionDep",
    "SettingsDep",
    "bearer_token",
    "get_admin_user",
    "get_container",
    "get_current_user",
    "get_request_id",
    "get_services",
    "get_session",
    "get_settings_dep",
]

#: 认证头前缀。按 HTTP 头的惯例做大小写不敏感比较（`Bearer` / `bearer` 都要能用）。
_BEARER_PREFIX = "bearer"

_UNAUTHORIZED = "缺少或格式错误的 Authorization 头（应为 `Bearer <token>`）"


def get_settings_dep(request: Request) -> Settings:
    """配置。优先取容器里的那份（测试替换过的容器与 `get_settings()` 单例可能不同）。"""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is not None:
        return container.settings
    return get_settings()


def get_container(request: Request) -> Container:
    """取进程级容器。没有装配好就是 503，而不是 500 —— 两者的排障方向完全不同。"""
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:
        raise NotReadyError("容器尚未初始化（服务正在启动或启动失败）")
    return container


def get_session(container: ContainerDep) -> Iterator[Session]:
    """请求级 session。**只负责关闭**，不 commit（事务边界在 service 层）。"""
    session = container.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_services(session: SessionDep, container: ContainerDep) -> RequestServices:
    """组装本请求的服务集合。

    **每次请求都新建**：`build_request_services` 会把 session 绑进检索器、工具与图运行器，
    复用实例等于让不同用户共用一个数据库会话（见 `container.py` 的说明）。
    """
    return container.build_request_services(session)


def bearer_token(request: Request) -> str:
    """从 `Authorization: Bearer <token>` 里取 token。

    手写解析而不用 `OAuth2PasswordBearer`（见模块 docstring 第 5 条）。
    """
    header = request.headers.get("Authorization") or ""
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != _BEARER_PREFIX:
        raise UnauthorizedError(_UNAUTHORIZED)
    token = parts[1].strip()
    if not token:
        raise UnauthorizedError("Authorization 头中的 token 为空")
    return token


def get_current_user(
    token: Annotated[str, Depends(bearer_token)],
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    """解析 token 并加载用户。

    顺序是刻意的：**先验签（无状态）再查库** —— 伪造的 token 连一次数据库往返都省了。
    用户不存在 / 被禁用由 `AuthService.get_active_user` 抛 401 / 403。
    """
    payload = decode_access_token(token, settings=settings)
    subject = payload.get("sub")
    if subject is None:
        raise UnauthorizedError("身份凭证缺少 sub 字段")
    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise UnauthorizedError("身份凭证中的 sub 不是合法用户 ID") from exc
    return AuthService(session=session, settings=settings).get_active_user(user_id)


def get_admin_user(
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    """管理员专用依赖。权限判断只在 `AuthService.ensure_admin` 一处，避免散落。"""
    AuthService(session=session, settings=settings).ensure_admin(user)
    return user


def get_request_id(request: Request) -> str:
    """当前请求的 request_id（中间件写入 `request.state`）。

    取不到时兜底生成一个：SSE 产帧与异常处理器都要它，
    而"某条路径没经过中间件"（直接调用路由函数的测试）不该让响应里出现 `request_id: null`。
    """
    value = getattr(request.state, "request_id", None)
    return str(value) if value else uuid4().hex


# --------------------------------------------------------------------------------------
# 类型别名：路由函数签名里写 `services: Services` 就够了。
# 用 `Annotated[...]` 而不是 `= Depends(...)` 的默认值写法：mypy 能正确推导类型，
# 且 FastAPI 0.142 对两种写法都支持（本项目统一用 Annotated）。
# --------------------------------------------------------------------------------------
ContainerDep = Annotated[Container, Depends(get_container)]
SessionDep = Annotated[Session, Depends(get_session)]
Services = Annotated[RequestServices, Depends(get_services)]
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(get_admin_user)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
RequestId = Annotated[str, Depends(get_request_id)]
