"""应用入口：`create_app()` + lifespan + 统一错误信封 + 运维三件套。

**一句话定位**：全项目唯一的 ASGI 应用工厂，把中间件、异常处理器、路由按固定顺序装好。

**在链路中的位置**：
``uvicorn knowflow.main:app`` -> **本模块** -> `api/router.py` -> `api/routes/*` -> `services/`。

**关键设计取舍**（每一条都对应一个真实会踩的坑）：

1. **启动失败必须让进程退出**。lifespan 里**不 try/except 吞掉 `build_container` 的异常**：
   配置写错、数据库连不上、向量库目录建不出来，这些必须在启动期就把进程干掉，
   而不是"服务起来了，第一个请求才 500"。后者在容器化环境里表现为
   "pod 反复重启但日志里只有请求报错"，极难定位。
2. **注册顺序：`RequestContextMiddleware` → CORS → 异常处理器 → 子路由 → 运维三件套**。
   `RequestContextMiddleware` 必须最外层：它要给**所有**响应（含 CORS 预检、
   含异常处理器产出的响应）都带上 `X-Request-ID` 与 `X-Process-Time-Ms`。
3. **CORS 必须 `expose_headers=["X-Request-ID", "X-Process-Time-Ms"]`**。
   浏览器默认只把"安全列表"里的响应头暴露给 JS；不加这一行，
   前端 `response.headers.get('x-request-id')` 永远是 `null`，
   错误 toast 里就永远没有 request_id —— 而那是排障的唯一线索。**真实踩过的坑。**
4. **统一错误信封只有一个出口**（`_install_exception_handlers`）。
   所有非 2xx（包括 FastAPI 自己抛的 404/405、Pydantic 的 422）
   都变成 `{"error": {code, message, detail, request_id, timestamp}}`。
   前端只写一套解包逻辑，不需要为"框架默认的 422 形状"再写一套（见 `client.ts`）。
5. **兜底 `Exception` 处理器不能泄漏异常细节**。响应里只回一句通用文案 + request_id，
   完整 traceback 进日志（`exc_info=True`）。把 `str(exc)` 直接回给用户
   等于把文件路径、SQL 片段、依赖版本一起送给对方。
6. **SSE 路由里的异常不走这里**。`StreamingResponse` 开始之后响应头已经发出，
   异常处理器不可能再改状态码；所以 SSE 的异常必须在生成器内部转成 `error` 帧
   （见 `api/routes/chat.py` 的 `_sse_frames`）。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from knowflow.api.router import api_router, include_ops
from knowflow.container import build_container
from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import KnowFlowError
from knowflow.core.logging import configure_logging, get_logger
from knowflow.middleware.context import RequestContextMiddleware

__all__ = ["app", "create_app"]

logger = get_logger(__name__)


def _now_iso() -> str:
    """错误信封里的 `timestamp`：UTC ISO8601 带 `Z`，精度到毫秒。

    格式必须与 `core/exceptions.py` 的 `KnowFlowError.to_dict()` 逐字一致——
    否则"业务异常"与"框架异常"两条路径给出的时间格式不同，
    前端解析错误时对会拿到 `Invalid Date`（只在部分错误上出现，最难发现）。
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


#: Starlette/FastAPI 自带 HTTPException 的状态码 -> 契约错误码（第二节）。
#: 不在表里的状态码用 `HTTP_<status>` 兜底：**稳定且可判断**，
#: 比统一塞一个 `NOT_FOUND` 强（405 被报成 404 会让人以为是路由没注册）。
_STATUS_CODES: dict[int, str] = {
    400: "VALIDATION_ERROR",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "FILE_TOO_LARGE",
    415: "UNSUPPORTED_FILE_TYPE",
    422: "REQUEST_VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    502: "UPSTREAM_ERROR",
    503: "NOT_READY",
}


def _request_id_of(request: Request) -> str:
    """取 request_id；中间件没跑到时（理论上不该发生）现生成一个。

    异常处理器**不能**因为"拿不到 request_id"而再抛异常——
    那会把一个 404 变成 500，而且真正的错误信息全丢了。
    """
    value = getattr(request.state, "request_id", None)
    return str(value) if value else uuid4().hex


def _envelope(
    *,
    code: str,
    message: str,
    status_code: int,
    request_id: str,
    detail: Any = None,
) -> JSONResponse:
    """构造统一错误信封。所有异常处理器共用这一个出口。"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "detail": detail,
                "request_id": request_id,
                "timestamp": _now_iso(),
            }
        },
    )


def _install_exception_handlers(app: FastAPI) -> None:
    """注册四个异常处理器（顺序无关，类型匹配由 FastAPI 自己做）。"""

    @app.exception_handler(KnowFlowError)
    async def _knowflow_error(_request: Request, exc: KnowFlowError) -> JSONResponse:
        """业务异常：`code` / `http_status` 都来自异常类，不再做映射。

        `to_dict()` 已经保证字段与契约一致（连 `timestamp` 都在里面），
        这里只负责把状态码和 request_id 补上。
        """
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(request_id=_request_id_of(_request)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Pydantic 422 -> 统一信封（契约第二节要求改写成统一形状）。

        `exc.errors()` 里可能带 `ValueError` 对象（自定义校验器抛的），
        直接塞进 JSONResponse 会在序列化时抛 `TypeError`（"Object of type ValueError
        is not JSON serializable"），于是 422 变成 500。
        所以先过一遍 `jsonable_encoder`，再用 `str` 兜底。
        """
        return _envelope(
            code="REQUEST_VALIDATION_ERROR",
            message="请求参数校验失败",
            status_code=422,
            request_id=_request_id_of(request),
            detail=jsonable_encoder(exc.errors(), custom_encoder={Exception: str}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """框架自带 HTTPException（含 FastAPI 的 404/405）-> 统一信封。"""
        status_code = int(exc.status_code)
        code = _STATUS_CODES.get(status_code, f"HTTP_{status_code}")
        detail = exc.detail
        # `detail` 是字符串时它就是给人看的 message；是 dict 时放 detail 里。
        if isinstance(detail, str):
            message = detail
            extra: Any = None
        else:
            message = str(code)
            extra = jsonable_encoder(detail, custom_encoder={Exception: str})
        return _envelope(
            code=code,
            message=message,
            status_code=status_code,
            request_id=_request_id_of(request),
            detail=extra,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """兜底：记完整 traceback，响应里只回通用文案（见模块 docstring 第 5 条）。"""
        request_id = _request_id_of(request)
        logger.error(
            "http.unhandled_exception",
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            error=f"{type(exc).__name__}: {exc}"[:500],
            exc_info=True,
        )
        return _envelope(
            code="INTERNAL_ERROR",
            message="服务内部错误，请稍后重试",
            status_code=500,
            request_id=request_id,
        )


def _install_cors(app: FastAPI, settings: Settings) -> None:
    """CORS。`expose_headers` 是核心（见模块 docstring 第 3 条）。"""
    origins = list(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )
    logger.debug("app.cors_installed", origins=origins)


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂。

    **注意 lifespan 挂在哪个 app 上**：`build_container()` 建的是进程级单例，
    所以同一进程里创建多个 app 会共享重组件（模型、向量库连接）——
    这在测试里是优点（不用重复加载模型），但也意味着测试之间要自己清状态
    （见 `db/session.py` 的 `reset_engine_state`）。
    """
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动装配 / 关闭释放。**启动期不吞异常**（见模块 docstring 第 1 条）。"""
        configure_logging(level=cfg.log_level, json_output=cfg.log_json)
        logger.info("app.startup", **cfg.public_snapshot())

        container = build_container(cfg)
        application.state.container = container
        application.state.started_at = time.time()
        logger.info(
            "app.ready",
            startup_ms=container.startup_ms,
            routes=len(application.routes),
        )
        try:
            yield
        finally:
            container.close()
            logger.info("app.shutdown")

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "KnowFlow 知识库问答系统。接口契约见 docs/01-数据库与接口契约.md 第五节；"
            "所有非 2xx 响应均为统一错误信封 {error:{code,message,detail,request_id,timestamp}}。"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=cfg.debug,
        lifespan=lifespan,
    )

    # ① 请求上下文：必须最外层（要给所有响应写 X-Request-ID / X-Process-Time-Ms）
    app.add_middleware(RequestContextMiddleware)
    # ② CORS
    _install_cors(app, cfg)
    # ③ 异常处理器
    _install_exception_handlers(app)
    # ④ 业务子路由（挂到 api_prefix 下）
    app.include_router(api_router, prefix=cfg.api_prefix)
    # ⑤ 运维三件套（根路径、不鉴权）
    include_ops(app)

    return app


#: `uvicorn knowflow.main:app` 的默认入口。
#:
#: **模块级就 `create_app()`**（而不是在 lifespan 里懒加载）：
#: 这样 `openapi.json` / `/docs` 在没有 lifespan 的场景（静态分析、
#: OpenAPI 快照测试）下也是完整的。真正的重组件仍然只在 lifespan 里装配。
app = create_app()
