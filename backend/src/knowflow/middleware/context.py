"""请求上下文中间件：request_id 贯穿、耗时统计、访问日志。

# 为什么不用 `BaseHTTPMiddleware`

这是本项目一个真实的技术决策点，不是风格偏好：

`starlette.middleware.base.BaseHTTPMiddleware` 用「一个任务跑下游 app + 一个内存
对象管道传消息」的方式实现。它把下游的 `send` 换成自己包装的版本，并通过
`anyio` 的内存对象流把响应消息回传给真正写出的那个任务。对流式响应
（`StreamingResponse` / `text/event-stream`）来说，这条路径会引入**缓冲**：
下游 `await send(...)` 之后消息先落进管道，由另一个任务再取出写出去。
实践中表现为 SSE 的帧被攒成一批才发出，"边生成边推"的效果失效——用户看到的
是转圈很久然后一次性出现整段答案，而不是逐字出现。这正是 `/chat/stream`
最核心的体验，不能有。

纯 ASGI 中间件没有这一层：它就是一个 `async def __call__(self, scope, receive, send)`，
把 `send` 换成自己的一层薄包装**同一协程内同步调用**，`await send(message)`
返回时数据已经交给服务器。零缓冲、零额外任务、零额外队列。

代价是纯 ASGI 中间件看不到「解析好的」请求对象（没有 `request.state` 的现成载体），
所以要自己处理 scope 里的 headers / client / path。下面的代码就是为了这个代价。

# 三件必须做对的事

1. **request_id 进 `request.state`**：鉴权依赖、异常处理器、SSE 产帧都需要它
   （错误信封里的 `request_id` 就是从这儿取的）。写进 `scope["state"]` 后，
   FastAPI 的 `Request.state` 会读到同一个 dict，业务代码写 `request.state.request_id`。
2. **`finally` 里必须 `clear_request_context()`**。contextvar 是**线程内共享**的，
   而 ASGI 服务器（uvicorn）用线程池或复用协程跑不同请求；不清理就会让
   **下一个请求的日志带上一个请求的 request_id**——日志里两个不同请求混成一条链，
   排查时会一路查到一个完全无关的用户身上。这类 bug 不会报错，只会让人得出错误结论，
   所以宁可多写一行也不能省。
3. **不在这里捕获业务异常**。异常到 HTTP 响应的映射统一在 `main.py` 的
   `exception_handler` 里做（`KnowFlowError` → 统一错误信封），中间件只负责
   "无论成不成都记日志、都清理上下文"，然后把原异常 `raise` 出去。
   在中间件里 catch 一次，就等于把状态码映射表复制了第二份。
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from knowflow.core.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)

__all__ = [
    "PROCESS_TIME_HEADER",
    "REQUEST_ID_HEADER",
    "SKIP_ACCESS_LOG_PATHS",
    "RequestContextMiddleware",
    "is_access_loggable",
]

logger = get_logger(__name__)

#: 入站请求可自带、出站响应必然回显的请求 ID 头。
REQUEST_ID_HEADER = "X-Request-ID"
#: 响应上回显的耗时头，单位毫秒，保留 2 位小数。
PROCESS_TIME_HEADER = "X-Process-Time-Ms"

#: 不记访问日志的路径前缀：容器探针默认几十秒打一次，K8s 多副本下
#: `/health` 能在几小时内刷出百万行日志，把真正有用的日志冲掉。
SKIP_ACCESS_LOG_PATHS: frozenset[str] = frozenset({"/health", "/ready", "/metrics"})

#: 不记访问日志的静态资源后缀。前端产物、图标、文档页的静态文件同理：
#: 一次页面加载就是几十条请求，且它们对排障毫无价值。
SKIP_ACCESS_LOG_SUFFIXES: tuple[str, ...] = (
    ".css",
    ".js",
    ".mjs",
    ".map",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
)


def is_access_loggable(path: str) -> bool:
    """这条路径要不要写访问日志。

    单独抽成函数是为了能被测试直接调——把判断散在中间件体内的 `if` 里，
    就只能靠"跑一遍看日志"来验证，探针路径一旦被漏掉也不会有人发现。
    """
    normalized = path if path.startswith("/") else f"/{path}"
    if normalized.rstrip("/") in SKIP_ACCESS_LOG_PATHS or normalized in SKIP_ACCESS_LOG_PATHS:
        return False
    lowered = normalized.lower()
    return not lowered.endswith(SKIP_ACCESS_LOG_SUFFIXES)


class RequestContextMiddleware:
    """给每个 HTTP 请求绑定 request_id、统计耗时并记一条访问日志。

    用法（`main.py`）：

        app.add_middleware(RequestContextMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # 非 http（lifespan / websocket）直接透传：
        # lifespan 没有请求概念，websocket 有自己的生命周期与日志方式，
        # 在这里统一处理只会让语义变模糊（而且 websocket 没有响应头可写）。
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._resolve_request_id(scope)
        # 写进 scope["state"]，FastAPI 的 Request.state 读的就是这个 dict。
        # 用 setdefault 而不是赋值：如果外层还挂了别的中间件已经放过 key，别覆盖它。
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id

        path = str(scope.get("path", ""))
        method = str(scope.get("method", ""))
        client = self._client_host(scope)
        log_access = is_access_loggable(path)

        started = time.perf_counter()
        status_code = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                # 用 MutableHeaders 在原有头上追加，而不是重建 headers：
                # 覆盖式写入会把下游设置的 Content-Type / CORS 头一起丢掉。
                # 传 `raw=` 而不是 `scope=`：`raw=` 原地改列表，
                # 不会把 `message["headers"]` 换成一个新对象（下游可能持有旧引用）。
                headers = MutableHeaders(raw=message["headers"])
                headers[REQUEST_ID_HEADER] = request_id
                headers[PROCESS_TIME_HEADER] = f"{(time.perf_counter() - started) * 1000:.2f}"
            await send(message)

        bind_request_context(request_id)
        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException:
            # 不在这里做异常到响应的映射（那是 main.py 的 exception_handler 的职责）。
            # 只补一个状态码用于日志：走到这里说明响应头还没发出去，
            # 对客户端而言必然是 5xx，日志里记 200 会掩盖故障。
            if status_code == 0:
                status_code = 500
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            if log_access:
                # 结构化字段而不是拼字符串：生产用 LOG_JSON=true 时这些字段
                # 直接变成可查询的 JSON 键，能按 duration_ms 排序找出最慢的接口。
                logger.info(
                    "http.request",
                    method=method,
                    path=path,
                    status=status_code,
                    duration_ms=duration_ms,
                    client=client,
                )
            # 必须清理：contextvar 在线程/协程复用时会带到下一个请求。
            # 放在 finally 里保证异常路径也执行——异常时留下的脏 request_id
            # 恰恰是最难查的（报错日志里挂着别人的 request_id）。
            clear_request_context()

    @staticmethod
    def _resolve_request_id(scope: Scope) -> str:
        """取 `X-Request-ID`，没有就生成一个 32 位 hex。

        用 `uuid4().hex` 而不是带横线的 `str(uuid4())`：日志里少 4 个字符、
        更容易双击复制整串，也不会有中划线被换行截断的问题。
        """
        raw = Headers(scope=scope).get(REQUEST_ID_HEADER)
        return raw.strip() if raw and raw.strip() else uuid4().hex

    @staticmethod
    def _client_host(scope: Scope) -> str:
        """客户端标识，形如 `127.0.0.1:53124`。

        不解析 `X-Forwarded-For`：本项目默认直接监听，代理头由客户端可伪造，
        在没有明确的反代部署前采信它只会让日志里的来源变成假的。
        """
        client = scope.get("client")
        if not client:
            return "-"
        host = str(client[0])
        port = client[1] if len(client) > 1 else None
        return f"{host}:{port}" if port is not None else host
