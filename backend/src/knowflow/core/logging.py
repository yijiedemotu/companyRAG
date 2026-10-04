"""structlog 日志配置。

要解决的三件事：

1. **一次请求的所有日志能被串起来**：用 `contextvars` 绑定 `request_id`，
   中间件一开始就 `bind_contextvars(request_id=...)`，之后所有日志自动带上它，
   业务代码一行都不用改。这就是"全链路追踪"最便宜的实现方式。
2. **本地好看、生产好解析**：开发用彩色 console，生产用 JSON（可直接进 ELK/Loki）。
   切换只由一个 `LOG_JSON` 决定。
3. **和标准库 logging 打通**：uvicorn / sqlalchemy 用的是标准库 logging，
   它们不打 structlog 的 PrintLogger 就会两套格式并存。
   所以用 `ProcessorFormatter` 把标准库日志也导流进同一个渲染管线。
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# 请求级上下文：request_id 与 trace_id 在中间件里绑定，之后全流程可读
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

_configured = False


def _add_context(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """把 contextvar 里的 request_id / trace_id 注入每条日志。

    没用 `structlog.contextvars.merge_contextvars` 是因为这里还需要兜底：
    某些后台任务（评测线程、启动过程）没有经过中间件，contextvar 是空的，
    这时不应该往日志里塞 `request_id=None`（会污染 JSON 结构）。
    """
    if (rid := request_id_var.get()) and "request_id" not in event_dict:
        event_dict["request_id"] = rid
    if (tid := trace_id_var.get()) and "trace_id" not in event_dict:
        event_dict["trace_id"] = tid
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """进程级初始化（幂等）。在 `main.py` 与脚本入口各调一次。"""
    global _configured

    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_output
        else structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
            # 把关键字段排到前面，扫日志时一眼能看到谁在报错
            sort_keys=False,
        )
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(log_level)

    # uvicorn 自带的 access 日志会和我们的中间件访问日志重复，关掉它
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(max(log_level, logging.WARNING))
    # SQLAlchemy 的 echo 由配置控制，这里只保证不被 root 级别淹没
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> Any:
    """业务代码统一用这个取 logger（返回 structlog 的 BoundLogger）。"""
    if not _configured:  # 脚本直接 import 时兜底，避免日志丢到默认 handler
        configure_logging()
    return structlog.get_logger(name)


def bind_request_context(request_id: str, trace_id: str | None = None) -> None:
    request_id_var.set(request_id)
    if trace_id is not None:
        trace_id_var.set(trace_id)


def clear_request_context() -> None:
    request_id_var.set(None)
    trace_id_var.set(None)


__all__ = [
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "request_id_var",
    "trace_id_var",
]
