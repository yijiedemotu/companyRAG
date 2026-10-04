"""中间件包。

只放**与业务无关的通用横切逻辑**。业务规则（鉴权、权限、限流）不在这里，
它们需要知道路由与依赖，用 FastAPI 的 `Depends` 表达更清楚。
"""

from __future__ import annotations

from knowflow.middleware.context import RequestContextMiddleware

__all__ = ["RequestContextMiddleware"]
