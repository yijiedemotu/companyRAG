"""HTTP 接口层：路由、依赖注入与统一错误信封的挂载点。

**一句话定位**：把 `services/` 的业务能力暴露成契约文档第五节的 REST 接口。

**在链路中的位置**：
``main.create_app()`` -> `api/router.py` -> 各 `api/routes/*.py` -> `services/` -> 数据库/向量库。

**关键设计取舍**：

1. **路由层不写 SQLAlchemy 查询**（唯一例外见下）。接口层只做三件事：
   解析请求（schema）、调用 service、把结果序列化成响应。
   一旦路由里出现 `select()`，"同一段业务被两个接口各写一遍"就只是时间问题。
2. **两处刻意的例外**：`api/routes/observability.py` 与 `api/routes/conversations.py`
   里有只读的 `select()`。理由写在各自文件顶部：前者是**纯聚合查询**（没有业务规则，
   放 service 只会多一层转发），后者的 JOIN 只服务于**响应形状补齐**（给 citation 补
   `doc_name` / `page_no` / `section_path`），不属于任何业务用例。
   两处都只读、都不写库。
3. **统一错误信封在 `main.py` 注册**，不在这里。路由只管抛 `KnowFlowError` 子类，
   错误码到 HTTP 状态的映射表只有一份（`core/exceptions.py`）。
"""

from __future__ import annotations

__all__: list[str] = []
