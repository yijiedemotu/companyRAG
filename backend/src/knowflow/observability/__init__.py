"""可观测层：把"这次请求慢在哪、花了多少、质量有没有退化"变成可查询的数据。

**一句话定位**：``observability`` 是第 ⑥ 层，负责「成本归因 + 延迟归因 + 质量趋势」，
它**不参与业务决策**，只观察业务。

**在链路中的位置**：

    ChatService / LangGraph 节点
      -> tracing.traced / TraceRecorder.span    （收集）
      -> tracing.persist                        （落 traces / trace_spans）
      -> metrics.collect_stats / collect_quality（聚合）
      -> api/routers/obs.py + /metrics          （出口）

**关键设计取舍**：

1. **可观测是旁路，失败绝不影响业务**。`persist()` 吞掉自己的所有异常，
   只记 error 日志。反过来说：**任何业务逻辑都不允许因为埋点而改变行为**。
2. **`ContextVar` + 装饰器实现零侵入**，业务代码不出现 `trace_id` 参数传递。
3. **价格与汇率集中在 `pricing.py`，用 `Decimal`**：`float` 累加会漂，
   而成本是要跟账单对的东西（见该模块注释）。
4. **聚合在 Python 里做时间分桶**，不用方言函数：MySQL/SQLite 两条路径一致，
   测试用 SQLite 才能真覆盖生产逻辑。

导出清单只放"业务代码会直接用的东西"；`metrics` 里的聚合函数要显式
``from knowflow.observability.metrics import collect_stats``，
免得 `observability` 顶层 import 就把 SQLAlchemy 的查询层拉进每个模块。
"""

from __future__ import annotations

from knowflow.observability.pricing import (
    Price,
    estimate_cost,
    format_cost,
    resolve_price,
    to_cny,
)
from knowflow.observability.tracing import (
    SpanContext,
    SpanRecord,
    TraceRecorder,
    current_recorder,
    get_recorder,
    persist,
    traced,
    use_recorder,
)

__all__ = [
    "Price",
    "SpanContext",
    "SpanRecord",
    "TraceRecorder",
    "current_recorder",
    "estimate_cost",
    "format_cost",
    "get_recorder",
    "persist",
    "resolve_price",
    "to_cny",
    "traced",
    "use_recorder",
]
