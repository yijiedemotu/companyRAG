"""可观测聚合：把 `traces` / `trace_spans` 变成面板上的数字。

**一句话定位**：只读聚合层——输入一个时间窗口，输出 ``/obs/stats``、``/obs/quality``
与 ``/metrics``（Prometheus 文本）需要的全部数字。

**在链路中的位置**：
``tracing.persist()`` 写库 -> **本模块聚合** -> ``api/routers/obs.py`` -> 前端 ECharts。

**关键设计取舍**：

1. **分位数用线性插值（numpy 的 linear 方法），不用最近秩法**。
   本项目的评测集与流量都小（一次评测 28 条），最近秩法在 10 个样本时
   ``P95`` 会**恒等于最大值**：用户看到"P95 = 最大值"会以为是 bug，
   而且小样本下它没有任何统计意义。线性插值至少随样本连续变化。
2. **不用数据库方言函数做时间分桶**（不用 MySQL 的 ``DATE_FORMAT``、也不用
   SQLite 的 ``strftime``）：那样代码要写两套分支，而且容易在切库时静默出错。
   本项目一个窗口最多几千行，**拉到 Python 里分桶**，跨库一致、可单测，
   代价是窗口内数据量很大时内存上升（真有那天再改成按需下推下推到 SQL）。
3. **`by_model` 必须跨表（`traces` × `messages`）**：`traces` 表**没有 model 列**
   （一次请求可能调用多个模型，一列放不下），所以"按模型看 token 与成本"只能从
   `messages` 取；而"请求数/失败数"只有 `traces` 有。两边按 model 名字**外连接式合并**，
   两个口径都标注清楚，避免前端把 `traces` 的请求数当成"该模型的请求数"（会偏大，
   因为一次 agent 请求可能产生多条 message）。
4. **时间序列补齐空缺小时**：前端折线图按点顺序画 x 轴，缺小时会让 x 轴"跳"，
   看起来像"请求量突然暴增"。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from knowflow.core.config import Settings
from knowflow.core.logging import get_logger
from knowflow.db.models.chat import ROLE_ASSISTANT, Message
from knowflow.db.models.observability import (
    STATUS_OK,
    Trace,
    TraceSpan,
)
from knowflow.db.types import utcnow

# 从子模块而不是包顶层导入：避免"包 __init__ 尚未执行完"时导入本模块造成的
# 部分初始化循环（`from knowflow.observability import pricing` 依赖 __init__ 的进度，
# 子模块直连没有这个隐含顺序依赖）。
from knowflow.observability import pricing as pricing_mod

logger = get_logger(__name__)

#: 分位数口径：前端固定按这五个点画图，指标名也固定（Prometheus 标签不能动态生成）。
_LATENCY_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p50", 50.0),
    ("p95", 95.0),
    ("p99", 99.0),
)

#: 时间序列的点数上限 = 窗口小时数（每小时一个点）。
#: 窗口过大时（例如 24*30）会把响应撑大，调用方应自己限制 `hours`。
_MAX_SERIES_HOURS = 24 * 31

#: `reflect` span 的名字子串：判定"这次请求是否走了反思"。
_REFLECT_HINTS = ("reflect", "反思")


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（与 ``numpy.percentile(..., method="linear")`` 同口径）。

    空序列返回 ``0.0``（不是抛异常也不是 NaN）：调用方是 API 层，
    "没有数据"应该渲染成 0 而不是 500。

    `p` 允许 0~100；超出范围会被夹住，避免插值下标越界。
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    ratio = min(max(float(p), 0.0), 100.0) / 100.0
    position = ratio * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    # 线性插值：p50 对 [1..100] 得到 50.5（不是 50）——这就是与最近秩法的区别
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def aggregate_latency(values: Sequence[int]) -> dict[str, float]:
    """一次算出 p50/p95/p99/max/avg，空输入全 0。"""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "avg": 0.0, "count": 0.0}
    floats = [float(v) for v in values]
    result = {name: round(percentile(floats, point), 2) for name, point in _LATENCY_PERCENTILES}
    result["max"] = round(max(floats), 2)
    result["avg"] = round(sum(floats) / len(floats), 2)
    result["count"] = float(len(floats))
    return result


def _to_float(value: Any) -> float:
    """把 `Decimal` / `None` 统一成 float（JSON 里 Decimal 无法直接序列化）。"""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _window_start(hours: int) -> datetime:
    """窗口起点（naive UTC，与库里的 `DATETIME(6)` 约定一致）。"""
    span = min(max(int(hours), 1), _MAX_SERIES_HOURS)
    return utcnow() - timedelta(hours=span)


def _hour_key(moment: datetime) -> datetime:
    """截断到整点。

    库里存的是 naive UTC（见 `db/types.py`），但保险起见先去掉 tzinfo：
    如果哪天有人写入带时区的 datetime，直接用 `.replace(minute=0)` 会保留时区，
    与窗口起点的 naive 值比较时会抛 `TypeError`，那时才查就太晚了。
    """
    if moment.tzinfo is not None:
        moment = moment.replace(tzinfo=None)
    return moment.replace(minute=0, second=0, microsecond=0)


def collect_stats(session: Session, *, hours: int, settings: Settings) -> dict[str, Any]:
    """聚合时间窗口内的请求量 / token / 成本 / 延迟 / 分组 / 时间序列。

    返回结构与 `docs/01-数据库与接口契约.md` 5.9 的 `ObsStatsOut` 对齐
    （字段名用 snake_case，由 schema 层决定是否改名）。
    """
    start = _window_start(hours)
    span_hours = min(max(int(hours), 1), _MAX_SERIES_HOURS)

    trace_rows = session.execute(
        select(
            Trace.trace_id,
            Trace.mode,
            Trace.status,
            Trace.latency_ms,
            Trace.prompt_tokens,
            Trace.completion_tokens,
            Trace.cost_usd,
            Trace.llm_calls,
            Trace.created_at,
        ).where(Trace.created_at >= start)
    ).all()

    # 预填窗口内每个小时的空桶（下面按 created_at 落点累加），
    # 这样即使某小时完全没有请求，时间序列里也有这个点（见模块 docstring 第 4 条）
    hourly: dict[datetime, dict[str, float]] = {}
    for offset in range(span_hours + 1):
        hourly[_hour_key(start) + timedelta(hours=offset)] = {
            "requests": 0.0,
            "errors": 0.0,
            "tokens": 0.0,
        }

    requests = len(trace_rows)
    errors = 0
    prompt_tokens = 0
    completion_tokens = 0
    llm_calls_total = 0
    cost_usd = Decimal(0)
    latency_values: list[int] = []
    by_mode: dict[str, dict[str, float]] = {}

    for row in trace_rows:
        _trace_id, mode, status, latency, p_tokens, c_tokens, cost, llm_calls, created_at = row
        latency = int(latency or 0)
        p_tokens = int(p_tokens or 0)
        c_tokens = int(c_tokens or 0)
        llm_calls = int(llm_calls or 0)
        latency_values.append(latency)
        prompt_tokens += p_tokens
        completion_tokens += c_tokens
        llm_calls_total += llm_calls
        cost_usd += Decimal(str(cost if cost is not None else 0))

        is_error = status != STATUS_OK
        if is_error:
            errors += 1

        bucket_key = mode or "unknown"
        bucket = by_mode.setdefault(
            bucket_key,
            {
                "requests": 0.0,
                "errors": 0.0,
                "tokens": 0.0,
                "llm_calls": 0.0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
            },
        )
        bucket["requests"] += 1
        bucket["errors"] += 1 if is_error else 0
        bucket["tokens"] += p_tokens + c_tokens
        bucket["llm_calls"] += llm_calls
        bucket["cost_usd"] += _to_float(cost)
        bucket["latency_ms"] += latency

        hour = _hour_key(created_at) if created_at is not None else None
        if hour is not None and hour in hourly:
            hourly[hour]["requests"] += 1
            hourly[hour]["errors"] += 1 if is_error else 0
            hourly[hour]["tokens"] += p_tokens + c_tokens

    for bucket in by_mode.values():
        count = bucket["requests"] or 1.0
        bucket["avg_latency_ms"] = round(bucket.pop("latency_ms") / count, 2)
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        bucket["tokens"] = int(bucket["tokens"])
        bucket["llm_calls"] = int(bucket["llm_calls"])

    latency = aggregate_latency(latency_values)
    cost_cny = pricing_mod.to_cny(cost_usd, settings)

    return {
        "hours": span_hours,
        "window_start": start.isoformat(timespec="seconds") + "Z",
        "requests": requests,
        "success": requests - errors,
        "errors": errors,
        "error_rate": round(errors / requests, 4) if requests else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "avg_tokens_per_request": round((prompt_tokens + completion_tokens) / requests, 2)
        if requests
        else 0.0,
        "cost_usd": round(_to_float(cost_usd), 6),
        "cost_cny": round(_to_float(cost_cny), 4),
        "llm_calls": llm_calls_total,
        "latency_ms": latency,
        "by_mode": by_mode,
        "by_model": _by_model(session, start=start),
        "series": _series(hourly),
    }


def _by_model(session: Session, *, start: datetime) -> list[dict[str, Any]]:
    """按模型聚合 token 与成本。

    **为什么必须跨 `messages` 表**：`traces` 表没有 model 列——一次 agent 请求
    可能先调便宜的小模型做 grade/reflect、再调大模型生成，"一个请求一个模型"
    这个假设本身不成立。真正带 model 字段的是 `messages`（assistant 消息记下
    生成它的模型）。所以：

    - token / 成本 / 消息条数：从 `messages` 按 model 聚合（**权威口径**）；
    - 请求数：`traces` 没有模型信息，这里不给出"每模型请求数"，
      只给 `traces.total`，避免前端把 messages 条数误当请求数。

    只统计 assistant 角色：user 消息没有 model，计入会把行数翻倍。
    """
    rows = session.execute(
        select(
            Message.model,
            func.count(Message.id),
            func.coalesce(func.sum(Message.prompt_tokens), 0),
            func.coalesce(func.sum(Message.completion_tokens), 0),
            func.coalesce(func.sum(Message.cost_usd), 0),
        )
        .where(Message.created_at >= start, Message.role == ROLE_ASSISTANT)
        .group_by(Message.model)
    ).all()

    total_messages = sum(int(row[1] or 0) for row in rows)
    out: list[dict[str, Any]] = []
    for model, count, p_tokens, c_tokens, cost in rows:
        messages = int(count or 0)
        out.append(
            {
                "model": str(model or "unknown"),
                "messages": messages,
                # 占比按 assistant 消息数算：这就是"这个模型承担了多少流量"
                "message_share": round(messages / total_messages, 4) if total_messages else 0.0,
                "prompt_tokens": int(p_tokens or 0),
                "completion_tokens": int(c_tokens or 0),
                "total_tokens": int(p_tokens or 0) + int(c_tokens or 0),
                "cost_usd": round(_to_float(cost), 6),
            }
        )
    # 成本降序：前端默认想看"谁在花钱"
    out.sort(key=lambda item: (-float(item["cost_usd"]), str(item["model"])))
    return out


def _series(hourly: dict[datetime, dict[str, float]]) -> list[dict[str, Any]]:
    """把小时桶摊平成有序数组（**补齐空缺小时**，见模块 docstring 第 4 条）。"""
    return [
        {
            "hour": hour.isoformat(timespec="seconds") + "Z",
            "requests": int(payload["requests"]),
            "errors": int(payload["errors"]),
            "tokens": int(payload["tokens"]),
        }
        for hour, payload in sorted(hourly.items())
    ]


def collect_quality(session: Session, *, hours: int) -> dict[str, Any]:
    """质量指标：拒答率、引用数分布、反思通过率。

    **`reflect_pass_rate` 是近似值**，必须写在返回值旁边一起暴露给前端：
    精确做法是"每次请求取最后一次 reflect span 的 passed 标记"，那要求在
    span 上另存一个业务字段；这里用"该 trace 的 reflect span 没有报错（status=ok）"
    近似。它会**高估**通过率（think 成功 ≠ 反思判定通过），
    所以返回里带上 `reflect_pass_rate_approx=True`，让使用方法上知道别当精确指标用。
    """
    start = _window_start(hours)

    rows = session.execute(
        select(
            Trace.refusal,
            Trace.source_count,
            Trace.retrieval_rounds,
        ).where(Trace.created_at >= start)
    ).all()

    total = len(rows)
    refusals = 0
    source_sum = 0
    zero_source = 0
    rounds_sum = 0
    for refusal, source_count, retrieval_rounds in rows:
        if bool(refusal):
            refusals += 1
        count = int(source_count or 0)
        source_sum += count
        if count == 0:
            zero_source += 1
        rounds_sum += int(retrieval_rounds or 0)

    # 反思通过率：必须触及 span 表，用一条聚合查询拿"有 reflect span 的 trace 数"，
    # 再把条件收紧到 status=ok 拿第二个数。**不把 span 行拉回 Python**：
    # 一个窗口的 span 可能是 trace 数的 10 倍，聚合在 DB 侧做才划算。
    all_reflect = session.execute(
        select(func.count(func.distinct(TraceSpan.trace_id))).where(
            TraceSpan.created_at >= start,
            TraceSpan.name.like("%reflect%"),
        )
    ).scalar()
    ok_reflect = session.execute(
        select(func.count(func.distinct(TraceSpan.trace_id))).where(
            TraceSpan.created_at >= start,
            TraceSpan.name.like("%reflect%"),
            TraceSpan.status == STATUS_OK,
        )
    ).scalar()

    all_reflect_count = int(all_reflect or 0)
    ok_reflect_count = int(ok_reflect or 0)

    return {
        "hours": min(max(int(hours), 1), _MAX_SERIES_HOURS),
        "requests": total,
        "refusal_rate": round(refusals / total, 4) if total else 0.0,
        "avg_source_count": round(source_sum / total, 2) if total else 0.0,
        "zero_source_rate": round(zero_source / total, 4) if total else 0.0,
        "avg_retrieval_rounds": round(rounds_sum / total, 2) if total else 0.0,
        # 分母用"有 reflect span 的 trace 数"而不是全部请求：
        # 快路径（rag 模式）根本没有 reflect 节点，把它算进分母会让通过率被稀释成"永远很低"
        "reflect_pass_rate": round(ok_reflect_count / all_reflect_count, 4)
        if all_reflect_count
        else 0.0,
        "reflect_span_traces": all_reflect_count,
        "reflect_pass_rate_approx": True,
    }


def _prom_lines(name: str, help_text: str, value: float, *, labels: str = "") -> list[str]:
    """生成一段 Prometheus 文本（HELP / TYPE / 值）。

    Prometheus 的 HELP 里不能出现换行与反斜杠，这里统一转义，避免生成非法文本
    让整个 /metrics 抓取失败（一个坏行会让 scrap 整体报错）。
    """
    safe_help = help_text.replace("\\", "\\\\").replace("\n", " ")
    return [f"# HELP {name} {safe_help}", f"# TYPE {name} gauge", f"{name}{labels} {value}"]


def prometheus_text(stats: dict[str, Any]) -> str:
    """把 `collect_stats` 的结果渲染成 Prometheus 文本格式。

    只暴露**稳定的 gauge**，不暴露 label 组合爆炸的东西（例如按模型分组的指标
    在模型名很多时会让时间序列数爆炸）。契约里的 `/metrics` 是给 Prometheus 抓的，
    面板细节走 `/obs/stats` 的 JSON。
    """
    lines: list[str] = []
    requests = float(stats.get("requests") or 0)

    lines += _prom_lines("knowflow_requests_total", "窗口内的请求数", requests)
    lines += _prom_lines(
        "knowflow_errors_total", "窗口内失败的请求数", float(stats.get("errors") or 0)
    )
    lines += _prom_lines(
        "knowflow_error_rate", "窗口内失败率（0~1）", float(stats.get("error_rate") or 0)
    )
    lines += _prom_lines(
        "knowflow_prompt_tokens_total",
        "窗口内 prompt token 总量",
        float(stats.get("prompt_tokens") or 0),
    )
    lines += _prom_lines(
        "knowflow_completion_tokens_total",
        "窗口内 completion token 总量",
        float(stats.get("completion_tokens") or 0),
    )
    lines += _prom_lines(
        "knowflow_tokens_total", "窗口内 token 总量", float(stats.get("total_tokens") or 0)
    )
    lines += _prom_lines(
        "knowflow_cost_usd", "窗口内估算成本（美元）", float(stats.get("cost_usd") or 0)
    )
    lines += _prom_lines(
        "knowflow_cost_cny", "窗口内估算成本（人民币）", float(stats.get("cost_cny") or 0)
    )

    latency = stats.get("latency_ms") or {}
    for key in ("p50", "p95", "p99", "max", "avg"):
        lines += _prom_lines(
            f"knowflow_latency_ms_{key}",
            f"窗口内请求延迟 {key}（毫秒）",
            float(latency.get(key) or 0),
        )
    lines += _prom_lines(
        "knowflow_inflight_requests", "保留指标：当前窗口请求数（兼容旧面板）", requests
    )

    by_mode = stats.get("by_mode") or {}
    for mode, payload in sorted(by_mode.items()):
        label = f'{{mode="{mode}"}}'
        lines += _prom_lines(
            "knowflow_requests_by_mode",
            "按模式分组的请求数",
            float(payload.get("requests") or 0),
            labels=label,
        )
        lines += _prom_lines(
            "knowflow_cost_usd_by_mode",
            "按模式分组的成本（美元）",
            float(payload.get("cost_usd") or 0),
            labels=label,
        )

    return "\n".join(lines) + "\n"


def span_type_breakdown(session: Session, *, hours: int) -> list[dict[str, Any]]:
    """按 span 类型聚合耗时（"时间花在哪一步"）。

    单独一个函数而不是塞进 `collect_stats`：这只是排障入口，
    `/obs/stats` 的主口径是 trace 行，加上它会让响应更大且掩盖重点。
    """
    start = _window_start(hours)
    rows = session.execute(
        select(
            TraceSpan.span_type,
            func.count(TraceSpan.id),
            func.coalesce(func.sum(TraceSpan.duration_ms), 0),
            func.coalesce(func.avg(TraceSpan.duration_ms), 0),
        )
        .where(TraceSpan.created_at >= start)
        .group_by(TraceSpan.span_type)
    ).all()

    out: list[dict[str, Any]] = []
    for span_type, count, total_ms, avg_ms in rows:
        out.append(
            {
                "span_type": str(span_type),
                "count": int(count or 0),
                "total_ms": int(total_ms or 0),
                "avg_ms": round(_to_float(avg_ms), 2),
            }
        )
    out.sort(key=lambda item: -int(item["total_ms"]))
    return out


__all__ = [
    "aggregate_latency",
    "collect_quality",
    "collect_stats",
    "percentile",
    "prometheus_text",
    "span_type_breakdown",
]
