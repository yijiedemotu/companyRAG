"""运维路由（契约 5.10）：`/health` `/ready` `/metrics`。

**一句话定位**：探针与监控的三个入口，**在根路径、不需要鉴权、不打访问日志**。

**在链路中的位置**：K8s / 负载均衡 / Prometheus / 前端顶部状态条 -> **本模块** ->
`Container.health()` 与 `observability.metrics`。

**关键设计取舍**：

1. **它们不在 `/api/v1` 下**。探针地址如果跟着 API 版本走，
   一次版本升级就会让所有探针失效 —— 而探针失效的表现是"编排系统以为服务死了，
   开始滚动重启"，那是灾难级的。契约 5.10 把它们钉在根路径是对的。
2. **不需要鉴权**：探针与 Prometheus 抓取器没有 token。所以这三个接口
   **绝不能返回任何秘密**：`Container.health()` 只回脱敏后的运行状态
   （`public_snapshot` 已经把 DSN 里的密码掩掉），绝不要把 `settings` 整个 dump 出来。
3. **`/ready` 与 `/health` 的语义必须分开**：
   `/health` 回 200 只要进程活着（"我在"），`/ready` 回 503 表示"我还不能干活"
   （数据库不通 / 容器没装配）。把两者合成一个，会让编排系统在数据库短暂抖动时
   **杀掉所有副本**（就绪探针失败 → 摘流量 → 不恢复），故障被放大。
4. **`/metrics` 没有响应模型**（`schemas/observability.py` 里写明了）：
   它是 Prometheus 文本格式，不是 JSON。加 `response_model` 会让 FastAPI
   尝试按 JSON 校验并失败。这里直接 `Response(media_type=...)`。
5. **`uptime_s` 由 `app.state.started_at` 算**（lifespan 里写入），
   而不是模块级变量：多 worker / 测试多实例时模块级变量会互相污染。
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from knowflow.api.deps import ContainerDep, Services, SettingsDep
from knowflow.core.logging import get_logger
from knowflow.db.session import check_database
from knowflow.observability.metrics import collect_stats, prometheus_text

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(tags=["运维"])

#: Prometheus 文本格式的 Content-Type。`version=0.0.4` 是 exposition format 的版本，
#: 抓取器按它解析；写成普通 `text/plain` 有些抓取器会拒绝。
_PROMETHEUS_MEDIA_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: `/metrics` 的默认统计窗口。1 小时是折中：够覆盖"最近发生了什么"，
#: 又不会让一次抓取拉太多行（Prometheus 默认 15s 抓一次，每次都很便宜）。
_METRICS_WINDOW_HOURS = 1


def _uptime_seconds(request: Request) -> float:
    """进程已运行秒数。启动时刻由 lifespan 写进 `app.state.started_at`。

    **不放在模块级变量里**：多 worker / 测试造多个 app 时模块级变量会互相污染，
    而"运行时长"恰恰是判断"是不是刚重启过"的依据，被污染就失去了意义。
    取不到时回 0，探针不该因为一个展示字段而 500。
    """
    started_at = getattr(request.app.state, "started_at", None)
    if not isinstance(started_at, (int, float)):
        return 0.0
    return round(max(time.time() - float(started_at), 0.0), 3)


@router.get("/health", summary="健康检查（含真实运行模式）")
def health(request: Request, container: ContainerDep) -> dict[str, object]:
    """如实上报运行期真实状态（契约 5.10：「不骗人」的硬规则）。

    返回 `Container.health()` 的原始 dict（含 `status` / `offline` / `llm_mode` /
    `embedding_mode` / `vector` / `db` / `warnings` 等），并补三个前端要的字段：

    - `uptime_s`：进程运行时长（前端顶栏用它显示"已运行 xx"）；
    - `vector_count`：向量库总向量数（`health()` 把细节放在 `vector` 里，
      但契约 5.10 要求顶层也有一个数）；
    - `pool`：连接池状态（`checkedout` 长期贴上限说明有地方没还连接）。

    **不加 `response_model`**：`Container.health()` 已经是"运行期真实状态"的
    权威形状，用 `HealthOut` 校验一遍会把 `vector` 这类扩展字段过滤掉——
    那些字段恰恰是排障时最有用的（例如 `vector.error`）。
    """
    payload = dict(container.health())
    payload["uptime_s"] = _uptime_seconds(request)

    # 契约 5.10 要求顶层有 `vector_count`，而 `Container.health()` 把细节放在
    # `vector.total_vectors` 里（那里还带 `kb_collections` 与可能的 `error`，都保留）。
    vector_info = payload.get("vector") or {}
    if isinstance(vector_info, dict) and "vector_count" not in payload:
        payload["vector_count"] = int(vector_info.get("total_vectors") or 0)

    payload.setdefault("pool", _pool_snapshot(container))
    return payload


@router.get("/ready", summary="就绪检查（数据库不通则 503）")
def ready(container: ContainerDep, response: Response) -> dict[str, object]:
    """数据库可用且容器存在 -> 200；否则 503。

    容器存在这件事本身由 `ContainerDep` 保证（拿不到就抛 `NotReadyError`，
    会被异常处理器映射成 503 信封）。这里只判断数据库。
    """
    db_info = check_database(container.settings)
    ok = bool(db_info.get("ok"))
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("health.not_ready", error=str(db_info.get("error"))[:200])
    return {
        "ready": ok,
        "db": db_info,
        "embedding_mode": container.embedding_mode(),
        "warnings": list(container.warnings),
    }


@router.get("/metrics", summary="Prometheus 文本指标")
def metrics(services: Services, settings: SettingsDep) -> Response:
    """窗口内的请求量 / token / 成本 / 延迟分位，Prometheus 文本格式。

    **没有响应模型**（见模块 docstring 第 4 条）。
    """
    stats = collect_stats(services.session, hours=_METRICS_WINDOW_HOURS, settings=settings)
    return Response(content=prometheus_text(stats), media_type=_PROMETHEUS_MEDIA_TYPE)


def _pool_snapshot(container: object) -> dict[str, int]:
    """连接池状态。取不到就回全 0 —— 探针不该因为拿不到池信息而 500。"""
    engine = getattr(container, "engine", None)
    pool = getattr(engine, "pool", None)
    if pool is None:
        return {"size": 0, "checkedin": 0, "checkedout": 0, "overflow": 0, "total": 0}

    def _number(name: str) -> int:
        value = getattr(pool, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:  # noqa: BLE001 - 池状态是尽力而为
                return 0
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    checkedin = _number("checkedin")
    checkedout = _number("checkedout")
    return {
        "size": _number("size"),
        "checkedin": checkedin,
        "checkedout": checkedout,
        "overflow": _number("overflow"),
        "total": checkedin + checkedout,
    }
