"""评测路由（契约 5.8）。

**一句话定位**：数据集管理、发起评测运行、查看逐条结果、消融对比与显著性检验。

**在链路中的位置**：前端 `/eval` 页面 -> **本模块** -> `evaluation.harness`（后台跑）
-> `eval_runs` / `eval_case_results` -> `evaluation.stats`（对比）。

**关键设计取舍**：

1. **`POST /eval/runs` 只创建 `eval_runs` 行并立刻返回 201**，真正跑评测交给
   `BackgroundTasks`。28 条用例跑一遍检索要几十秒到几分钟，同步等必然超时；
   契约里 run 有 `status`（pending/running/done/failed），前端就是按它轮询的。
2. **后台任务里用独立的 `session_factory()` 建自己的会话**，不复用请求的 session：
   请求结束后依赖会 close 掉那个 session，后台任务再拿它写库就会拿到
   "Session is closed"。这里的代价是后台任务要自己保证 close（用 `try/finally`）。
3. **`GET /eval/datasets` 返回数组而不是 `Page`**。契约 5.8 就是这么写的，
   前端也已按数组处理（见 `frontend/src/api/eval.ts`）。
   **不要"顺手统一成 Page"**：那会让前端的 `.map()` 崩在 `undefined` 上。
4. **`GET /eval/compare` 的 `significant` / `ci95` 来自 `paired_bootstrap_test`**，
   不是简单比较两个平均值。28 条用例上"涨了 4 个点"很可能只是噪声，
   不做检验就等于在自欺欺人（见 `evaluation/stats.py` 的说明）。
5. **本文件里有 `select()`**（第三处例外，见 `api/__init__.py`）：
   评测数据集的 CRUD 没有对应的 service（`evaluation/datasets.py` 只提供
   幂等导入 `get_or_create_dataset`），而"列出/分页查询"是纯只读聚合。
   所有写操作都走 `evaluation.datasets` 的现成函数或显式构造 ORM 行。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query, status
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from knowflow.api.deps import ContainerDep, CurrentUser, Services
from knowflow.container import Container
from knowflow.core.exceptions import NotFoundError, RunNotFoundError, ValidationError
from knowflow.core.logging import get_logger
from knowflow.db.models.evaluation import (
    RUN_DONE,
    EvalCase,
    EvalCaseResult,
    EvalDataset,
    EvalRun,
)
from knowflow.evaluation.datasets import case_from_dict, get_or_create_dataset
from knowflow.evaluation.harness import SUPPORTED_MODES, fail_run, run_evaluation
from knowflow.evaluation.stats import paired_bootstrap_test
from knowflow.schemas.common import Page, PageParams
from knowflow.schemas.evaluation import (
    AblationGroup,
    AblationResponse,
    CompareResponse,
    DatasetCreateRequest,
    DatasetOut,
    EvalCaseOut,
    EvalCaseResultOut,
    EvalRunOut,
    RunCreateRequest,
)

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(prefix="/eval", tags=["评测"])

PageDep = Annotated[PageParams, Depends()]

#: 消融实验的展示名。取值与 `schemas.evaluation.RunMode` 对齐（`agent` 不在消融里）。
_MODE_LABELS: dict[str, str] = {
    "vector": "纯向量检索",
    "bm25": "纯关键词检索（BM25）",
    "hybrid": "混合检索（RRF 融合）",
    "hybrid_rerank": "混合检索 + 重排",
    "agent": "Agent 链路",
}

#: 消融的基线模式：契约与前端都把它当成"对照组的默认值"。
_DEFAULT_BASELINE = "hybrid"


# --------------------------------------------------------------------------------------
# 数据集与用例
# --------------------------------------------------------------------------------------
@router.get("/datasets", response_model=list[DatasetOut], summary="数据集列表（数组）")
def list_datasets(params: PageDep, services: Services) -> list[DatasetOut]:
    """**返回数组**（契约 5.8），不是 `Page[DatasetOut]`。

    `page` / `size` 仍然接受（前端默认传 `size=100`），只是响应形状固定为数组。
    """
    rows = list(
        services.session.execute(
            select(EvalDataset)
            .order_by(EvalDataset.id.desc())
            .offset(params.offset)
            .limit(params.limit)
        ).scalars()
    )
    return [DatasetOut.model_validate(row) for row in rows]


@router.post(
    "/datasets",
    response_model=DatasetOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建/更新数据集（按 name 幂等）",
)
def create_dataset(payload: DatasetCreateRequest, services: Services) -> DatasetOut:
    """按 `name` 幂等 upsert，用例按 `question` 幂等 upsert。

    幂等是刻意的（见 `evaluation/datasets.py`）：标注集会被反复修订，
    每次导入都 INSERT 会让同一道题堆多个版本，`recall@5` 的分母被无声放大。
    """
    cases = [case_from_dict(case.model_dump()) for case in payload.cases]
    dataset = get_or_create_dataset(
        services.session, name=payload.name, description=payload.description, cases=cases
    )
    services.session.commit()
    return DatasetOut.model_validate(dataset)


@router.get(
    "/datasets/{dataset_id}/cases",
    response_model=Page[EvalCaseOut],
    summary="数据集用例",
)
def list_dataset_cases(dataset_id: int, params: PageDep, services: Services) -> Page[EvalCaseOut]:
    dataset = services.session.get(EvalDataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"评测数据集 {dataset_id} 不存在")

    total = int(
        services.session.execute(
            select(func.count(EvalCase.id)).where(EvalCase.dataset_id == dataset_id)
        ).scalar()
        or 0
    )
    rows = (
        list(
            services.session.execute(
                select(EvalCase)
                .where(EvalCase.dataset_id == dataset_id)
                .order_by(EvalCase.id)
                .offset(params.offset)
                .limit(params.limit)
            ).scalars()
        )
        if total
        else []
    )
    return Page[EvalCaseOut].create(
        items=[EvalCaseOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


# --------------------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------------------
@router.post(
    "/runs",
    response_model=EvalRunOut,
    status_code=status.HTTP_201_CREATED,
    summary="发起评测（后台执行）",
)
def create_run(
    payload: RunCreateRequest,
    background: BackgroundTasks,
    services: Services,
    user: CurrentUser,
    container: ContainerDep,
) -> EvalRunOut:
    """创建 `pending` 行 → 201 返回 → 后台线程真正跑。

    返回体里的 `status` 是 `pending`/`running`，前端按 `GET /eval/runs/{id}` 轮询。
    """
    dataset = services.session.get(EvalDataset, payload.dataset_id)
    if dataset is None:
        raise NotFoundError(f"评测数据集 {payload.dataset_id} 不存在")
    if dataset.case_count == 0:
        raise ValidationError(f"数据集「{dataset.name}」没有任何用例，评测没有意义")

    mode = payload.mode
    if mode not in SUPPORTED_MODES and mode != "agent":
        raise ValidationError(f"不支持的评测模式：{mode}")

    run = EvalRun(
        dataset_id=payload.dataset_id,
        name=payload.name or f"{mode}@{dataset.name}",
        mode=mode,
        top_k=payload.top_k,
        config_json={
            "mode": mode,
            "top_k": payload.top_k,
            "use_rerank": payload.use_rerank,
            "use_llm_judge": payload.use_llm_judge,
            "requested_by": user.id,
        },
        status="pending",
        case_count=0,
    )
    services.session.add(run)
    services.session.commit()
    run_id = int(run.id)

    background.add_task(
        _run_in_background,
        container=container,
        run_id=run_id,
        dataset_id=payload.dataset_id,
        mode=mode,
        top_k=payload.top_k,
        use_rerank=payload.use_rerank,
        use_llm_judge=payload.use_llm_judge,
    )
    return EvalRunOut.model_validate(run)


def _run_in_background(
    *,
    container: Container,
    run_id: int,
    dataset_id: int,
    mode: str,
    top_k: int,
    use_rerank: bool,
    use_llm_judge: bool,
) -> None:
    """后台执行体。**自己开 session、自己关**（见模块 docstring 第 2 条）。

    异常在这里被捕获并写回 `eval_runs.status=failed` + `error`：
    后台线程里抛异常不会有人看见（没有请求上下文），
    不写库的话前端会永远轮询到一个停在 `pending` 的运行。
    """
    session: Session = container.session_factory()
    try:
        # 用**全局**检索器（跨所有启用的 KB）：评测不绑定某一个 KB，
        # 否则"换 KB 重跑"就得不到可比的数字。
        services = container.build_request_services(session)
        kb_ids = services.kb_service.active_kb_ids()
        if not kb_ids:
            raise ValidationError("没有任何启用的知识库，无法评测（先建 KB 并上传文档）")

        run_evaluation(
            session=session,
            dataset_id=dataset_id,
            mode=mode,
            top_k=top_k,
            use_rerank=use_rerank,
            use_llm_judge=use_llm_judge,
            settings=container.settings,
            retriever=services.retriever,
            run_id=run_id,
            kb_ids=kb_ids,
            judge_model=container.judge_model,
        )
    except Exception as exc:  # noqa: BLE001 - 后台任务：任何异常都要落库可见
        logger.error(
            "eval.background_failed",
            run_id=run_id,
            error=f"{type(exc).__name__}: {exc}"[:300],
            exc_info=True,
        )
        try:
            session.rollback()
            fail_run(session, run_id, error=f"{type(exc).__name__}: {exc}")
        except Exception as inner:  # noqa: BLE001
            logger.error("eval.fail_run_write_failed", run_id=run_id, error=str(inner)[:200])
    finally:
        session.close()


@router.get("/runs", response_model=Page[EvalRunOut], summary="评测运行列表")
def list_runs(
    params: PageDep,
    services: Services,
    dataset_id: Annotated[int | None, Query(ge=1, description="按数据集过滤")] = None,
) -> Page[EvalRunOut]:
    conditions: list[ColumnElement[bool]] = []
    if dataset_id is not None:
        conditions.append(EvalRun.dataset_id == dataset_id)

    total = int(
        services.session.execute(select(func.count(EvalRun.id)).where(*conditions)).scalar() or 0
    )
    rows = (
        list(
            services.session.execute(
                select(EvalRun)
                .where(*conditions)
                .order_by(EvalRun.id.desc())
                .offset(params.offset)
                .limit(params.limit)
            ).scalars()
        )
        if total
        else []
    )
    return Page[EvalRunOut].create(
        items=[EvalRunOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


@router.get("/runs/{run_id}", response_model=EvalRunOut, summary="运行详情（含指标）")
def get_run(run_id: int, services: Services) -> EvalRunOut:
    run = services.session.get(EvalRun, run_id)
    if run is None:
        raise RunNotFoundError(f"评测运行 {run_id} 不存在")
    return EvalRunOut.model_validate(run)


@router.get(
    "/runs/{run_id}/results",
    response_model=Page[EvalCaseResultOut],
    summary="逐条结果",
)
def list_run_results(run_id: int, params: PageDep, services: Services) -> Page[EvalCaseResultOut]:
    run = services.session.get(EvalRun, run_id)
    if run is None:
        raise RunNotFoundError(f"评测运行 {run_id} 不存在")

    total = int(
        services.session.execute(
            select(func.count(EvalCaseResult.id)).where(EvalCaseResult.run_id == run_id)
        ).scalar()
        or 0
    )
    rows = (
        list(
            services.session.execute(
                select(EvalCaseResult)
                .where(EvalCaseResult.run_id == run_id)
                .order_by(EvalCaseResult.id)
                .offset(params.offset)
                .limit(params.limit)
            ).scalars()
        )
        if total
        else []
    )
    return Page[EvalCaseResultOut].create(
        items=[EvalCaseResultOut.model_validate(row) for row in rows],
        total=total,
        page=params.page,
        size=params.size,
    )


# --------------------------------------------------------------------------------------
# 消融与对比
# --------------------------------------------------------------------------------------
@router.get("/ablation", response_model=AblationResponse, summary="消融对比")
def ablation(
    services: Services,
    dataset_id: Annotated[int, Query(ge=1, description="数据集 ID")],
    top_k: Annotated[int, Query(ge=1, le=50, description="按哪个 top_k 取结果")] = 5,
) -> AblationResponse:
    """把同一数据集、同一 `top_k` 下**各种模式的最近一次已完成运行**并列出来。

    每个模式取最近一次 `status=done` 的 run（同一模式可能跑过很多次，
    取最新的是"当前配置"的最好近似），并算出相对基线（默认 `hybrid`）的差值。
    """
    dataset = services.session.get(EvalDataset, dataset_id)
    if dataset is None:
        raise NotFoundError(f"评测数据集 {dataset_id} 不存在")

    runs = list(
        services.session.execute(
            select(EvalRun)
            .where(EvalRun.dataset_id == dataset_id)
            .where(EvalRun.top_k == top_k)
            .where(EvalRun.status == RUN_DONE)
            .order_by(EvalRun.id)
        ).scalars()
    )
    # 后出现的覆盖先出现的 → 每个模式留下"最新那次"
    # 循环变量叫 item 而不是 run：下面 `run = latest.get(mode)` 的类型是 `EvalRun | None`，
    # 若这里也用 run，mypy 会把 run 的类型先推成 `EvalRun`，之后赋 None 就报不兼容。
    latest: dict[str, EvalRun] = {}
    for item in runs:
        latest[str(item.mode)] = item

    baseline = latest.get(_DEFAULT_BASELINE) or (runs[-1] if runs else None)
    baseline_metrics = _metrics_of(baseline)

    groups: list[AblationGroup] = []
    for mode in (*SUPPORTED_MODES, "agent"):
        run = latest.get(mode)
        metrics = _metrics_of(run)
        groups.append(
            AblationGroup(
                mode=mode,  # type: ignore[arg-type]  # 取值来自 RunMode 集合
                label=_MODE_LABELS.get(mode, mode),
                top_k=top_k,
                run_id=int(run.id) if run is not None else None,
                case_count=int(metrics.get("case_count") or 0),
                hit_rate=_opt_float(metrics.get("hit_rate")),
                recall_at_k=_opt_float(metrics.get("recall_at_k")),
                mrr=_opt_float(metrics.get("mrr")),
                ndcg=_opt_float(metrics.get("ndcg")),
                avg_latency_ms=_opt_float(metrics.get("avg_latency_ms")),
                delta_vs_baseline=_deltas(metrics, baseline_metrics),
            )
        )

    return AblationResponse(
        dataset_id=dataset_id,
        top_k=top_k,
        baseline_mode=baseline.mode if baseline is not None else None,  # type: ignore[arg-type]
        groups=groups,
    )


@router.get("/compare", response_model=CompareResponse, summary="两次运行对比（含显著性）")
def compare(
    services: Services,
    run_a: Annotated[int, Query(ge=1, description="运行 A（通常作为基线）")],
    run_b: Annotated[int, Query(ge=1, description="运行 B（通常作为改进版）")],
) -> CompareResponse:
    """对比两次运行：`delta` 是 **B − A**，`ci95` / `significant` 来自成对 bootstrap。

    `significant` 用**用例级**逐条得分的成对差分算（而不是拿两个均值比大小）：
    两次运行跑的是同一批用例，成对差分消掉了"题目难度"这个共同方差，
    在小评测集（本项目 28~43 条）上是唯一能分辨"真提升"与"噪声"的做法。
    """
    a = _get_run(services, run_a)
    b = _get_run(services, run_b)
    if a.dataset_id != b.dataset_id:
        raise ValidationError(
            f"两次运行不属于同一数据集（{a.dataset_id} vs {b.dataset_id}），"
            "对比没有意义：题目都不一样，任何 delta 都无法解释"
        )

    metrics_a = _metrics_of(a)
    metrics_b = _metrics_of(b)
    delta = _deltas(metrics_b, metrics_a)

    scores_a = _case_scores(services, run_a)
    scores_b = _case_scores(services, run_b)

    stats = _paired(scores_a, scores_b)
    return CompareResponse.model_validate(
        {
            "a": a,
            "b": b,
            "delta": delta,
            "ci95": {
                "low": float(stats.get("ci95", {}).get("low", 0.0)),
                "high": float(stats.get("ci95", {}).get("high", 0.0)),
            },
            "significant": bool(stats.get("significant", False)),
        }
    )


def _get_run(services: Services, run_id: int) -> EvalRun:
    run = services.session.get(EvalRun, run_id)
    if run is None:
        raise RunNotFoundError(f"评测运行 {run_id} 不存在")
    return run


def _metrics_of(run: EvalRun | None) -> dict[str, Any]:
    if run is None or not isinstance(run.metrics_json, dict):
        return {}
    return dict(run.metrics_json)


def _opt_float(value: Any) -> float | None:
    """`None` 保持 `None`（"没算"与"算出来是 0"是两件事，见 `EvalMetrics` 说明）。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: 参与消融对比的指标集合。与 `EvalMetrics` 的可比字段对齐（延迟方向相反，展示时要注意）。
_COMPARABLE_METRICS: tuple[str, ...] = (
    "hit_rate",
    "recall_at_k",
    "mrr",
    "ndcg",
    "faithfulness",
    "answer_relevance",
    "avg_latency_ms",
)


def _deltas(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    """`current - baseline` 的带符号差值（正数表示更好，延迟类除外）。"""
    out: dict[str, float] = {}
    for key in _COMPARABLE_METRICS:
        left = _opt_float(current.get(key))
        right = _opt_float(baseline.get(key))
        if left is None or right is None:
            continue
        out[key] = round(left - right, 6)
    return out


def _case_scores(services: Services, run_id: int) -> list[float]:
    """按 `case_id` 升序取每条用例的"综合命中分"。

    综合分 = `(hit + recall_at_k + ndcg) / 3`。**为什么要合成一个数**：
    `paired_bootstrap_test` 检验的是一个标量的成对差分，而"这次提升算不算显著"
    是对整个运行而言的。三个指标各自检验会得到三个 p 值，
    读者还得自己去调和"一个显著两个不显著"。
    合成分稳定、有界（[0,1]）、且对三种失败模式都敏感（没命中 / 没捞全 / 排太后的）。
    """
    rows = list(
        services.session.execute(
            select(
                EvalCaseResult.hit,
                EvalCaseResult.recall_at_k,
                EvalCaseResult.ndcg,
            )
            .where(EvalCaseResult.run_id == run_id)
            .order_by(EvalCaseResult.case_id)
        ).all()
    )
    scores: list[float] = []
    for hit, recall, ndcg in rows:
        scores.append((float(bool(hit)) + float(recall or 0.0) + float(ndcg or 0.0)) / 3.0)
    return scores


def _paired(scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
    """成对 bootstrap。长度不等（对比了不同数据集）时不抛错，返回"不显著 + 零区间"。

    这里**刻意不把 `ValueError` 抛给用户**：对比两次用例数不同的运行是很容易发生的
    误操作（比如数据集补了题之后重跑），报 400 帮不上忙；
    返回 `significant=false` + 全 0 的 CI，前端会显示"差异不显著"，
    读者立刻知道这个对比不可信。真正的错误（不是同一数据集）在前面已经挡掉了。
    """
    if not scores_a or not scores_b or len(scores_a) != len(scores_b):
        return {"significant": False, "ci95": {"low": 0.0, "high": 0.0}, "p_value": 1.0}
    return paired_bootstrap_test(scores_a, scores_b)
