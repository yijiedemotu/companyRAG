"""评测编排：把"一份考卷 + 一种检索配置"跑成可以对比的数字。

**一句话定位**：评测层唯一的编排者 —— 逐条跑检索、算指标、需要时调 LLM 判官，
并把过程与结果落进 `eval_case_results` / `eval_runs`。

**在链路中的位置**：
``POST /eval/runs``（后台线程）-> **本模块** -> `evaluation.metrics` 算指标
-> `eval_runs.metrics_json` -> ``GET /eval/compare`` 做显著性检验（`evaluation.stats`）。

**关键设计取舍**：

1. **相关性判定 = 「期望文档名命中」∪「期望小节命中」（取并集）**。
   只认 `section_path` 会让指标**系统性虚低**：小节路径的格式完全取决于切分器
   （`员工报销制度 > 差旅报销标准` 还是 `差旅报销标准` 还是带编号的 `2.1`），
   标注者写的是他看见的那一种。只认文档名又太松（一篇文档几十个切片，全算命中）。
   并集是这两者之间唯一"既不冤枉好结果、也不放过坏结果"的折中。
   归一化（大小写 + 空白 + 全角空格）是必须的：Windows 上标注意外带空格是常态。
2. **检索候选的稳定标识用 `chunk_id`，没有就退回 `vector_id`**。
   `metrics.recall_at_k` 等函数只要求"可哈希的稳定 id"。
   用 `chunk_id` 而不是 `vector_id` 是因为引用最终落到 `chunk_id`，
   两者在同一个 run 内是等价的（`vector_id = "{doc_id}:{chunk_index}"`）。
3. **`agent` 模式退化成 `hybrid_rerank`**。契约把 `agent` 列进了 `RunMode`，
   但真实跑整条 Agent 链路要对每条用例调 3~5 次大模型（离线模式下还只能拿到抽取式答案，
   指标毫无意义）。所以这里显式降级并把原因写进日志与 `config_json`，
   **不静默**：`eval_runs.config_json.requested_mode` 会保留原始请求。
4. **用例级异常不中断整轮评测**。一条用例检索失败（比如向量库抖了一下）
   只让它自己计 0 分并把 `error` 写进结果行；整轮失败（`status=failed`）留给
   "数据集不存在""模型不可用"这类无法继续的情况。否则 28 条里有一条挂了，
   整轮就白跑，用户还得重跑一遍。
5. **本模块不 import `retrieval.text`**（`evaluation` 包的刻意约束，见其 docstring）：
   评测层必须能在最干净的环境里单独导入。需要的切词能力由
   `evaluation.metrics.tokenize` 提供。
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from knowflow.core.config import Settings
from knowflow.core.logging import get_logger
from knowflow.db.models.evaluation import (
    RUN_DONE,
    RUN_FAILED,
    RUN_RUNNING,
    EvalCase,
    EvalCaseResult,
    EvalDataset,
    EvalRun,
)
from knowflow.db.models.knowledge import KnowledgeBase
from knowflow.db.types import utcnow
from knowflow.evaluation.metrics import hit_rate, mrr, ndcg_at_k, recall_at_k
from knowflow.llm.prompts import REFUSAL_MARKERS

__all__ = ["SUPPORTED_MODES", "fail_run", "run_evaluation"]

logger = get_logger(__name__)

#: 消融要跑的模式（契约 5.8 的 `RunMode` 去掉 `agent`，见模块 docstring 第 3 条）。
SUPPORTED_MODES: tuple[str, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank")

#: `agent` 降级后实际使用的检索模式。
_AGENT_FALLBACK_MODE = "hybrid_rerank"

#: 判官只看前 N 条上下文：判官的成本要可控，而且前几条已经覆盖了主要依据。
_JUDGE_CONTEXT_LIMIT = 5


# --------------------------------------------------------------------------------------
# 归一化与相关性判定
# --------------------------------------------------------------------------------------
def _normalize(text: str | None) -> str:
    """归一化用于比较的文本：去首尾空白、折叠内部空白、全角空格转半角、大小写不敏感。

    为什么必须归一化：`data/eval_cases.jsonl` 里的 `expected_doc` 是人工写的，
    Windows 上从资源管理器复制文件名时常带上不可见字符或多余空格；
    不归一化就会出现"注明明明是同一个文件却判不中"这种假失败。
    """
    if not text:
        return ""
    collapsed = text.replace("\u3000", " ").strip()
    return " ".join(collapsed.split()).casefold()


def _candidate_keys(hit: dict[str, Any]) -> tuple[str, ...]:
    """一个候选的**唯一比较单元**：`chunk_id`（没有就退回 `vector_id`）。

    `metrics` 里的函数只要求"可哈希且同一次评测内稳定"。用 `chunk_id` 而不是
    `vector_id` 是因为引用最终落到 `chunk_id`；`vector_id` 只在命中快照不完整时兜底。
    **一个候选只能产出 0 或 1 个 id 键**——产出多个会让 `recall@k` 的分母与实际召回
    条数对不上（同一个候选被数好几次，指标虚高）。
    """
    chunk_id = hit.get("chunk_id")
    if chunk_id is not None:
        return (f"chunk:{chunk_id}",)
    vector_id = hit.get("vector_id")
    if vector_id:
        return (f"vector:{vector_id}",)
    index = hit.get("chunk_index")
    doc_id = hit.get("doc_id")
    if doc_id is not None and index is not None:
        return (f"vector:{doc_id}:{index}",)
    # 连 id 都没有的候选（理论上不该出现）：用文本哈希兜底，保证稳定
    return (f"text:{abs(hash(str(hit.get('content', ''))))}",)


def _doc_keys(name: str | None) -> set[str]:
    """文档名的可匹配键：全名 + 去扩展名的词干。

    两步都给，是为了兼容两种人工标注习惯：`员工报销制度.md` 与 `员工报销制度`。
    只给全名的话，少写一个 `.md` 的标注就永远判不中 —— 指标虚低且看不出原因。
    """
    text = _normalize(name)
    if not text:
        return set()
    keys = {f"doc:{text}"}
    if "." in text:
        stem = text.rsplit(".", 1)[0].strip()
        if stem:
            keys.add(f"doc:{stem}")
    return keys


def _section_keys(section: str | None) -> set[str]:
    """小节路径的可匹配键：完整路径 + 最后一段（宽松后缀匹配）。

    标注者写 `差旅报销标准`，切分器给出 `员工报销制度 > 差旅报销标准`——
    指的是同一节。不产出后缀键就会把它判成未命中（指标虚低）。
    """
    text = _normalize(section)
    if not text:
        return set()
    keys = {f"section:{text}"}
    for separator in (" > ", ">", "/", "／", "|"):
        if separator in text:
            tail = text.split(separator)[-1].strip()
            if tail:
                keys.add(f"section:{tail}")
            break
    return keys


def _expected_keys(case: EvalCase) -> set[str]:
    """期望命中的标识集合：文档名与小节路径**取并集**（见模块 docstring 第 1 条）。"""
    keys: set[str] = set()
    keys |= _doc_keys(case.expected_doc)
    for section in list(case.expected_sections or []):
        keys |= _section_keys(str(section))
    return keys


def _expected_doc_keys(case: EvalCase) -> set[str]:
    """期望的**文档级** id 集合（算召回类指标的分母）。

    只从 `expected_doc` 取。原因：`expected_sections` 里没有文档信息，
    硬推出来的"文档级期望"会变成"该文档所有切片都算相关"，
    那是把指标往反方向做松（虚高），比虚低更危险。
    **所以数据集必须为每条用例标注 `expected_doc`** —— 这条要求写在文档里，
    而不是在这里静默兜底成"没有期望 = 不计分"（那会让指标看起来是满分）。
    """
    return {f"name:{key}" for key in _doc_keys(case.expected_doc)}


def _candidate_match_keys(hit: dict[str, Any]) -> set[str]:
    """候选侧的语义匹配键（文档名 + 小节路径），用于 `hit` 判据。"""
    keys = _doc_keys(str(hit.get("doc_name") or ""))
    keys |= _section_keys(str(hit.get("section_path") or ""))
    return keys


def _matches(hit: dict[str, Any], expected: set[str]) -> bool:
    """候选是否命中期望集合（**语义并集判据**：文档名或小节路径命中即算命中）。"""
    if not expected:
        return False
    return bool(_candidate_match_keys(hit) & expected)


def _candidate_doc_ids(hit: dict[str, Any]) -> set[str]:
    """候选的**文档级** id 别名集合，用来与 `_expected_doc_keys` 求交。

    为什么要给别名：候选侧能拿到的是 `doc_id`（数字），而期望侧只有文件名。
    两者必须能对上，所以候选同时给出 `doc-id:<id>`、`name:<全名>`、`name:<词干>`
    三种键；期望侧给出 `name:<全名>/<词干>`。交集非空即"这条命中期望文档"。
    """
    aliases: set[str] = set()
    doc_id = hit.get("doc_id")
    if doc_id is not None:
        aliases.add(f"doc-id:{doc_id}")
    for key in _doc_keys(str(hit.get("doc_name") or "")):
        aliases.add(f"name:{key}")
    return aliases


# --------------------------------------------------------------------------------------
# 检索模式归一化
# --------------------------------------------------------------------------------------
def _resolve_mode(mode: str) -> tuple[str, str | None]:
    """返回 `(实际使用的检索模式, 降级说明)`。"""
    normalized = (mode or "").strip().lower()
    if normalized in SUPPORTED_MODES:
        return normalized, None
    if normalized == "agent":
        return (
            _AGENT_FALLBACK_MODE,
            "agent 模式在评测中退化为 hybrid_rerank（逐条调 Agent 链路成本与耗时不可控，"
            "且离线模式下答案是抽取式的，指标无意义）",
        )
    return "hybrid", f"未知模式 {mode!r}，已退化为 hybrid"


# --------------------------------------------------------------------------------------
# 指标聚合
# --------------------------------------------------------------------------------------
def _percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数。

    刻意**再写一份私有实现**而不是 import `observability.metrics.percentile`：
    评测层必须零项目内依赖（见 `evaluation/metrics.py` 的说明），
    让评测包依赖观测包会把这条设计约束破坏掉。
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(p), 0.0), 100.0) / 100.0 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _mean(values: Sequence[float]) -> float | None:
    """均值；空集合回 `None` 而不是 0。

    `None` 的语义是"没算"，0 是"算出来是 0 分"——**在评测里这是完全相反的结论**，
    见 `schemas/evaluation.py` 的 `EvalMetrics` 说明。
    """
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate(rows: list[dict[str, Any]], latencies: list[int]) -> dict[str, Any]:
    """汇总成 `schemas.evaluation.EvalMetrics` 的字段形状。"""
    hits = [bool(row["hit"]) for row in rows]
    recalls = [float(row["recall_at_k"]) for row in rows]
    mrrs = [float(row["mrr"]) for row in rows]
    ndcgs = [float(row["ndcg"]) for row in rows]
    faithfulness = [row["faithfulness"] for row in rows if row["faithfulness"] is not None]
    relevance = [row["answer_relevance"] for row in rows if row["answer_relevance"] is not None]
    refusals = [bool(row["refusal"]) for row in rows]

    return {
        "case_count": len(rows),
        "hit_rate": round(hit_rate(hits), 6) if rows else None,
        "recall_at_k": _round(_mean(recalls)),
        "mrr": _round(_mean(mrrs)),
        "ndcg": _round(_mean(ndcgs)),
        "faithfulness": _round(_mean([float(v) for v in faithfulness])),
        "answer_relevance": _round(_mean([float(v) for v in relevance])),
        "avg_latency_ms": _round(_mean([float(v) for v in latencies])),
        "p95_latency_ms": _round(_percentile([float(v) for v in latencies], 95.0))
        if latencies
        else None,
        "refusal_rate": round(sum(1 for item in refusals if item) / len(refusals), 6)
        if refusals
        else None,
    }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _looks_like_refusal(answer: str) -> bool:
    """答案是否属于拒答。

    与 `llm.prompts.detect_refusal` 用同一批标记，但这里是**简化版**：
    只做"命中标记 + 短文本"判断，不 import 那个函数（它带 200 字长度规则，
    评测场景下我们更关心"有没有明确表示没有依据"）。
    """
    text = (answer or "").strip()
    return bool(text) and len(text) <= 200 and any(marker in text for marker in REFUSAL_MARKERS)


# --------------------------------------------------------------------------------------
# LLM 判官
# --------------------------------------------------------------------------------------
def _judge_scores(
    *,
    judge_model: Any,
    question: str,
    answer: str,
    contexts: list[tuple[int, str]],
) -> tuple[float | None, float | None]:
    """用判官模型算 `faithfulness` / `answer_relevance`。

    任何异常都退化成 `(None, None)` 而不是让它冒泡：判官是**加分项**，
    它挂了不该让整轮评测失败（`None` 的语义是"没算"，前端会显示"—"）。
    """
    from knowflow.llm.parsing import extract_score
    from knowflow.llm.prompts import build_judge_messages

    faithfulness: float | None = None
    relevance: float | None = None
    if not answer.strip():
        return None, None

    try:
        faith_messages = build_judge_messages(
            kind="faithfulness", question=question, answer=answer, contexts=contexts
        )
        faithfulness = extract_score(judge_model.complete(faith_messages).text)
    except Exception as exc:  # noqa: BLE001 - 判官失败不影响主流程
        logger.warning("eval.judge_faithfulness_failed", error=f"{type(exc).__name__}: {exc}"[:200])

    try:
        relevance_messages = build_judge_messages(
            kind="answer_relevance", question=question, answer=answer
        )
        relevance = extract_score(judge_model.complete(relevance_messages).text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval.judge_relevance_failed", error=f"{type(exc).__name__}: {exc}"[:200])

    return faithfulness, relevance


# --------------------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------------------
def run_evaluation(
    *,
    session: Session,
    dataset_id: int,
    mode: str,
    settings: Settings,
    retriever: Any,
    top_k: int = 5,
    use_rerank: bool = False,
    use_llm_judge: bool = False,
    run_id: int | None = None,
    kb_ids: Sequence[int] | None = None,
    judge_model: Any | None = None,
) -> dict[str, Any]:
    """跑一轮评测，落库并返回聚合指标。

    参数说明（契约里没有、但调用方必须知道的三个）：

    - `run_id`：已存在的 `eval_runs.id`。给了就**更新**那一行（HTTP 接口的用法：
      先 201 返回 pending 行，再在后台线程里补齐），不给就新建一行（脚本/CI 的用法）。
    - `kb_ids`：参与检索的知识库。`None` 表示"所有启用的 KB"，
      由调用方（HTTP 层）通过 `KBService.active_kb_ids()` 取好后传入 ——
      评测层不碰数据库里的 KB 业务规则。
    - `judge_model`：`use_llm_judge=True` 且给了它才算忠实度。不给就自动
      用 `llm.factory.build_chat_model(settings)` 建一个（离线时它是 mock 实现）。

    返回（**结构固定，调用方依赖它**）::

        {
          "run_id": int,          # eval_runs.id —— 顶层，调用方要回查 eval_case_results
          "dataset_id": int,
          "mode": str,            # 实际执行的模式（agent 会退化成 hybrid_rerank）
          "metrics": {...},       # 字段与 schemas.evaluation.EvalMetrics 对齐
        }

    `metrics` 里除 `case_count` 之外**全部允许 `None`**：`None` 的语义是"没算"
    （判官关闭时 faithfulness / answer_relevance 无从计算），
    给 0 会被读成"算出来是 0 分"——那是完全相反的结论（见 `EvalMetrics` 的说明）。
    """
    started = time.perf_counter()
    actual_mode, degrade_reason = _resolve_mode(mode)
    if degrade_reason:
        logger.warning(
            "eval.mode_degraded", requested=mode, actual=actual_mode, reason=degrade_reason
        )

    dataset = session.get(EvalDataset, dataset_id)
    if dataset is None:
        raise ValueError(f"评测数据集 {dataset_id} 不存在")

    run = _acquire_run(
        session,
        run_id=run_id,
        dataset_id=dataset_id,
        name=f"{actual_mode}@{dataset.name}",
        mode=actual_mode,
        top_k=top_k,
    )
    run.config_json = {
        "mode": actual_mode,
        "requested_mode": mode,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "use_llm_judge": use_llm_judge,
        "kb_ids": list(kb_ids) if kb_ids is not None else None,
        "degrade_reason": degrade_reason,
        "dataset": dataset.name,
        "embedding_provider": settings.embedding_provider,
        "vector_backend": settings.vector_backend,
        "fusion": settings.fusion,
    }
    run.status = RUN_RUNNING
    run.started_at = utcnow()
    run.error = None
    session.commit()

    judge = (
        judge_model
        if judge_model is not None
        else _build_judge(settings)
        if use_llm_judge
        else None
    )

    cases = list(
        session.execute(
            select(EvalCase).where(EvalCase.dataset_id == dataset_id).order_by(EvalCase.id)
        ).scalars()
    )
    rows: list[dict[str, Any]] = []
    latencies: list[int] = []
    failures = 0

    # ⚠ kb_ids 缺省时必须在这里兜底解析（这是一次真实事故换来的）：
    # 检索引擎在"没有任何 KB 目标"时会**跳过向量召回**（向量库按 KB 分集合，
    # 没目标就无从查起；而 BM25 可以跨库搜）。于是评测会跑出一张看起来正常的表：
    #     vector recall@5 = 0.000 （不是 0 分，是根本没检索）
    #     bm25/hybrid/hybrid_rerank 三者完全相同（向量那一路完全没参与）
    # **"hybrid 与 bm25 一模一样"就是向量没参与的指纹。**
    # 如果只看总数，很容易把"向量没用"当成结论去调融合权重 —— 方向完全错了。
    # 评测工具最怕的就是静默地测了个空。
    if kb_ids is None:
        resolved_kb_ids: list[int] = [
            int(row)
            for row in session.execute(
                select(KnowledgeBase.id)
                .where(KnowledgeBase.deleted_at.is_(None))
                .where(KnowledgeBase.is_active.is_(True))
                .order_by(KnowledgeBase.id)
            ).scalars()
        ]
        logger.info("eval.kb_ids_resolved", count=len(resolved_kb_ids), kb_ids=resolved_kb_ids)
        if not resolved_kb_ids:
            logger.warning("eval.no_active_kb", detail="没有任何启用的知识库，评测结果必然全为 0")
    else:
        resolved_kb_ids = list(kb_ids)

    for case in cases:
        case_started = time.perf_counter()
        record = _run_case(
            session=session,
            run=run,
            case=case,
            mode=actual_mode,
            top_k=top_k,
            use_rerank=use_rerank,
            kb_ids=resolved_kb_ids,
            retriever=retriever,
            judge=judge,
        )
        elapsed = int((time.perf_counter() - case_started) * 1000)
        record["latency_ms"] = elapsed
        latencies.append(elapsed)
        if record["error"]:
            failures += 1
        rows.append(record)

    metrics = _aggregate(rows, latencies)
    passed = sum(1 for row in rows if row["hit"])

    run.status = RUN_DONE
    run.case_count = len(rows)
    run.passed_count = passed
    run.metrics_json = metrics
    run.finished_at = utcnow()
    if failures:
        # 部分用例失败**不算整轮失败**（见模块 docstring 第 4 条），
        # 但要把条数写出来，否则"指标偏低"会被误读成"检索变差了"。
        run.error = f"{failures}/{len(rows)} 条用例执行失败（其余结果仍有效）"
    session.commit()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    metrics["elapsed_ms"] = elapsed_ms
    metrics["failed_cases"] = failures
    logger.info(
        "eval.run_done",
        run_id=run.id,
        dataset=dataset.name,
        mode=actual_mode,
        cases=len(rows),
        passed=passed,
        hit_rate=metrics.get("hit_rate"),
        recall_at_k=metrics.get("recall_at_k"),
        elapsed_ms=elapsed_ms,
    )
    # 返回结构固定为 `{"run_id", "dataset_id", "mode", "metrics"}`：
    # `run_id` 必须在**顶层**，因为调用方要用它回查 `eval_case_results` 做配对 bootstrap
    # （见 `backend/scripts/run_eval.py`）；指标统一藏在 `metrics` 下面，
    # 不与元信息混在同一层（混在一起时 `metrics.get("hit_rate")` 与
    # `metrics.get("run_id")` 看起来一样，很容易把元信息写进对比表）。
    return {
        "run_id": int(run.id),
        "dataset_id": int(dataset_id),
        "mode": actual_mode,
        "metrics": metrics,
    }


def _acquire_run(
    session: Session,
    *,
    run_id: int | None,
    dataset_id: int,
    name: str,
    mode: str,
    top_k: int,
) -> EvalRun:
    """取已有 run 行或新建一行。"""
    if run_id is not None:
        existing = session.get(EvalRun, run_id)
        if existing is not None:
            existing.mode = mode
            existing.top_k = top_k
            return existing
    run = EvalRun(dataset_id=dataset_id, name=name, mode=mode, top_k=top_k)
    session.add(run)
    session.flush()
    return run


def _build_judge(settings: Settings) -> Any | None:
    """按配置建判官模型。失败返回 `None` —— 判官不可用不该让评测起不来。"""
    try:
        from knowflow.llm.factory import build_chat_model

        return build_chat_model(settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("eval.judge_model_unavailable", error=f"{type(exc).__name__}: {exc}"[:200])
        return None


def _run_case(
    *,
    session: Session,
    run: EvalRun,
    case: EvalCase,
    mode: str,
    top_k: int,
    use_rerank: bool,
    kb_ids: Sequence[int] | None,
    retriever: Any,
    judge: Any | None,
) -> dict[str, Any]:
    """跑一条用例：检索 -> 算指标 -> （可选）判官 -> 写 `eval_case_results` 行。

    返回的是**内存里的统计行**（给 `_aggregate` 用），与落库的那一行字段一一对应，
    这样"返回的指标"和"库里的结果"不可能不一致。
    """
    expected = _expected_keys(case)
    expected_doc_ids = _expected_doc_keys(case)
    record: dict[str, Any] = {
        "case_id": case.id,
        "question": case.question,
        "hit": False,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "faithfulness": None,
        "answer_relevance": None,
        "refusal": False,
        "error": None,
        "retrieved": None,
        "answer": None,
    }

    try:
        result = retriever.search(
            query=case.question,
            kb_id=None,
            kb_ids=list(kb_ids) if kb_ids is not None else None,
            mode=mode,
            top_k=top_k,
            use_rerank=use_rerank,
        )
    except Exception as exc:  # noqa: BLE001 - 单条用例失败不该中断整轮
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        logger.warning("eval.case_failed", case_id=case.id, error=record["error"])
        _write_result(session, run=run, case=case, record=record)
        return record

    hits = list(result.hits())
    record["retrieved"] = hits

    # `hit` 与 `recall/mrr/ndcg` 用的是**两种不同粒度的判据**，这是刻意的：
    #
    # - `hit`（命不命中）用**并集语义**：期望文档名命中 **或** 期望小节命中都算命中。
    #   只认 section_path 会让指标系统性虚低（小节路径格式完全取决于切分器）。
    # - `recall/mrr/ndcg` 需要一个"能求交的稳定 id 集合"，所以用**文档级**对齐：
    #   期望侧是文件名（`name:<文件名>`），候选侧给出 `doc-id:<id>` 与
    #   `name:<文件名>` 两种别名，交集非空即算这条命中期望文档。
    #   不把 chunk_id 当 id 用，是因为期望侧根本没有 chunk_id 可对
    #   （`eval_cases` 只标到文档与小节），硬凑出来只会得到一堆空交集。
    scored_ids: list[str] = []
    for hit in hits:
        scored_ids.extend(sorted(_candidate_doc_ids(hit)))

    record["hit"] = any(_matches(hit, expected) for hit in hits)
    record["recall_at_k"] = recall_at_k(scored_ids, expected_doc_ids, top_k)
    record["mrr"] = mrr(scored_ids, expected_doc_ids)
    record["ndcg"] = ndcg_at_k(scored_ids, expected_doc_ids, top_k)

    # 离线/无判官时不给答案打分：`answer` 留 None，指标留 None。
    if judge is not None:
        answer = _extract_answer(result)
        record["answer"] = answer
        record["refusal"] = _looks_like_refusal(answer)
        contexts = [
            (index, str(hit.get("content") or ""))
            for index, hit in enumerate(hits[:_JUDGE_CONTEXT_LIMIT], start=1)
        ]
        faithfulness, relevance = _judge_scores(
            judge_model=judge, question=case.question, answer=answer, contexts=contexts
        )
        record["faithfulness"] = faithfulness
        record["answer_relevance"] = relevance

    _write_result(session, run=run, case=case, record=record)
    return record


def _extract_answer(result: Any) -> str:
    """从检索结果里取一段"答案"。

    `RetrievalResult` 里没有生成答案（检索接口不调大模型，见契约 5.4），
    所以判官能看到的只是**首条命中正文**。这是刻意的：
    评测 `vector`/`bm25`/`hybrid` 这三种模式时，"答案"本就应该是检索片段本身
    （衡量的是"检索出来的内容能不能支撑问题"），而不是再调一次模型凭空生成。
    """
    candidates = getattr(result, "candidates", None) or []
    if not candidates:
        return ""
    first = candidates[0]
    return str(getattr(first, "context_text", "") or getattr(first, "content", "") or "")


def _write_result(
    session: Session, *, run: EvalRun, case: EvalCase, record: dict[str, Any]
) -> None:
    """写一行 `eval_case_results` 并 commit。

    每条用例提交一次（而不是整轮一次）：一轮 28 条要跑几分钟，
    前端轮询 `GET /eval/runs/{id}/results` 时应该能看到**已经跑完的条目**，
    而不是"跑完了才一次出现"。代价是 28 次 commit，可以接受。
    """
    row = EvalCaseResult(
        run_id=run.id,
        case_id=case.id,
        question=case.question,
        retrieved_json=record["retrieved"],
        answer=record["answer"],
        hit=bool(record["hit"]),
        recall_at_k=float(record["recall_at_k"]),
        mrr=float(record["mrr"]),
        ndcg=float(record["ndcg"]),
        faithfulness=record["faithfulness"],
        answer_relevance=record["answer_relevance"],
        latency_ms=record.get("latency_ms"),
        error=record["error"],
    )
    session.add(row)
    session.commit()


def fail_run(session: Session, run_id: int, *, error: str) -> None:
    """把一次运行标记成失败（供后台任务在捕获到无法继续的异常时调用）。"""
    run = session.get(EvalRun, run_id)
    if run is None:
        return
    run.status = RUN_FAILED
    run.error = error[:500]
    run.finished_at = utcnow()
    session.commit()
