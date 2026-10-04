"""评测契约：数据集 / 用例 / 运行 / 逐条结果 / 消融 / 对比。

字段来源：契约文档 5.8 节。

契约里有两处只给了形状、没给完整字段，这里的判断都写在对应模型上：

- `AblationGroup`：契约只写 `groups: [AblationGroup]`。既然它是**消融实验**，
  一个分组必然要回答"哪种模式、命中率多少、召回多少、比基线差多少"，
  所以字段按这个最小充分集定义。
- `CompareResponse`：契约明确了 `{a, b, delta, ci95: {low, high}, significant}`。
  `a` / `b` 用 `EvalRunOut`（前端要显示两次运行的名字与配置），
  `delta` 用 `{指标: 差值}` 的字典而不是固定字段——指标集合会变，
  写死字段每加一个指标就要改三处（ORM、schema、前端）。

**`AblationResponse` 里的 `dataset_id` / `top_k` 必须回显**：消融结果是按这两个
参数算出来的，不回显的话前端换个 top_k 再查，看到的还是上一次的结果也不知道。
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from knowflow.db.models import (
    RUN_DONE,
    RUN_FAILED,
    RUN_PENDING,
    RUN_RUNNING,
)
from knowflow.schemas.common import UtcDatetime

__all__ = [
    "AblationGroup",
    "AblationResponse",
    "CaseIn",
    "CompareResponse",
    "DatasetCreateRequest",
    "DatasetOut",
    "EvalCaseOut",
    "EvalCaseResultOut",
    "EvalMetrics",
    "EvalRunOut",
    "RunCreateRequest",
    "RunMode",
    "RunStatus",
]

#: 评测运行模式。比检索模式多一个 `agent`：整条 Agent 链路也要能被评测。
#: `Literal[...]` 只接受字面量（mypy 硬限制），`RunStatus` 后面的断言负责
#: 保证它与 ORM 的状态常量不漂移。
RunMode = Literal["vector", "bm25", "hybrid", "hybrid_rerank", "agent"]

#: 运行状态。
RunStatus = Literal["pending", "running", "done", "failed"]

assert get_args(RunStatus) == (RUN_PENDING, RUN_RUNNING, RUN_DONE, RUN_FAILED), (
    "RunStatus 与 db/models/evaluation.py 的状态常量不一致"
)


# --------------------------------------------------------------------------------------
# 数据集与用例
# --------------------------------------------------------------------------------------
class CaseIn(BaseModel):
    """录入一道评测题（`DatasetCreateRequest.cases[]`）。

    和 `EvalCaseOut` 的区别只有"没有服务端生成的字段"（id/created_at），
    但**刻意分开定义**：录入接口不该接受客户端指定 `id`——
    那会让 `id` 变成可注入字段（重复 id 会直接 500 或覆盖别人的数据）。
    """

    model_config = ConfigDict(from_attributes=True)

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="评测问题",
    )
    ground_truth: str | None = Field(
        default=None, description="参考答案（可选）；不填则只算召回类指标，不算忠实度"
    )
    expected_doc: str | None = Field(
        default=None,
        max_length=255,
        description="应命中的文档文件名，用于计算命中率/召回",
    )
    expected_sections: list[str] | None = Field(
        default=None,
        description="应命中的 section_path 列表；一个问题合理地跨多节时全部列出，只认一个会让指标虚低",
    )
    tags: list[str] | None = Field(
        default=None,
        description="标签，如 synonym / proper_noun / multi_hop，用于分类看失败模式",
    )


class DatasetCreateRequest(BaseModel):
    """创建评测数据集（契约 5.8：`{name, description?, cases: [...]}`）。

    `cases` 要求至少 1 条：空数据集能建成，但跑起来必然 `case_count=0`、
    指标全是 `null`，看起来像评测功能坏了。在入口就挡住更省事。
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(min_length=1, max_length=128, description="数据集名称，全局唯一")
    description: str | None = Field(
        default=None, max_length=512, description="描述，建议写清来源与规模"
    )
    cases: list[CaseIn] = Field(min_length=1, description="用例列表，至少 1 条")


class DatasetOut(BaseModel):
    """评测数据集（契约 5.8 的 `[DatasetOut]`）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="数据集 ID")
    name: str = Field(description="数据集名称")
    description: str | None = Field(default=None, description="描述")
    case_count: int = Field(default=0, ge=0, description="用例条数（冗余计数）")


class EvalCaseOut(BaseModel):
    """一道评测题（契约 5.8 的 `Page[EvalCaseOut]`）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="用例 ID")
    dataset_id: int = Field(description="所属数据集 ID")
    question: str = Field(description="评测问题")
    ground_truth: str | None = Field(default=None, description="参考答案，可能为空")
    expected_doc: str | None = Field(default=None, description="应命中的文档文件名")
    expected_sections: list[str] = Field(
        default_factory=list, description="应命中的 section_path 列表"
    )
    tags: list[str] = Field(default_factory=list, description="标签列表")
    created_at: UtcDatetime = Field(default=None, description="创建时间，UTC ISO8601 带 Z")


# --------------------------------------------------------------------------------------
# 运行与指标
# --------------------------------------------------------------------------------------
class RunCreateRequest(BaseModel):
    """发起一次评测（契约 5.8：`{dataset_id, name?, mode, top_k?, use_rerank?, use_llm_judge?}`）。

    接口**立即返回 201**，评测在后台执行（`status=pending/running`），
    前端轮询 `GET /eval/runs/{id}`。不用同步等待的原因很直接：
    28 条用例跑一遍 agent 链路要几分钟，HTTP 请求早超时了。
    """

    model_config = ConfigDict(from_attributes=True)

    dataset_id: int = Field(ge=1, description="要评测的数据集 ID")
    name: str | None = Field(
        default=None,
        max_length=128,
        description="本次运行名称；留空由服务端按模式+时间生成",
    )
    mode: RunMode = Field(
        description="检索/问答模式：vector / bm25 / hybrid / hybrid_rerank / agent"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="每次检索取前 K 条")
    use_rerank: bool = Field(
        default=False, description="是否启用重排（hybrid_rerank 之外的模式也可叠加）"
    )
    use_llm_judge: bool = Field(
        default=False,
        description="是否用大模型判忠实度/相关性；开启会更准但更贵更慢，离线模式无效",
    )


class EvalMetrics(BaseModel):
    """聚合指标（契约 5.8 的 `EvalMetrics`）。

    全部指标都允许 `null`：LLM 判官关闭时 faithfulness / answer_relevance 无从计算，
    给 `null` 表示"没算"；给 0 会被误读成"算出来是 0 分"，那是完全相反的结论。
    """

    model_config = ConfigDict(from_attributes=True)

    case_count: int = Field(default=0, ge=0, description="参与统计的用例数")
    hit_rate: float | None = Field(default=None, description="命中期望文档/小节的比例")
    recall_at_k: float | None = Field(default=None, description="前 K 条内的召回率")
    mrr: float | None = Field(default=None, description="平均倒数排名")
    ndcg: float | None = Field(default=None, description="归一化折损累计增益")
    faithfulness: float | None = Field(
        default=None, description="答案对检索内容的忠实度；需开启 LLM 判官"
    )
    answer_relevance: float | None = Field(
        default=None, description="答案与问题的相关性；需开启 LLM 判官"
    )
    avg_latency_ms: float | None = Field(default=None, ge=0, description="平均延迟（毫秒）")
    p95_latency_ms: float | None = Field(default=None, ge=0, description="P95 延迟（毫秒）")
    refusal_rate: float | None = Field(default=None, description="拒答率，越低越好但不能为 0 作假")


class EvalRunOut(BaseModel):
    """一次评测运行（契约 5.8）。

    `config_json` 原样返回：没有它，两周后看到 "recall@5 = 0.82" 根本不知道
    当时用的什么参数，也就无法复现或改进。这是评测体系里最容易省、也最不该省的东西。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="运行 ID")
    dataset_id: int = Field(description="所属数据集 ID")
    name: str = Field(description="运行名称")
    mode: RunMode = Field(description="本次使用的模式")
    top_k: int = Field(default=5, ge=1, description="本次使用的 top_k")
    config_json: dict[str, Any] | None = Field(default=None, description="本次运行的完整参数快照")
    status: RunStatus = Field(description="状态：pending / running / done / failed")
    case_count: int = Field(default=0, ge=0, description="用例总数")
    passed_count: int = Field(default=0, ge=0, description="命中（通过）的用例数")
    metrics_json: EvalMetrics | None = Field(default=None, description="聚合指标；未完成时为 null")
    error: str | None = Field(default=None, description="失败原因")
    started_at: UtcDatetime = Field(default=None, description="开始时间，UTC ISO8601 带 Z")
    finished_at: UtcDatetime = Field(default=None, description="结束时间，UTC ISO8601 带 Z")
    created_at: UtcDatetime = Field(default=None, description="创建时间，UTC ISO8601 带 Z")


class EvalCaseResultOut(BaseModel):
    """一条用例的作答记录（契约 5.8）。

    `retrieved_json` 保留命中快照（含每个候选的各路分数），
    这样"这条为什么没命中"可以直接从前端查出来，不用重跑一次评测。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="结果 ID")
    run_id: int = Field(description="所属运行 ID")
    case_id: int = Field(description="对应用例 ID")
    question: str = Field(description="问题（冗余存一份，用例被删后历史结果仍可读）")
    retrieved_json: list[dict[str, Any]] | None = Field(
        default=None, description="命中列表快照，含各阶段分数与排名"
    )
    answer: str | None = Field(default=None, description="生成的答案")
    hit: bool = Field(default=False, description="是否命中期望文档/小节")
    recall_at_k: float | None = Field(default=None, description="本条召回率")
    mrr: float | None = Field(default=None, description="本条倒数排名")
    ndcg: float | None = Field(default=None, description="本条 NDCG")
    faithfulness: float | None = Field(default=None, description="本条忠实度，需 LLM 判官")
    answer_relevance: float | None = Field(default=None, description="本条答案相关性，需 LLM 判官")
    latency_ms: int | None = Field(default=None, description="本条耗时（毫秒）")
    error: str | None = Field(default=None, description="本条失败原因")
    created_at: UtcDatetime = Field(default=None, description="创建时间，UTC ISO8601 带 Z")


# --------------------------------------------------------------------------------------
# 消融与对比
# --------------------------------------------------------------------------------------
class AblationGroup(BaseModel):
    """消融实验的一个分组（契约 5.8 只给了名字，字段由此定义）。

    `delta_vs_baseline` 是**带符号差值**（分组指标 − 基线指标）：
    只给绝对值，读者还得自己在脑子里做减法，而且做反了方向（把提升看成下降）
    是很容易发生的事。
    """

    model_config = ConfigDict(from_attributes=True)

    mode: RunMode = Field(description="该分组使用的模式")
    label: str = Field(default="", description="展示名称，如「hybrid + rerank」")
    top_k: int = Field(default=5, ge=1, description="该分组使用的 top_k")
    run_id: int | None = Field(default=None, description="对应运行 ID；未跑过为 null")
    case_count: int = Field(default=0, ge=0, description="用例数")
    hit_rate: float | None = Field(default=None, description="命中率")
    recall_at_k: float | None = Field(default=None, description="召回率")
    mrr: float | None = Field(default=None, description="MRR")
    ndcg: float | None = Field(default=None, description="NDCG")
    avg_latency_ms: float | None = Field(default=None, ge=0, description="平均延迟（毫秒）")
    delta_vs_baseline: dict[str, float] = Field(
        default_factory=dict,
        description="相对基线的指标差值，正数表示更好（延迟类指标需要注意方向相反）",
    )


class AblationResponse(BaseModel):
    """消融实验结果（契约 5.8：`{dataset_id, groups: [AblationGroup]}`）。"""

    model_config = ConfigDict(from_attributes=True)

    dataset_id: int = Field(description="数据集 ID")
    top_k: int = Field(default=5, ge=1, description="本次按哪个 top_k 计算（回显，避免看错结果）")
    baseline_mode: RunMode | None = Field(
        default=None, description="作为基线的模式；其余分组都与它比较"
    )
    groups: list[AblationGroup] = Field(default_factory=list, description="各模式分组结果")


class CompareResponse(BaseModel):
    """两次运行的对比（契约 5.8：`{a, b, delta, ci95, significant}`）。

    `ci95` 与 `significant` 是**统计显著性**的表达，不能省：
    28 条用例上 recall 从 0.82 涨到 0.86（差 2 条）完全可能是噪声，
    不加置信区间就宣布"优化有效"，等于在自欺欺人。前端据此决定是否标"显著"。
    """

    model_config = ConfigDict(from_attributes=True)

    a: EvalRunOut = Field(description="运行 A（通常作为基线）")
    b: EvalRunOut = Field(description="运行 B（通常作为改进版）")
    delta: dict[str, float] = Field(default_factory=dict, description="B − A 的各指标差值")
    ci95: dict[str, float] = Field(
        default_factory=dict,
        description="差值 95% 置信区间，形如 {low: -0.03, high: 0.11}",
    )
    significant: bool = Field(default=False, description="差异是否统计显著（置信区间不跨 0）")
