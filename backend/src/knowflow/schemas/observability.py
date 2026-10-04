"""可观测与运维契约（`/api/v1/obs/*`、`/health`）。

字段来源：契约文档 5.9（可观测）与 5.10（运维）。

**为什么 `/health` 的字段要这么细**：契约 5.10 说得非常直接——
「`/health` 必须如实上报运行期真实模式」。这不是装饰性接口，是**不让系统骗人**的
第一道闸门：

- `offline=true` 表示没配 API Key，答案不是大模型写的；
- `embedding_mode="hash(fallback:local_load_failed)"` 表示向量化降级成了哈希兜底，
  检索质量已经不可信（哈希向量没有语义，召回全靠运气）；
- `db.ok=false`、`warnings: [...]` 让"能启动但功能不全"这种状态可见。

这三种情况如果不暴露，页面看起来一切正常，用户会以为检索质量差是"模型不行"。

`/metrics` 返回 Prometheus 文本格式，因此**没有对应的响应模型**——
它不是 JSON，用 `PlainTextResponse` 返回。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from knowflow.schemas.common import UtcDatetime

__all__ = [
    "ByModeOut",
    "ByModelOut",
    "DBHealthOut",
    "HealthOut",
    "LatencyOut",
    "ObsStatsOut",
    "PoolHealthOut",
    "QualityOut",
    "TimelinePointOut",
    "TokenStatsOut",
    "TraceDetailOut",
    "TraceOut",
    "TraceSpanOut",
]


class TokenStatsOut(BaseModel):
    """token 用量汇总（契约 5.9 的 `tokens: {prompt, completion, total}`）。"""

    model_config = ConfigDict(from_attributes=True)

    prompt: int = Field(default=0, ge=0, description="输入 token 合计")
    completion: int = Field(default=0, ge=0, description="输出 token 合计")
    total: int = Field(default=0, ge=0, description="合计 token 数")


class LatencyOut(BaseModel):
    """延迟分位（契约 5.9 的 `latency: {p50,p95,p99,max,avg}`）。

    为什么必须给分位数而不是只有平均值：平均 1.2s 可能藏着 10% 的请求要 8s，
    而用户感知到的正是那些慢请求。只看平均值做优化，方向往往完全是错的。
    """

    model_config = ConfigDict(from_attributes=True)

    p50: float = Field(default=0.0, ge=0, description="P50 延迟（毫秒）")
    p95: float = Field(default=0.0, ge=0, description="P95 延迟（毫秒）")
    p99: float = Field(default=0.0, ge=0, description="P99 延迟（毫秒）")
    max: float = Field(default=0.0, ge=0, description="最大延迟（毫秒）")
    avg: float = Field(default=0.0, ge=0, description="平均延迟（毫秒）")


class ByModeOut(BaseModel):
    """按模式分组的统计（契约 5.9 的 `by_mode`）。

    分模式看才能真正定位问题：agent 比 rag 慢是预期的，
    但如果 `hybrid_rerank` 的 P95 突然等于 `vector` 的 10 倍，那说明重排有问题。
    """

    model_config = ConfigDict(from_attributes=True)

    mode: str = Field(description="模式或名称，如 hybrid_rerank / agent")
    requests: int = Field(default=0, ge=0, description="请求数")
    tokens: int = Field(default=0, ge=0, description="token 合计")
    cost_usd: float = Field(default=0.0, ge=0, description="费用合计（美元）")
    avg_latency_ms: float = Field(default=0.0, ge=0, description="平均延迟（毫秒）")
    refusal_rate: float = Field(default=0.0, ge=0, le=1, description="拒答率 0–1")


class ByModelOut(BaseModel):
    """按模型分组的统计（契约 5.9 的 `by_model`）。用来回答"钱花在哪个模型上"。"""

    model_config = ConfigDict(from_attributes=True)

    model: str = Field(description="模型名，如 deepseek-chat；离线时为说明性取值")
    requests: int = Field(default=0, ge=0, description="请求数")
    prompt_tokens: int = Field(default=0, ge=0, description="输入 token 合计")
    completion_tokens: int = Field(default=0, ge=0, description="输出 token 合计")
    cost_usd: float = Field(default=0.0, ge=0, description="费用合计（美元）")
    avg_latency_ms: float = Field(default=0.0, ge=0, description="平均延迟（毫秒）")


class TimelinePointOut(BaseModel):
    """时间序列上的一个点（契约 5.9 的 `timeline`）。

    `bucket` 是时间桶的起始时刻（UTC ISO8601 带 Z），由服务端按 `hours` 选粒度
    （24 小时按小时、7 天按天）。给起始时刻而不是"第 n 个桶"，
    前端不用知道粒度就能正确画横轴。
    """

    model_config = ConfigDict(from_attributes=True)

    bucket: UtcDatetime = Field(default=None, description="时间桶起点，UTC ISO8601 带 Z")
    requests: int = Field(default=0, ge=0, description="该桶内请求数")
    tokens: int = Field(default=0, ge=0, description="该桶内 token 合计")
    cost_usd: float = Field(default=0.0, ge=0, description="该桶内费用（美元）")
    avg_latency_ms: float = Field(default=0.0, ge=0, description="该桶内平均延迟（毫秒）")
    refusal_rate: float = Field(default=0.0, ge=0, le=1, description="该桶内拒答率")


class ObsStatsOut(BaseModel):
    """观测统计（契约 5.9：请求量 / token / 成本 / 延迟分位 / 按模式与模型分组 / 时间序列）。"""

    model_config = ConfigDict(from_attributes=True)

    hours: int = Field(default=24, ge=1, description="统计窗口（小时），回显以便确认看的哪段时间")
    requests: int = Field(default=0, ge=0, description="窗口内请求总数")
    tokens: TokenStatsOut = Field(default_factory=TokenStatsOut, description="token 用量汇总")
    cost_usd: float = Field(default=0.0, ge=0, description="费用合计（美元）")
    cost_cny: float = Field(default=0.0, ge=0, description="费用合计（人民币），由 USD_TO_CNY 换算")
    latency: LatencyOut = Field(default_factory=LatencyOut, description="延迟分位统计")
    by_mode: list[ByModeOut] = Field(default_factory=list, description="按模式分组")
    by_model: list[ByModelOut] = Field(default_factory=list, description="按模型分组")
    timeline: list[TimelinePointOut] = Field(
        default_factory=list, description="时间序列，按桶起点升序"
    )


class TraceOut(BaseModel):
    """一次请求的 trace 汇总（契约 5.9 的 `Page[TraceOut]`）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="自增主键")
    trace_id: str = Field(description="链路 ID（对外标识）")
    request_id: str | None = Field(
        default=None, description="同一次 HTTP 请求的 ID；可用它把访问日志与 trace 对上"
    )
    user_id: int | None = Field(default=None, description="发起用户 ID")
    conversation_id: int | None = Field(default=None, description="所属会话 ID")
    kb_id: int | None = Field(default=None, description="涉及的知识库 ID")
    name: str = Field(description="trace 名称：chat / search / ingest / eval")
    mode: str | None = Field(default=None, description="本次使用的模式")
    status: str = Field(default="ok", description="状态：ok / error")
    latency_ms: int = Field(default=0, ge=0, description="端到端耗时（毫秒）")
    retrieval_ms: int = Field(default=0, ge=0, description="检索阶段耗时")
    rerank_ms: int = Field(default=0, ge=0, description="重排阶段耗时")
    generate_ms: int = Field(default=0, ge=0, description="生成阶段耗时")
    llm_calls: int = Field(default=0, ge=0, description="大模型调用次数")
    prompt_tokens: int = Field(default=0, ge=0, description="输入 token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 token 数")
    cost_usd: float = Field(default=0.0, ge=0, description="本次费用（美元）")
    retrieval_rounds: int = Field(default=0, ge=0, description="检索轮数")
    source_count: int = Field(default=0, ge=0, description="引用来源条数")
    refusal: bool = Field(default=False, description="是否拒答")
    error: str | None = Field(default=None, description="失败原因")
    created_at: UtcDatetime = Field(default=None, description="发生时间，UTC ISO8601 带 Z")


class TraceSpanOut(BaseModel):
    """trace 内部的一个步骤（契约 5.9 的瀑布图数据）。

    `start_offset_ms` 是**相对 trace 起点**的偏移，不是绝对时间戳：
    前端画瀑布图只需要「偏移 + 时长」两个数字就能定位所有方块，
    不用处理时区与时钟漂移。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="自增主键")
    trace_id: str = Field(description="所属链路 ID")
    seq: int = Field(default=0, ge=0, description="同一 trace 内的递增序号，用于瀑布图排序")
    name: str = Field(
        description="步骤名：analyze/retrieve/grade/rewrite/generate/reflect/bm25/vector/embed_query…"
    )
    span_type: str = Field(description="类型：node / llm / retrieval / db / tool")
    start_offset_ms: int = Field(default=0, ge=0, description="相对 trace 起点的偏移（毫秒）")
    duration_ms: int = Field(default=0, ge=0, description="本步骤耗时（毫秒）")
    status: str = Field(default="ok", description="状态：ok / error")
    input_json: Any | None = Field(default=None, description="输入快照（截断到 2000 字符）")
    output_json: Any | None = Field(default=None, description="输出快照（截断到 2000 字符）")
    error: str | None = Field(default=None, description="失败原因")
    created_at: UtcDatetime = Field(default=None, description="发生时间，UTC ISO8601 带 Z")


class TraceDetailOut(BaseModel):
    """`GET /obs/traces/{trace_id}`（契约 5.9）：`{trace, spans, messages}`。

    把三类数据一次返回，前端进详情页只发一个请求。
    分三次请求会出现"瀑布图有了但消息还没到"的半成品页面。
    """

    model_config = ConfigDict(from_attributes=True)

    trace: TraceOut = Field(description="汇总信息")
    spans: list[TraceSpanOut] = Field(
        default_factory=list, description="步骤列表，按 seq 升序，用于画瀑布图"
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本次涉及的消息（含角色与内容摘要）",
    )


class QualityOut(BaseModel):
    """质量指标（契约 5.9 的 `GET /obs/quality`）。

    这五个值合起来能回答"系统是不是在假装工作"：

    - `refusal_rate` 太高 = 闸门太严或索引太差，用户在不停地被拒；
    - `zero_source_rate` 高 = 经常没检索到东西却在回答（可能已经脱离知识库了）；
    - `reflect_pass_rate` 低 = 生成的内容经常没有检索依据，编造风险高。
    """

    model_config = ConfigDict(from_attributes=True)

    hours: int = Field(default=24, ge=1, description="统计窗口（小时），回显用")
    total: int = Field(default=0, ge=0, description="参与统计的请求数")
    refusal_rate: float = Field(default=0.0, ge=0, le=1, description="拒答率 0–1")
    avg_source_count: float = Field(default=0.0, ge=0, description="平均引用条数")
    zero_source_rate: float = Field(
        default=0.0, ge=0, le=1, description="零引用请求占比 0–1（越高越可疑）"
    )
    avg_retrieval_rounds: float = Field(default=0.0, ge=0, description="平均检索轮数")
    reflect_pass_rate: float = Field(
        default=0.0, ge=0, le=1, description="自省通过率 0–1（越低说明答案越缺依据）"
    )


class DBHealthOut(BaseModel):
    """数据库健康（契约 5.10 的 `db: {ok, dialect, version}`）。"""

    model_config = ConfigDict(from_attributes=True)

    ok: bool = Field(description="数据库是否可用（能执行一次探测查询）")
    dialect: str = Field(default="", description="方言，如 mysql / sqlite")
    version: str = Field(default="", description="数据库版本")
    error: str | None = Field(default=None, description="不可用时的原因，便于直接定位")


class PoolHealthOut(BaseModel):
    """连接池健康（契约 5.10 的 `pool: {...}`）。

    `checkedout` 长期贴着 `size + overflow` 上限，说明有地方没还连接
    （典型是流式响应里持有 session 到生成结束）。只看接口延迟发现不了这个问题。
    """

    model_config = ConfigDict(from_attributes=True)

    size: int = Field(default=0, ge=0, description="池内保持的连接数上限")
    checkedin: int = Field(default=0, ge=0, description="已归还的连接数")
    checkedout: int = Field(default=0, ge=0, description="正在被占用的连接数")
    overflow: int = Field(default=0, description="溢出连接数（可为负）")
    total: int = Field(default=0, ge=0, description="当前池内连接总数")


class HealthOut(BaseModel):
    """`GET /health`（契约 5.10）。

    **这个接口的每个字段都必须来自运行期真实状态**，不允许返回"配置里写了什么"：
    配置说 `local`、实际降级成 `hash` 时，必须报 `hash(fallback:local_load_failed)`。
    探针与前端面板都读它，这里撒一次谎，后面所有判断都是错的。
    """

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(default="ok", description="总体状态：ok / degraded / error")
    version: str = Field(default="", description="应用版本，对应 APP_VERSION")
    uptime_s: float = Field(default=0.0, ge=0, description="进程已运行秒数")
    offline: bool = Field(default=False, description="是否处于离线模式（无可用 API Key）")
    llm_mode: str = Field(default="", description="大模型模式：online / offline")
    embedding_mode: str = Field(
        default="",
        description="向量化实际模式；降级时形如 hash(fallback:local_load_failed)",
    )
    vector_backend: str = Field(default="", description="向量库后端：chroma / memory")
    vector_count: int = Field(default=0, ge=0, description="向量库中的向量总数")
    bm25_doc_count: int = Field(default=0, ge=0, description="BM25 索引覆盖的切片数")
    db: DBHealthOut = Field(description="数据库健康")
    pool: PoolHealthOut = Field(default_factory=PoolHealthOut, description="连接池状态")
    warnings: list[str] = Field(
        default_factory=list,
        description="运行期警告（降级、缺失依赖等），前端要原样显示——这是不骗人的硬规则",
    )
