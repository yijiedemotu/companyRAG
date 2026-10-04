"""检索调试契约（`POST /api/v1/kbs/{kb_id}/search`）。

字段来源：契约文档 5.4 节。

**这个接口不调用大模型**（不花钱），是调参与排障的主力工具，所以请求体里
每一个可调参数都必须能过——少一个参数，用户就只能改 `.env` 重启服务来试参，
调参成本从"点一下"变成"重启一次"。契约 5.4 节列出的字段一个不漏。

两个刻意的设计：

- `vector_threshold` / `keyword_threshold` 默认 `None` = **用服务端配置默认值**
  （`VECTOR_MIN_SCORE` / `KEYWORD_MIN_COVERAGE`）。响应里会在 `gate` 中回显
  **实际生效**的阈值，前端不需要自己去读配置就知道这次用的什么数。
  如果这里写死 `0.35`，改配置时接口默认值不会跟着变，两边就会不一致。
- `SearchHit.vector_score` / `bm25_score` / `rerank_score` / `vector_rank` /
  `bm25_rank` 全部可空：`mode=vector` 时 BM25 根本没跑，给 0 会被误读成
  "跑了但得分是 0"；给 `null` 才是"没参与"。排障时这个区别很关键。

`mode` / `fusion` 用 `Literal`，取值直接从 `core/config.py` 的类型别名取，
不在 schema 里重写字符串字面量。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowflow.core.config import FusionStrategy, RetrievalMode, get_settings

__all__ = [
    "SearchDebug",
    "SearchGate",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
]

# 默认值取服务端配置，保证"接口默认"与"服务端默认"永远是同一个数。
_DEFAULTS = get_settings()


class SearchRequest(BaseModel):
    """检索请求（契约 5.4 节的请求体 JSON 字段全量对应）。"""

    model_config = ConfigDict(from_attributes=True)

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="检索问题，1–4000 字符",
    )
    mode: RetrievalMode = Field(
        default="hybrid_rerank",
        description="检索模式：vector（纯向量）/ bm25（纯关键词）/ hybrid（混合）/ hybrid_rerank（混合+重排）",
    )
    top_k: int = Field(
        default=_DEFAULTS.top_k,
        ge=1,
        le=50,
        description="最终返回条数，1–50",
    )
    fetch_k: int = Field(
        default=_DEFAULTS.fetch_k,
        ge=1,
        le=500,
        description="每路召回的候选数，必须 ≥ top_k（先多召回再筛选）",
    )
    fusion: FusionStrategy = Field(
        default=_DEFAULTS.fusion,
        description="混合融合策略：rrf（倒数排名融合）或 weighted（加权求和），仅 hybrid/hybrid_rerank 生效",
    )
    alpha: float = Field(
        default=_DEFAULTS.alpha,
        ge=0,
        le=1,
        description="weighted 融合时的向量权重，0=全关键词、1=全向量；仅 fusion=weighted 生效",
    )
    rerank: bool = Field(
        default=_DEFAULTS.rerank_enabled,
        description="是否重排，仅 hybrid_rerank 生效",
    )
    use_autocut: bool = Field(
        default=_DEFAULTS.autocut_enabled,
        description="是否启用分数断崖截断（把明显低分的尾部候选砍掉）",
    )
    vector_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="向量相似度闸门下限，null = 用服务端配置 VECTOR_MIN_SCORE",
    )
    keyword_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="关键词覆盖率闸门下限，null = 用服务端配置 KEYWORD_MIN_COVERAGE",
    )
    include_debug: bool = Field(
        default=True,
        description="是否返回 debug 统计（各阶段候选数），排障用，关掉可省一点序列化开销",
    )

    @model_validator(mode="after")
    def _check_fetch_k(self) -> SearchRequest:
        if self.fetch_k < self.top_k:
            raise ValueError(
                f"fetch_k({self.fetch_k}) 不能小于 top_k({self.top_k})："
                "先召回才有得筛，候选数少于目标条数时结果必然不足"
            )
        return self


class SearchHit(BaseModel):
    """一条命中。各路分数与排名**独立保留**，这是排障的全部价值所在。"""

    model_config = ConfigDict(from_attributes=True)

    rank: int = Field(ge=1, description="最终排名，从 1 开始（对应正文里的 [n]）")
    chunk_id: int = Field(description="切片 ID")
    doc_id: int = Field(description="所属文档 ID")
    doc_name: str = Field(description="文档文件名，前端直接展示，无需再查文档接口")
    page_no: int | None = Field(default=None, description="页码，仅 PDF 有值")
    section_path: str | None = Field(default=None, description="所属小节路径")
    content: str = Field(default="", description="切片完整正文")
    snippet: str = Field(default="", description="用于展示的摘要片段（最长 512 字符）")
    score: float = Field(description="最终得分，越大越相似（已统一为「越大越好」）")
    vector_score: float | None = Field(default=None, description="向量相似度；该路未参与时为 null")
    bm25_score: float | None = Field(default=None, description="BM25 原始分；该路未参与时为 null")
    vector_rank: int | None = Field(default=None, description="在向量召回中的排名")
    bm25_rank: int | None = Field(default=None, description="在 BM25 召回中的排名")
    rerank_score: float | None = Field(default=None, description="重排模型给出的分数")


class SearchGate(BaseModel):
    """相关度闸门结果。**没通过闸门时不编答案，直接拒答**，所以要把判断依据暴露出来。"""

    model_config = ConfigDict(from_attributes=True)

    passed: bool = Field(description="是否通过闸门；false 表示知识库里没有相关内容")
    reason: str = Field(
        default="",
        description="判定原因，如 vector_score_above_threshold / no_candidates",
    )
    vector_threshold: float = Field(
        description="本次实际生效的向量阈值（请求未传时为服务端配置默认值）"
    )
    keyword_threshold: float = Field(description="本次实际生效的关键词覆盖率阈值")
    best_vector_score: float | None = Field(
        default=None, description="候选中的最佳向量相似度，用于对照阈值看差多少"
    )
    best_keyword_coverage: float | None = Field(
        default=None, description="候选中的最佳关键词覆盖率"
    )


class SearchDebug(BaseModel):
    """各阶段候选数与降级信息。用来定位"结果为什么少"：召回不足，还是融合/截断砍太多。

    字段与 `retrieval.types.RetrievalDebug` **一一对应**（后者是 dataclass，
    这里是对外契约）。以前这里少了 5 个字段，结果是 `/search` 响应把
    `vector_error` 这类"这次是降级跑的"关键信息静默丢掉了——
    Pydantic 默认忽略未知字段，不报错，只是前端永远看不到。
    任何给 `RetrievalDebug` 新增的字段都必须同步加到这里。
    """

    model_config = ConfigDict(from_attributes=True)

    vector_candidates: int = Field(default=0, ge=0, description="向量召回条数")
    bm25_candidates: int = Field(default=0, ge=0, description="BM25 召回条数")
    fused: int = Field(default=0, ge=0, description="融合去重后的候选数")
    after_autocut: int = Field(default=0, ge=0, description="autocut 截断后剩余数")
    after_rerank: int = Field(default=0, ge=0, description="重排后剩余条数")
    dropped_by_gate: int = Field(default=0, ge=0, description="被相关度闸门丢弃的条数")
    dropped_by_budget: int = Field(default=0, ge=0, description="因上下文预算不足被丢弃的条数")
    dropped_orphans: int = Field(
        default=0,
        ge=0,
        description="孤儿向量条数（向量库里有、MySQL 里已无对应 chunk），必须丢弃否则引用会 404",
    )
    vector_error: str | None = Field(
        default=None,
        description="向量召回失败原因；**非 null 即表示本次已降级为纯 BM25**，前端要明确提示",
    )
    kb_ids: list[int] = Field(
        default_factory=list,
        description="本次检索实际用到的知识库 ID 列表（多库合并时用）",
    )


class SearchResponse(BaseModel):
    """检索响应（契约 5.4 节的响应 JSON）。

    各阶段耗时单独返回（embedding / vector / bm25 / rerank），而不是只给总数：
    检索慢的时候，一眼能看出是向量化慢（模型加载）、向量库慢还是重排慢，
    只看总耗时要靠猜。
    """

    model_config = ConfigDict(from_attributes=True)

    query: str = Field(description="实际用于检索的查询串（若做过改写则为改写后的）")
    mode: RetrievalMode = Field(description="本次实际执行的检索模式")
    latency_ms: int = Field(ge=0, description="端到端耗时（毫秒）")
    embedding_ms: int = Field(default=0, ge=0, description="查询向量化耗时")
    vector_ms: int = Field(default=0, ge=0, description="向量检索耗时")
    bm25_ms: int = Field(default=0, ge=0, description="BM25 检索耗时")
    rerank_ms: int = Field(default=0, ge=0, description="重排耗时")
    hits: list[SearchHit] = Field(default_factory=list, description="命中列表，按 rank 升序")
    gate: SearchGate = Field(description="相关度闸门结果")
    debug: SearchDebug | None = Field(
        default=None, description="各阶段候选数统计；include_debug=false 时为 null"
    )
