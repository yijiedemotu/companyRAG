"""检索调试路由（契约 5.4）：`POST /api/v1/kbs/{kb_id}/search`。

**一句话定位**：**不调用大模型、不花钱**的调参与排障入口。

**在链路中的位置**：前端 `/search` 页面 -> **本模块** -> `ChatService.search` ->
`HybridRetriever`（向量 + BM25 + 融合 + 重排 + 闸门）。

**关键设计取舍**：

1. **响应逐字段照抄契约 5.4，包含 `debug.vector_error` 与 `debug.kb_ids`**。
   少了 `vector_error`，前端就无法提示"这次是降级成纯 BM25 跑的"，
   用户会把降级后的差结果当成正常检索质量来评价——这是最坏的一种误导。
2. **`include_debug=false` 时 `debug` 必须回 `null`**（不是空对象）：
   给 `{}` 会让前端以为"统计了但全是 0"，那与"没统计"是完全不同的结论。
3. **各阶段耗时分字段返回**（embedding / vector / bm25 / rerank）而不是只给总数：
   检索慢的时候要能一眼看出是向量化慢、向量库慢还是重排慢，只看总耗时要靠猜。
4. **`mode` / `fusion` 的取值来自 `core/config.py` 的字面量**：请求体已经过 Pydantic
   校验，service 层接受 `str`，所以这里不需要再做一次转换。
"""

from __future__ import annotations

from fastapi import APIRouter

from knowflow.api.deps import Services
from knowflow.schemas.search import (
    SearchDebug,
    SearchGate,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

__all__ = ["router"]

router = APIRouter(tags=["检索"])


@router.post(
    "/kbs/{kb_id}/search",
    response_model=SearchResponse,
    summary="检索调试（不调用大模型）",
)
def search_kb(kb_id: int, payload: SearchRequest, services: Services) -> SearchResponse:
    """按请求参数跑一次检索并返回完整中间信号。

    `vector_threshold` / `keyword_threshold` 为 `None` 时用服务端配置默认值，
    响应里的 `gate` 会**回显实际生效的阈值**，前端不必自己去读配置。
    """
    result = services.chat_service.search(
        kb_id=kb_id,
        query=payload.query,
        mode=payload.mode,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        fusion=payload.fusion,
        alpha=payload.alpha,
        use_rerank=payload.rerank,
        use_autocut=payload.use_autocut,
        vector_threshold=payload.vector_threshold,
        keyword_threshold=payload.keyword_threshold,
    )

    gate = _gate_of(result.gate)
    debug = SearchDebug.model_validate(result.debug.to_dict()) if payload.include_debug else None

    return SearchResponse(
        query=result.query,
        mode=result.mode,  # type: ignore[arg-type]  # 引擎保证取值属于 Literal 集合
        latency_ms=result.total_ms,
        embedding_ms=result.embedding_ms,
        vector_ms=result.vector_ms,
        bm25_ms=result.bm25_ms,
        rerank_ms=result.rerank_ms,
        hits=[SearchHit.model_validate(hit) for hit in result.hits()],
        gate=gate,
        debug=debug,
    )


def _gate_of(gate: object) -> SearchGate:
    """把 `retrieval.types.GateResult` 转成契约模型。

    引擎一定会给 gate，但契约里 `gate` 是必填字段：
    宁可显式回一个"未通过"的默认值，也不要让响应 500
    （这个接口的用途就是排障，自己在排障时 500 是最糟的体验）。
    """
    to_dict = getattr(gate, "to_dict", None)
    if callable(to_dict):
        return SearchGate.model_validate(to_dict())
    return SearchGate(
        passed=False,
        reason="gate_unavailable",
        vector_threshold=0.0,
        keyword_threshold=0.0,
    )
