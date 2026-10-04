"""混合检索引擎：整条在线链路的心脏。

**一次 `search()` 的七个阶段（每个阶段都单独计时，落进 trace span）**

    ① embed_query      问题 → 1024 维向量        （本地 BGE-M3，约 60~200ms）
    ② vector recall    向量库取 fetch_k 条        （Chroma，约 10~80ms）
    ③ bm25 recall      倒排索引取 fetch_k 条      （纯内存，约 1~10ms）
    ④ fuse             两路融合（RRF / 加权）     （<1ms）
    ⑤ normalize+cut    归一化 → 分数断崖截断      （<1ms）
    ⑥ rerank           重排（LLM 或启发式）        （0ms 或 500~2000ms）
    ⑦ gate+budget      相关性闸门 → 字符预算裁剪   （<1ms）

**为什么把"融合"和"归一化"分成两步**：融合（RRF）输出的分数没有物理含义
（只是 `Σ 1/(k+rank)`），不能直接和阈值比较；必须先归一化到 [0,1]，
autocut 的 `ratio` 与 UI 展示的 `score` 才有可解释的含义。

**为什么向量召回失败要降级而不是报错**：BM25 仍然能工作。
一次"只用关键词"的检索远好于一个 502 —— 但**必须在 `debug.vector_error` 里
如实记录**，让 `/health`、trace 和前端都能看到"这次检索是降级的"。
降级可以，静默不行。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import EmbeddingError, VectorStoreError
from knowflow.core.logging import get_logger
from knowflow.embeddings.base import Embedder, EmbeddingError as EmbedderFailure
from knowflow.llm.base import ChatModel
from knowflow.retrieval.autocut import autocut, evaluate_gate, trim_to_budget
from knowflow.retrieval.bm25 import BM25Registry
from knowflow.retrieval.fusion import minmax_normalize, rank_of, rrf_fuse, weighted_fuse
from knowflow.retrieval.rerank import rerank as rerank_candidates
from knowflow.retrieval.text import coverage
from knowflow.retrieval.types import (
    Candidate,
    ChunkEnricher,
    RetrievalDebug,
    RetrievalResult,
)
from knowflow.vectorstore.base import VectorStore, VectorStoreError as VectorStoreFailure

logger = get_logger(__name__)

VALID_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _failure_reason(exc: BaseException) -> str:
    """把两种异常体系的原因都转成一行文本。

    项目里有两套并存的异常类（历史原因）：
    - `core.exceptions.EmbeddingError` / `VectorStoreError`（`KnowFlowError` 子类，有 `.message`）；
    - `embeddings.base.EmbeddingError` / `vectorstore.base.VectorStoreError`（普通 Exception）。

    **实现方抛的是后者，而降级路径原来只捕获前者** —— 结果是"向量召回失败降级为纯 BM25"
    这条承诺在最需要它的时候（模型加载失败、向量库维度不符）根本不生效，异常会直接
    冒到 HTTP 层。这里把两套都接住，并统一取原因文本（`.message` 可能不存在）。
    """
    return str(getattr(exc, "message", None) or exc)


class HybridRetriever:
    """向量 + BM25 混合检索器。

    依赖注入而不是自己 new：这样单测可以塞假的 embedder / 假的向量库，
    完全不碰磁盘与网络（测试能在 7 秒内跑完 100+ 用例的关键）。
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Registry,
        chat_model: ChatModel | None = None,
        enricher: ChunkEnricher | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.chat_model = chat_model
        self.enricher = enricher

    # ------------------------------------------------------------------ 内部
    def _vector_candidates(
        self,
        *,
        query: str,
        kb_ids: Sequence[int],
        fetch_k: int,
        where: dict[str, Any] | None,
        debug: RetrievalDebug,
    ) -> tuple[list[Candidate], int]:
        """向量召回。返回 `(候选, embedding 耗时ms)`。"""
        started = time.perf_counter()
        try:
            vector = self.embedder.embed_query(query)
        except (EmbeddingError, EmbedderFailure) as exc:
            debug.vector_error = f"embedding: {_failure_reason(exc)}"
            logger.warning("retrieval.embed_failed", error=_failure_reason(exc))
            return [], _ms(started)
        embedding_ms = _ms(started)

        if not kb_ids:
            # 没有指定 KB 时不做向量检索：向量库按 KB 分集合，没有目标就无从查起。
            # 记录原因而不是静默跳过——否则"为什么这次检索是纯 BM25"会变成悬案。
            debug.vector_error = "no kb specified; vector recall skipped"
            return [], embedding_ms

        merged: list[Candidate] = []
        for kb_id in kb_ids:
            try:
                hits = self.vector_store.query(
                    kb_id=kb_id, vector=vector, top_k=fetch_k, where=where
                )
            except (VectorStoreError, VectorStoreFailure) as exc:
                debug.vector_error = f"vector_store: {_failure_reason(exc)}"
                logger.warning("retrieval.vector_failed", kb_id=kb_id, error=_failure_reason(exc))
                continue
            for hit in hits:
                meta = dict(hit.metadata or {})
                merged.append(
                    Candidate(
                        vector_id=str(hit.id),
                        doc_id=int(meta.get("doc_id") or 0),
                        doc_name=str(meta.get("doc_name") or ""),
                        chunk_id=_safe_int(meta.get("chunk_id")),
                        chunk_index=int(meta.get("chunk_index") or 0),
                        content=hit.content or "",
                        page_no=_safe_int(meta.get("page_no")),
                        section_path=_opt_str(meta.get("section_path")),
                        ext=_opt_str(meta.get("ext")),
                        vector_score=float(hit.score),
                    )
                )

        # 多 KB 合并：按向量分排序取 fetch_k
        merged.sort(key=lambda c: (-(c.vector_score or 0.0), c.vector_id))
        merged = merged[:fetch_k]
        ranks = rank_of({c.vector_id: c.vector_score or 0.0 for c in merged})
        for candidate in merged:
            candidate.vector_rank = ranks.get(candidate.vector_id)
        return merged, embedding_ms

    def _bm25_candidates(
        self, *, query: str, kb_ids: Sequence[int], fetch_k: int, debug: RetrievalDebug
    ) -> list[Candidate]:
        """关键词召回。`kb_ids` 为空时跨所有 KB 搜索。"""
        candidates: list[Candidate] = []

        if not kb_ids:
            rows = self.bm25.search(kb_id=None, query=query, top_k=fetch_k)
        else:
            rows = []
            for kb_id in kb_ids:
                rows.extend(self.bm25.search(kb_id=kb_id, query=query, top_k=fetch_k))
            rows.sort(key=lambda item: -item["score"])
            rows = rows[:fetch_k]

        for row in rows:
            payload = row.get("payload") or {}
            candidates.append(
                Candidate(
                    vector_id=str(row["vector_id"]),
                    doc_id=int(payload.get("doc_id") or 0),
                    doc_name=str(payload.get("doc_name") or ""),
                    chunk_id=_safe_int(payload.get("chunk_id")),
                    chunk_index=int(payload.get("chunk_index") or 0),
                    content=str(payload.get("content") or ""),
                    context_text=str(payload.get("parent_content") or payload.get("content") or ""),
                    page_no=_safe_int(payload.get("page_no")),
                    section_path=_opt_str(payload.get("section_path")),
                    ext=_opt_str(payload.get("ext")),
                    bm25_score=float(row["score"]),
                )
            )

        ranks = rank_of({c.vector_id: c.bm25_score or 0.0 for c in candidates})
        for candidate in candidates:
            candidate.bm25_rank = ranks.get(candidate.vector_id)
        return candidates

    def _enrich(self, candidates: list[Candidate], debug: RetrievalDebug) -> list[Candidate]:
        """用 MySQL 的权威数据覆盖候选的正文与出处，并丢掉孤儿向量。"""
        if self.enricher is None or not candidates:
            return candidates

        by_id = self.enricher.fetch([c.vector_id for c in candidates])
        enriched: list[Candidate] = []
        for candidate in candidates:
            info = by_id.get(candidate.vector_id)
            if info is None:
                debug.dropped_orphans += 1
                continue
            candidate.chunk_id = info.chunk_id
            candidate.doc_id = info.doc_id
            candidate.doc_name = info.doc_name
            candidate.chunk_index = info.chunk_index
            candidate.content = info.content
            candidate.context_text = info.context_text
            candidate.page_no = info.page_no
            candidate.section_path = info.section_path
            candidate.ext = info.ext
            enriched.append(candidate)
        return enriched

    @staticmethod
    def _merge_by_id(
        vector_candidates: Sequence[Candidate], bm25_candidates: Sequence[Candidate]
    ) -> dict[str, Candidate]:
        """两路结果按 `vector_id` 合并成同一批候选（各自的分数字段都保留）。

        **不能丢任何一路独有的候选**：只在 BM25 里命中的（专有名词）与只在
        向量里命中的（同义改写）都是我们想要的，融合阶段会决定谁排前面。
        """
        merged: dict[str, Candidate] = {}
        for candidate in vector_candidates:
            merged[candidate.vector_id] = candidate
        for candidate in bm25_candidates:
            existing = merged.get(candidate.vector_id)
            if existing is None:
                merged[candidate.vector_id] = candidate
            else:
                existing.bm25_score = candidate.bm25_score
                existing.bm25_rank = candidate.bm25_rank
                # 向量路可能没带出父块（metadata 里不放长文本），BM25 的 payload 有，补上
                if not existing.context_text and candidate.context_text:
                    existing.context_text = candidate.context_text
                if not existing.content and candidate.content:
                    existing.content = candidate.content
        return merged

    # ------------------------------------------------------------------ 对外
    def search(
        self,
        *,
        query: str,
        kb_id: int | None = None,
        kb_ids: Sequence[int] | None = None,
        mode: str | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
        fusion: str | None = None,
        alpha: float | None = None,
        use_rerank: bool | None = None,
        use_autocut: bool | None = None,
        vector_threshold: float | None = None,
        keyword_threshold: float | None = None,
        max_context_chars: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        cfg = self.settings
        started_all = time.perf_counter()

        # ---- 参数归一化（显式入参 > 配置默认值）----
        target_kbs: list[int] = list(kb_ids) if kb_ids else ([kb_id] if kb_id else [])
        resolved_mode = (mode or ("hybrid_rerank" if cfg.rerank_enabled else "hybrid")).lower()
        if resolved_mode not in VALID_MODES:
            resolved_mode = "hybrid"
        k = top_k or cfg.top_k
        fk = fetch_k or max(cfg.fetch_k, k)
        fusion_strategy = (fusion or cfg.fusion).lower()
        alpha_value = cfg.alpha if alpha is None else alpha
        do_rerank = cfg.rerank_enabled if use_rerank is None else use_rerank
        do_autocut = cfg.autocut_enabled if use_autocut is None else use_autocut
        v_threshold = cfg.vector_min_score if vector_threshold is None else vector_threshold
        k_threshold = cfg.keyword_min_coverage if keyword_threshold is None else keyword_threshold
        budget = max_context_chars or cfg.max_context_chars

        debug = RetrievalDebug(kb_ids=target_kbs)

        # ---- ① ② 向量召回 ----
        vector_candidates: list[Candidate] = []
        embedding_ms = 0
        vector_ms = 0

        if resolved_mode in ("vector", "hybrid", "hybrid_rerank"):
            started = time.perf_counter()
            vector_candidates, embedding_ms = self._vector_candidates(
                query=query, kb_ids=target_kbs, fetch_k=fk, where=where, debug=debug
            )
            vector_ms = _ms(started) - embedding_ms
            debug.vector_candidates = len(vector_candidates)
        else:
            # 纯 bm25 模式也要把向量分留空，闸门会只看覆盖率
            debug.vector_error = None

        # ---- ③ BM25 召回 ----
        bm25_candidates: list[Candidate] = []
        bm25_ms = 0
        if resolved_mode in ("bm25", "hybrid", "hybrid_rerank"):
            started = time.perf_counter()
            bm25_candidates = self._bm25_candidates(
                query=query, kb_ids=target_kbs, fetch_k=fk, debug=debug
            )
            bm25_ms = _ms(started)
            debug.bm25_candidates = len(bm25_candidates)

        # ---- 选择候选集合 ----
        if resolved_mode == "vector":
            candidates = list(vector_candidates)
        elif resolved_mode == "bm25":
            candidates = list(bm25_candidates)
        else:
            candidates = list(self._merge_by_id(vector_candidates, bm25_candidates).values())

        # ---- 回查 MySQL，拿到权威正文与出处（并丢掉孤儿）----
        candidates = self._enrich(candidates, debug)

        # ---- 关键词覆盖率（闸门与启发式重排都要用）----
        for candidate in candidates:
            candidate.keyword_coverage = coverage(query, candidate.content)

        # ---- ④ 融合 ----
        started = time.perf_counter()
        if resolved_mode in ("hybrid", "hybrid_rerank") and candidates:
            vector_scores = {
                c.vector_id: c.vector_score for c in candidates if c.vector_score is not None
            }
            bm25_scores = {
                c.vector_id: c.bm25_score for c in candidates if c.bm25_score is not None
            }
            if fusion_strategy == "weighted":
                fused = weighted_fuse(vector_scores, bm25_scores, alpha=alpha_value)
            else:
                vector_order = [
                    c.vector_id
                    for c in sorted(
                        (c for c in candidates if c.vector_score is not None),
                        key=lambda c: -(c.vector_score or 0.0),
                    )
                ]
                bm25_order = [
                    c.vector_id
                    for c in sorted(
                        (c for c in candidates if c.bm25_score is not None),
                        key=lambda c: -(c.bm25_score or 0.0),
                    )
                ]
                fused = rrf_fuse([vector_order, bm25_order], k=cfg.rrf_k)
        else:
            # 单路模式：融合分就用那一路的分数（归一化在下一步做）
            fused = {
                c.vector_id: (
                    c.vector_score if resolved_mode == "vector" else (c.bm25_score or 0.0)
                )
                or 0.0
                for c in candidates
            }

        normalized = minmax_normalize(fused)
        for candidate in candidates:
            candidate.fused_score = round(fused.get(candidate.vector_id, 0.0), 8)
            candidate.score = round(normalized.get(candidate.vector_id, 0.0), 6)
        debug.fused = len(candidates)
        fusion_ms = _ms(started)

        # ---- 排序 + ⑤ autocut ----
        candidates.sort(key=lambda c: (-c.score, c.vector_id))
        if do_autocut:
            # autocut 的 max_keep 用 fetch_k 而不是 top_k：
            # 重排需要在更大的候选池里挑，过早压到 top_k 会让重排失去意义
            candidates = autocut(candidates, ratio=cfg.autocut_ratio, min_keep=1, max_keep=fk)
        debug.after_autocut = len(candidates)

        # ---- ⑥ 重排 ----
        rerank_ms = 0
        rerank_meta: dict[str, Any] = {"reranked": False, "strategy": "none"}
        if resolved_mode == "hybrid_rerank" and do_rerank:
            started = time.perf_counter()
            candidates, rerank_meta = rerank_candidates(
                query,
                candidates,
                model=self.chat_model,
                top_n=cfg.rerank_top_n,
                enabled=True,
                settings=cfg,
            )
            rerank_ms = _ms(started)
        debug.after_rerank = len(candidates)

        # ---- ⑦ 闸门 + 预算 ----
        gate = evaluate_gate(
            candidates, vector_threshold=v_threshold, keyword_threshold=k_threshold
        )
        if not gate.passed:
            debug.dropped_by_gate = len(candidates)
            candidates = []
        else:
            candidates = candidates[:k]
            candidates, dropped = trim_to_budget(candidates, max_chars=budget)
            debug.dropped_by_budget = dropped

        total_ms = _ms(started_all)
        logger.debug(
            "retrieval.done",
            mode=resolved_mode,
            fusion=fusion_strategy,
            candidates=len(candidates),
            gate=gate.reason,
            total_ms=total_ms,
            rerank=rerank_meta.get("strategy"),
            vector_error=debug.vector_error,
        )

        return RetrievalResult(
            query=query,
            mode=resolved_mode,
            candidates=candidates,
            gate=gate,
            debug=debug,
            embedding_ms=embedding_ms,
            vector_ms=max(vector_ms, 0),
            bm25_ms=bm25_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
        )

    def retrieve(
        self, query: str, *, kb_id: int | None = None, top_k: int | None = None
    ) -> list[Candidate]:
        """给 Agent 工具用的简化入口：只关心"有没有捞到料"。"""
        result = self.search(query=query, kb_id=kb_id, top_k=top_k)
        return result.candidates if result.passed_gate else []


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["VALID_MODES", "HybridRetriever"]
