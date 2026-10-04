"""检索层：混合检索（向量 + BM25）→ 融合 → 重排 → 闸门 → 上下文组装。

分层与依赖方向：

    engine.py   编排（唯一对外入口 HybridRetriever）
      ├─ text.py      中文切词（**全项目唯一实现**，BM25 与覆盖率共用）
      ├─ bm25.py      自研 BM25 + 倒排索引 + 按 KB 隔离的注册表
      ├─ fusion.py    min-max 加权 / RRF 两种融合
      ├─ rerank.py    LLM 重排 + 启发式兜底
      ├─ autocut.py   分数断崖截断 + 相关性闸门 + 字符预算
      └─ types.py     Candidate / GateResult / RetrievalResult

这一层**不 import 数据库**（靠 `ChunkEnricher` 协议反向注入），
所以可以脱离 MySQL 单测，跑得又快又稳。
"""

from knowflow.retrieval.autocut import autocut, evaluate_gate, trim_to_budget
from knowflow.retrieval.bm25 import BM25Index, BM25Match, BM25Registry
from knowflow.retrieval.engine import VALID_MODES, HybridRetriever
from knowflow.retrieval.fusion import minmax_normalize, rank_of, rrf_fuse, weighted_fuse
from knowflow.retrieval.rerank import heuristic_rerank, llm_rerank, rerank
from knowflow.retrieval.text import coverage, query_terms, tokenize_zh
from knowflow.retrieval.types import (
    Candidate,
    ChunkEnricher,
    ChunkInfo,
    GateResult,
    RetrievalDebug,
    RetrievalResult,
)

__all__ = [
    "VALID_MODES",
    "BM25Index",
    "BM25Match",
    "BM25Registry",
    "Candidate",
    "ChunkEnricher",
    "ChunkInfo",
    "GateResult",
    "HybridRetriever",
    "RetrievalDebug",
    "RetrievalResult",
    "autocut",
    "coverage",
    "evaluate_gate",
    "heuristic_rerank",
    "llm_rerank",
    "minmax_normalize",
    "query_terms",
    "rank_of",
    "rerank",
    "rrf_fuse",
    "tokenize_zh",
    "trim_to_budget",
    "weighted_fuse",
]
