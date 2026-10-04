"""KnowFlow 向量化层：`Embedder` 协议 + 三种实现（local / hash / api）。

对外只需两个入口：

- `build_embedder(settings)`：造出带自动降级的实现，进程内共用一个实例；
- `get_embedder_status(embedder)`：读真实运行状态，给 `/health` 如实上报用。

**导入很轻是刻意的**：本包不 import `sentence_transformers`、不加载模型，
只有 `ChromaVectorStore` 那种才会在导入时引入重依赖。所以
`import knowflow.embeddings` 不会让进程启动慢下来，也不会在离线 CI 里
因为缺模型而失败。
"""

from __future__ import annotations

from knowflow.embeddings.api_embedder import ApiEmbedder
from knowflow.embeddings.base import (
    BGE_ZH_QUERY_INSTRUCTION,
    Embedder,
    EmbeddingError,
    build_text_for_embedding,
    l2_normalize,
    metadata_of_embedder,
    warmup_embedder,
    zero_vector,
)
from knowflow.embeddings.factory import (
    EmbedderStatus,
    build_embedder,
    get_embedder_status,
)
from knowflow.embeddings.hash_embedder import HashEmbedder
from knowflow.embeddings.local_bge import LocalBgeEmbedder, sanitize_model_name

__all__ = [
    "BGE_ZH_QUERY_INSTRUCTION",
    "ApiEmbedder",
    "Embedder",
    "EmbedderStatus",
    "EmbeddingError",
    "HashEmbedder",
    "LocalBgeEmbedder",
    "build_embedder",
    "build_text_for_embedding",
    "get_embedder_status",
    "l2_normalize",
    "metadata_of_embedder",
    "sanitize_model_name",
    "warmup_embedder",
    "zero_vector",
]
