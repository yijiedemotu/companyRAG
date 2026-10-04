"""向量化抽象层：`Embedder` 协议与共用工具。

**一句话定位**：把「文本 → 向量」收敛成一个协议，上层只依赖协议，
不依赖具体供应商（本地 BGE / hash 兜底 / OpenAI 兼容 API）。

**在整条链路中的位置**：入库（ingest）用 `embed_documents` 编码切分后的块，
检索（retrieve）用 `embed_query` 编码用户问题；两侧必须来自同一个 embedder，
否则同一句话在两侧落在不同的语义空间，检索会**静默**变差（不报错、只是不准）。
`factory.build_embedder` 负责选实现与降级，`/health` 通过
`factory.get_embedder_status` 读取真实运行模式。

**关键设计取舍**：
1. 协议只要求 4 个属性 + 2 个方法，且不要求继承 —— 单测里 20 行就能写个假实现。
2. `warmup()` / `is_loaded` **不进协议**：hash 兜底没有模型可加载，写进协议会逼
   所有实现造假动作；需要预热的调用方统一用 `warmup_embedder()` 做「有则调用」。
3. 空文本/纯空白一律返回**零向量**而不是抛异常：入库时难免切出空白块，
   为它抛异常会让整篇文档入库失败；而零向量与任何文本的余弦相似度都是 0，
   过不了 `VECTOR_MIN_SCORE` 闸门，所以既安全又不会污染检索结果。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, Protocol

import structlog

logger = structlog.get_logger(__name__)

# 如果哪天把 EMBEDDING_MODEL 换成 BAAI/bge-large-zh-v1.5，必须在
# build_text_for_embedding 里把它加到**查询侧**，见该函数的 docstring。
BGE_ZH_QUERY_INSTRUCTION: Final[str] = "为这个句子生成表示以用于检索相关文章："


class EmbeddingError(Exception):
    """向量化失败（加载模型失败 / 推理失败 / 远端接口报错 / 维度不匹配）。

    上层（服务层）捕获它并转成 HTTP 502 `UPSTREAM_ERROR`，
    所以消息里要带够排查信息：模型名、来源、状态码、响应体片段。
    """


class Embedder(Protocol):
    """文本向量化器的统一契约。

    实现必须满足三条约定，缺一条都会在检索端变成难查的问题：

    1. `embed_documents` 与 `embed_query` 返回**同维度**且已 **L2 归一化**的向量
       （归一化后点积 == 余弦相似度，阈值 `VECTOR_MIN_SCORE` 才有稳定含义）；
    2. 空文本返回**零向量**，不抛异常；
    3. `dim` 必须等于向量库集合的维度 —— 换模型必须重建向量库。
    """

    provider: str
    """`local` | `hash` | `api`。"""

    model: str
    """模型标识，如 `BAAI/bge-m3`。"""

    dim: int
    """向量维度，必须等于 `EMBEDDING_DIM`。"""

    display_name: str
    """给人看的名字，如 `BAAI/bge-m3 (local, cpu)`，进 `/health` 与启动日志。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档侧文本（入库用）。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """编码查询侧文本（检索用）。"""
        ...


def l2_normalize(vec: Sequence[float]) -> list[float]:
    """L2 归一化：返回 `v / ||v||`。

    为什么必须归一化：Chroma 用 `cosine` 空间，余弦相似度的分母就是两个向量的
    模长；提前归一化后「点积 == 余弦相似度」，阈值才有稳定含义，也才能和 BM25
    的分数一起做融合。

    为什么是纯 python 而不是 numpy：本函数会被单测反复调用，且**不能**让 numpy
    成为 embeddings 的硬依赖 —— 只想跑 hash 兜底的离线环境也要能 import 本模块。

    零向量（`||v|| == 0`，空文本的产物）原样返回零向量：除以 0 会得到 NaN，
    而 NaN 一旦写进向量库，之后**所有**距离计算都会是 NaN，且不会有任何报错。
    """
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [float(x) / norm for x in vec]


def build_text_for_embedding(text: str) -> str:
    """所有文本进入向量模型前的**唯一入口**（当前实现：只裁掉两端空白）。

    当前项目用 `BAAI/bge-m3`，官方说明它**不需要**查询指令前缀，
    所以文档侧与查询侧走同一条路，本函数目前是对称的。

    ⚠️ 如果换成 `BAAI/bge-large-zh-v1.5` 这类模型，**必须只在本函数里改**：
    查询侧要加前缀 `为这个句子生成表示以用于检索相关文章：`（常量
    `BGE_ZH_QUERY_INSTRUCTION`），文档侧不加。届时签名要扩成
    `build_text_for_embedding(text, *, is_query: bool)`，并同步改两个调用点
    （文档侧传 False、查询侧传 True）。之所以强调「唯一入口」：加了前缀却漏改
    文档侧（或反之）不会报任何错，只会让召回率悄悄掉一截。
    """
    return text.strip()


def warmup_embedder(embedder: Embedder) -> bool:
    """尽力预热：实现里有 `warmup()` 就调用它。

    返回 True 表示预热调用没有抛异常。注意**降级包装器**吞掉异常后也会返回 True，
    所以调用方预热完要再读一次 `get_embedder_status(embedder).degraded`
    才知道「模型到底是不是真的加载成功」。
    """
    warmup = getattr(embedder, "warmup", None)
    if not callable(warmup):
        return False
    try:
        warmup()
    except Exception as exc:  # noqa: BLE001 - 预热失败不该让启动流程炸掉
        logger.warning(
            "向量化实现预热失败",
            provider=getattr(embedder, "provider", "unknown"),
            error=f"{type(exc).__name__}: {exc}",
        )
        return False
    return True


def zero_vector(dim: int) -> list[float]:
    """空文本的规范产物：全 0 向量（长度必须等于 `dim`）。"""
    return [0.0] * dim


def metadata_of_embedder(embedder: Embedder) -> dict[str, Any]:
    """给日志/`/health` 用的轻量描述（不含任何 Key）。"""
    return {
        "provider": embedder.provider,
        "model": embedder.model,
        "dim": embedder.dim,
        "display_name": embedder.display_name,
    }


__all__ = [
    "BGE_ZH_QUERY_INSTRUCTION",
    "Embedder",
    "EmbeddingError",
    "build_text_for_embedding",
    "l2_normalize",
    "metadata_of_embedder",
    "warmup_embedder",
    "zero_vector",
]
