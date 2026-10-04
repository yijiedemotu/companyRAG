"""确定性哈希向量：零依赖、零网络的离线兜底实现。

**一句话定位**：用 hashing trick 把文本映射成 `EMBEDDING_DIM` 维向量，
保证「同词同向量」，但没有一丝语义能力。

**在整条链路中的位置**：它是 `EMBEDDING_PROVIDER=hash` 时的正式实现，
也是 `local` / `api` 实现失败时 `factory` 的自动降级目标。
它的维度默认 1024，与 BGE-M3 一致，所以降级后**不需要**重建向量库表结构
（但已入库的向量语义会和新查询向量不一致，检索质量会掉，见下方警告）。

**算法（hashing trick / random projection）**：先把文本切成
**1-gram + 2-gram**（切之前丢掉空白与标点），然后对每个 token：

1. `raw = blake2b(token.encode("utf-8"), digest_size=8).digest()` 取 8 字节；
2. `h = int.from_bytes(raw, "big")`，桶 = `h % dim`，符号由 `(h >> 63) & 1` 决定
   （`+1` 或 `-1`）；
3. 把带符号的值累加到对应桶，最后对整条向量做 L2 归一化。

符号位的存在是为了**降低哈希碰撞的破坏力**：两个不同 token 撞进同一个桶时，
有 50% 概率互相抵消而不是叠加放大，这是 random projection 的标准做法。

⚠️ **这是离线兜底，检索质量差。** 它没有语义：问「出差住房能报多少」和
文档里的「住宿标准 600 元」只要没有共同字，相似度就是 0。它存在的**唯一目的**
是让 CI 零成本、零网络跑通全链路（入库 → 检索 → 生成 → 评测 → `/health`），
以及在没有模型、没有 API Key 时链路不中断。**生产环境不要用它。**
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence

from knowflow.core.config import Settings, get_settings
from knowflow.embeddings.base import (
    Embedder,
    build_text_for_embedding,
    l2_normalize,
    zero_vector,
)

# 符号位取最高位：桶用低若干位、符号用最高位，两者互不干扰。
_SIGN_SHIFT = 63
_DIGEST_BYTES = 8


def iter_tokens(text: str) -> Iterator[str]:
    """产出 1-gram + 2-gram token（先丢掉空白与标点）。

    为什么不引 jieba：兜底实现要的是**零依赖 + 确定性**，分词质量对它是次要的；
    而且中文 `split()` 切出来是「一个词」，BM25 与哈希兜底都会直接失效，
    所以这里用字符级 n-gram —— 1-gram 保证不漏，2-gram 提供一点区分度。
    """
    chars = [ch.lower() for ch in text if ch.isalnum()]
    yield from chars
    for i in range(len(chars) - 1):
        yield chars[i] + chars[i + 1]


class HashEmbedder(Embedder):
    """`Embedder` 协议的确定性哈希实现（无模型、无网络、无 numpy）。"""

    provider = "hash"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self.dim = self._settings.embedding_dim
        self.model = f"hash-blake2b-{self.dim}"
        self.display_name = f"hash-blake2b-{self.dim} (offline, no semantics)"

    @property
    def is_loaded(self) -> bool:
        """哈希实现没有模型，永远处于「已就绪」状态。"""
        return True

    def warmup(self) -> None:
        """空实现：没有模型可加载，接口对齐而已（`/health` 会调它）。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码；空文本自然退化成零向量（没有 token 就没有累加）。"""
        return [self._embed_one(build_text_for_embedding(t)) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码查询；与文档侧走**完全相同**的路径（无前缀差异）。"""
        return self._embed_one(build_text_for_embedding(text))

    def _embed_one(self, text: str) -> list[float]:
        """把一条文本累加成一个向量。

        累加用 float 而不是 int：后续要 L2 归一化，用 float 避免一次多余的转换。
        """
        vec = zero_vector(self.dim)
        for token in iter_tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self.dim
            sign = 1.0 if (h >> _SIGN_SHIFT) & 1 == 0 else -1.0
            vec[bucket] += sign
        return l2_normalize(vec)


__all__ = ["HashEmbedder", "iter_tokens"]
