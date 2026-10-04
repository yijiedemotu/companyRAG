"""自研 BM25：倒排索引 + 中文 n-gram。

**BM25 公式（白板可推导，面试重点）**

    对查询 Q 中的每个词 t，对每篇文档 D：

        score(D, Q) = Σ_t  IDF(t) · ──────────────────────────────
                                    tf(t,D) + k1 · (1 - b + b · |D|/avgdl)

        IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

    其中：
      tf(t,D)  词 t 在文档 D 中出现的次数
      df(t)    包含词 t 的文档数
      N        文档总数
      |D|      文档 D 的长度（词元数）
      avgdl    平均文档长度
      k1=1.5   词频饱和系数：词出现 100 次不等于相关 100 倍，收益递减
      b=0.75   长度归一强度：长文档天然更容易命中，要惩罚

**为什么 IDF 里要 `+1`**：经典 BM25 的 IDF 在 `df > N/2` 时会变成负数，
导致"包含该词的文档反而被扣分"。加 1 后恒为正（这是 Lucene 采用的形式）。
**这是一个真实踩过的坑**：早期版本忘了 +1，高频词（如"公司"）一出现就把
相关文档排到最后，表现是"越常见的关键词越查不到"。

**为什么不用第三方库**：见 `retrieval/text.py` 的说明。核心是
"这套打分逻辑面试能讲到底层"，引库会把这个优势抹掉。
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from knowflow.core.logging import get_logger
from knowflow.retrieval.text import tokenize_zh

logger = get_logger(__name__)

# 默认参数（Lucene / Elasticsearch 的默认值，工业界验证过）
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(slots=True)
class BM25Match:
    doc_key: str
    score: float
    payload: dict[str, Any]


class BM25Index:
    """一个知识库对应一个索引实例。

    **数据结构**：
        _postings:  term -> {doc_key: tf}     倒排表（为什么快：只扫含查询词的文档）
        _doc_len:   doc_key -> 词元数
        _payload:   doc_key -> 业务元数据（chunk_id / doc_id / 正文 / 页码 / 小节路径）
        _df:        term -> 文档频次

    **时间复杂度**：查询 O(Σ 查询词的 postings 长度)，与库大小无关。
    暴力扫描是 O(N) 且每次都要分词，几万 chunk 就会明显卡顿。
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_len: dict[str, int] = {}
        self._payload: dict[str, dict[str, Any]] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._total_len = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 写
    def add(self, doc_key: str, text: str, payload: dict[str, Any]) -> None:
        """加入/覆盖一篇文档。覆盖时会先把旧的词频清掉（保证幂等）。"""
        tokens = tokenize_zh(text)
        with self._lock:
            if doc_key in self._payload:
                self._remove_locked(doc_key)

            tf_map: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1

            for term, tf in tf_map.items():
                self._postings[term][doc_key] = tf
                self._df[term] += 1

            self._doc_len[doc_key] = max(len(tokens), 1)
            self._payload[doc_key] = payload
            self._total_len += self._doc_len[doc_key]

    def add_many(self, items: list[tuple[str, str, dict[str, Any]]]) -> int:
        for doc_key, text, payload in items:
            self.add(doc_key, text, payload)
        return len(items)

    def _remove_locked(self, doc_key: str) -> None:
        for term in list(self._postings.keys()):
            posting = self._postings[term]
            if doc_key in posting:
                del posting[doc_key]
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]
                if not posting:
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(doc_key, 0)
        self._payload.pop(doc_key, None)

    def remove(self, doc_key: str) -> bool:
        with self._lock:
            if doc_key not in self._payload:
                return False
            self._remove_locked(doc_key)
            return True

    def remove_where(self, predicate: Any) -> int:
        """按 payload 条件批量删除，例如 `lambda p: p["doc_id"] == 3`。

        删除文档时要按 doc_id 清索引，但 chunk 的 doc_key 是 `doc_id:index`，
        没有反向映射表，所以这里扫一遍 payload —— 删除是低频操作，可接受。
        """
        with self._lock:
            victims = [key for key, payload in self._payload.items() if predicate(payload)]
            for key in victims:
                self._remove_locked(key)
            return len(victims)

    def clear(self) -> None:
        with self._lock:
            self._postings.clear()
            self._doc_len.clear()
            self._payload.clear()
            self._df.clear()
            self._total_len = 0

    # ------------------------------------------------------------------ 读
    @property
    def doc_count(self) -> int:
        return len(self._payload)

    @property
    def term_count(self) -> int:
        return len(self._postings)

    @property
    def avgdl(self) -> float:
        return self._total_len / len(self._doc_len) if self._doc_len else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._payload)
        df = self._df.get(term, 0)
        if n == 0 or df == 0:
            return 0.0
        # +1 保证 IDF 恒正：经典形式在 df > N/2 时变负，会让高频词"倒扣分"
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, top_k: int = 20) -> list[BM25Match]:
        """返回得分最高的 `top_k` 条（分数 > 0 才会返回）。"""
        terms = tokenize_zh(query)
        if not terms:
            return []

        with self._lock:
            n = len(self._payload)
            if n == 0:
                return []
            avgdl = self.avgdl or 1.0
            scores: dict[str, float] = defaultdict(float)

            for term in terms:
                posting = self._postings.get(term)
                if not posting:
                    continue
                idf = self._idf(term)
                if idf <= 0:
                    continue
                for doc_key, tf in posting.items():
                    doc_len = self._doc_len.get(doc_key, 1)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
                    scores[doc_key] += idf * (tf * (self.k1 + 1.0)) / denom

            if not scores:
                return []
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            return [
                BM25Match(doc_key=key, score=score, payload=dict(self._payload.get(key, {})))
                for key, score in ranked
            ]

    def coverage_of(self, query: str, doc_key: str) -> float:
        """某个文档对查询词的覆盖率（用文档的原始词元集合算）。"""
        payload = self._payload.get(doc_key)
        if not payload:
            return 0.0
        from knowflow.retrieval.text import coverage

        return coverage(query, str(payload.get("content", "")))

    def stats(self) -> dict[str, Any]:
        return {
            "docs": self.doc_count,
            "terms": self.term_count,
            "avgdl": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }


class BM25Registry:
    """按知识库隔离的 BM25 索引注册表。

    **为什么按 KB 分索引**：`avgdl` 与 `df` 都是"库内统计量"。
    如果所有 KB 共用一个大索引，一个全是短句的 KB 会拉低 avgdl，
    从而影响另一个 KB 的打分 —— 跨库污染。分开索引还让"删除整个 KB"
    变成一次 `drop`，干净利落。

    **为什么必须有**：启动时从 MySQL 的 `chunks` 表全量重建，
    解决了"BM25 是内存索引、进程重启就丢"这个问题。
    **如果漏掉重建，服务不报错、不掉线，只是精确关键词查询静默变差** ——
    最危险的一类 bug。所以 `/health` 会把 `bm25_doc_count` 报出来。
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._indexes: dict[int, BM25Index] = {}
        self._lock = threading.RLock()

    def index_for(self, kb_id: int) -> BM25Index:
        with self._lock:
            index = self._indexes.get(kb_id)
            if index is None:
                index = BM25Index(k1=self.k1, b=self.b)
                self._indexes[kb_id] = index
            return index

    def add_chunks(self, kb_id: int, chunks: list[dict[str, Any]]) -> int:
        """`chunks` 每项需含 `vector_id`/`content`/`doc_id`/`doc_name`/`chunk_index`
        /`page_no`/`section_path`。"""
        index = self.index_for(kb_id)
        count = 0
        for chunk in chunks:
            doc_key = str(chunk["vector_id"])
            payload = {**chunk, "kb_id": kb_id}
            index.add(doc_key, str(chunk.get("content", "")), payload)
            count += 1
        return count

    def remove_document(self, kb_id: int, doc_id: int) -> int:
        with self._lock:
            index = self._indexes.get(kb_id)
        if index is None:
            return 0
        return index.remove_where(lambda payload: payload.get("doc_id") == doc_id)

    def drop_kb(self, kb_id: int) -> bool:
        with self._lock:
            return self._indexes.pop(kb_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._indexes.clear()

    def _select_indexes(self, kb_id: int | None) -> dict[int, BM25Index]:
        """选出这次要搜的索引集合。单个 KB 不存在时返回空 dict（而不是抛错）。"""
        with self._lock:
            if kb_id is None:
                return dict(self._indexes)
            index = self._indexes.get(kb_id)
            return {kb_id: index} if index is not None else {}

    def search(self, *, kb_id: int | None, query: str, top_k: int) -> list[dict[str, Any]]:
        """检索。`kb_id=None` 表示跨所有 KB 合并（按分数归并后取 top_k）。"""
        merged: list[dict[str, Any]] = []
        for _kid, index in self._select_indexes(kb_id).items():
            for match in index.search(query, top_k=top_k):
                merged.append(
                    {
                        "vector_id": match.doc_key,
                        "score": match.score,
                        "payload": match.payload,
                    }
                )
        merged.sort(key=lambda item: -item["score"])
        return merged[:top_k]

    @property
    def doc_count(self) -> int:
        with self._lock:
            return sum(index.doc_count for index in self._indexes.values())

    def doc_count_of(self, kb_id: int) -> int:
        with self._lock:
            index = self._indexes.get(kb_id)
        return index.doc_count if index else 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kb_count": len(self._indexes),
                "total_docs": sum(i.doc_count for i in self._indexes.values()),
                "per_kb": {str(k): v.doc_count for k, v in sorted(self._indexes.items())},
            }


__all__ = [
    "DEFAULT_B",
    "DEFAULT_K1",
    "BM25Index",
    "BM25Match",
    "BM25Registry",
]
