"""Chroma 向量库实现（单机文件型，`PersistentClient` 重启不丢）。

**一句话定位**：把 `VectorStore` 协议落到 Chroma 上：集合按 KB 隔离、
`hnsw:space=cosine`、写入按批、删除支持三种组合。

**在整条链路中的位置**：`ingest` 的最后一步在这里 `upsert`，`retrieve` 的第一步
在这里 `query`；`KBService.stats() 的 consistent 字段 + scripts/smoke_pipeline.py 的一致性断言` 用 `count` / `list_vector_ids`
比对它和 MySQL `chunks` 表；`/health` 用 `health()` 上报 `vector_count`。

**关键设计取舍**：

1. **距离 → 相似度必须转**：Chroma 的 cosine 空间返回的是**距离**
   （0 = 完全相同，2 = 完全相反），而项目契约要求 `score` 越大越相似，
   所以 `score = 1 - distance`。转换收在 `_distance_to_score` 一个函数里，
   带 docstring 说明 —— 这是新手最容易搞反、且搞反后「看起来还能用」的一处。
2. **collection 的获取加锁**：Chroma 客户端在多线程下并发
   `get_or_create_collection` 有竞态（实测会报集合已存在的错误）。
   用锁 + 双检缓存把「每个 KB 只获取一次」钉死，顺便省掉每次检索的元数据查询。
3. **显式传 `embedding_function=None`**：不传的话 Chroma 会把默认的
   ONNX all-MiniLM 记进集合配置，一旦有人写了 `add(documents=...)` 而没给向量，
   它会**静默下载 79MB 模型**再算一组 384 维向量（维度不符才报错）。
   我们的向量一律来自 `knowflow.embeddings`，所以禁用它的内置 EF；
   同时也避免了「集合已存在不同 EF」的配置冲突异常。
4. **只在 `upsert` 里创建集合**：`count` / `query` / `list_vector_ids` 对
   不存在的集合直接返回 0 / 空列表，不产生「查一下就把集合建出来」的副作用。
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

import chromadb
import structlog
from chromadb.errors import NotFoundError

from knowflow.core.config import Settings, get_settings
from knowflow.vectorstore.base import (
    KB_COLLECTION_PREFIX,
    VectorHit,
    VectorItem,
    VectorStoreError,
    collection_name,
    sanitize_metadata,
    validate_where,
)

if TYPE_CHECKING:  # 只为类型检查导入；运行期用 chromadb.PersistentClient 工厂函数
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Include, Metadata, Where

logger = structlog.get_logger(__name__)

_SPACE_KEY: Final[str] = "hnsw:space"
_COSINE: Final[str] = "cosine"
_DIM_KEY: Final[str] = "dim"
_QUERY_INCLUDE: Final[Include] = ["documents", "metadatas", "distances"]
# 取不到距离时的兜底（理论上不会发生，只是让类型和边界都明确）
_UNKNOWN_DISTANCE: Final[float] = 1.0

# chroma 的 `embeddings` 参数声明为不变型的 `list[...]`，直接传 `list[list[float]]`
# 会被 mypy 拒绝（list 不变型，`list[float]` 不等于 `Sequence[float]`）。
# 按它自己的元素类型声明即可，不必 cast，也不必用 type: ignore 掩盖。
_EmbeddingElement: TypeAlias = Sequence[float] | Sequence[int]


class ChromaVectorStore:
    """基于 Chroma `PersistentClient` 的向量库实现。"""

    backend = "chroma"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self.dim = self._settings.embedding_dim
        self._path = self._settings.chroma_path
        self._path.mkdir(parents=True, exist_ok=True)
        self._client: ClientAPI = chromadb.PersistentClient(path=str(self._path))
        # 每个 KB 只 get_or_create 一次；键是 kb_id，值是 Collection 句柄
        self._collections: dict[int, Collection] = {}
        self._lock = threading.Lock()
        logger.info("Chroma 向量库已就绪", path=str(self._path), dim=self.dim)

    # ------------------------------------------------------------ 集合管理
    def _existing_collection(self, kb_id: int) -> Collection | None:
        """取已存在的集合；不存在则返回 None（**不创建**）。"""
        cached = self._collections.get(kb_id)
        if cached is not None:
            return cached
        name = collection_name(kb_id)
        for collection in self._iter_collections():
            if collection.name == name:
                self._collections[kb_id] = collection
                return collection
        return None

    def _get_collection(self, kb_id: int) -> Collection:
        """取集合，不存在就创建（只在写入路径调用）。"""
        cached = self._collections.get(kb_id)
        if cached is not None:
            return cached
        # 双检锁：多线程并发 get_or_create_collection 有竞态（会报集合已存在），
        # 而且锁外的快速路径让高频检索不用每次都抢锁。
        with self._lock:
            cached = self._collections.get(kb_id)
            if cached is not None:
                return cached
            name = collection_name(kb_id)
            try:
                collection = self._client.get_or_create_collection(
                    name=name,
                    metadata={_SPACE_KEY: _COSINE, _DIM_KEY: self.dim},
                    # 见模块 docstring 第 3 条：禁止 Chroma 用自带 EF
                    embedding_function=None,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"获取 Chroma 集合失败（{name}）：{type(exc).__name__}: {exc}"
                ) from exc
            self._collections[kb_id] = collection
            return collection

    def _iter_collections(self) -> list[Collection]:
        """列出所有 `knowflow_kb_*` 集合（用于 `count(None)` / `reset(None)`）。"""
        try:
            listed = self._client.list_collections()
        except Exception as exc:
            raise VectorStoreError(f"列出 Chroma 集合失败：{type(exc).__name__}: {exc}") from exc
        result: list[Collection] = []
        for item in listed:
            # 不同 0.x 版本 list_collections 返回过「名字字符串」和「Collection 对象」两种
            name = item if isinstance(item, str) else item.name
            if not str(name).startswith(KB_COLLECTION_PREFIX):
                continue
            found = self._client.get_collection(str(name)) if isinstance(item, str) else item
            result.append(found)
        return result

    # ---------------------------------------------------------------- 写入
    def upsert(self, *, kb_id: int, items: Sequence[VectorItem]) -> int:
        """幂等分批写入（同 id 覆盖），返回写入/覆盖条数。"""
        if not items:
            # Chroma 对空列表直接抛 ValueError（实测），提前返回更省事也更清晰
            return 0
        collection = self._get_collection(kb_id)
        batch_size = self._settings.embedding_batch_size
        written = 0
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            ids, embeddings, documents, metadatas = self._prepare_batch(kb_id, batch)
            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"写入 Chroma 失败（kb_id={kb_id}, 本批 {len(ids)} 条）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            written += len(ids)
        return written

    def _prepare_batch(
        self, kb_id: int, batch: Sequence[VectorItem]
    ) -> tuple[list[str], list[_EmbeddingElement], list[str], list[Metadata]]:
        """校验维度并规范化 metadata（metadata 的规范由 base 统一负责）。"""
        ids: list[str] = []
        embeddings: list[_EmbeddingElement] = []
        documents: list[str] = []
        metadatas: list[Metadata] = []
        for item in batch:
            vector = list(item.vector)
            if len(vector) != self.dim:
                raise VectorStoreError(
                    f"向量维度不符（id={item.id}）：收到 {len(vector)} 维，"
                    f"集合是 {self.dim} 维。换 embedding 模型必须重建向量库。"
                )
            ids.append(item.id)
            embeddings.append(vector)
            documents.append(item.content)
            metadatas.append(sanitize_metadata(kb_id, item.metadata))
        return ids, embeddings, documents, metadatas

    # ---------------------------------------------------------------- 检索
    def query(
        self,
        *,
        kb_id: int,
        vector: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """检索并返回 `score = 1 - distance`（越大越相似）的结果。"""
        if top_k <= 0:
            raise VectorStoreError(f"top_k 必须大于 0，收到 {top_k}")
        query_vector = list(vector)
        if len(query_vector) != self.dim:
            raise VectorStoreError(
                f"查询向量维度不符：收到 {len(query_vector)} 维，集合是 {self.dim} 维"
            )
        validate_where(where)

        collection = self._existing_collection(kb_id)
        if collection is None or collection.count() == 0:
            # 空集合直接返回：既避免 Chroma 打「请求数超过索引大小」的警告，
            # 也避免「查一次就建出一个空集合」的副作用。
            return []

        try:
            # 同样按 chroma 声明的元素类型收敛（见 _EmbeddingElement 的注释）
            query_embeddings: list[_EmbeddingElement] = [query_vector]
            result = collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=_to_chroma_where(where),
                include=_QUERY_INCLUDE,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"检索 Chroma 失败（kb_id={kb_id}, top_k={top_k}）：{type(exc).__name__}: {exc}"
            ) from exc
        return self._to_hits(result)

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """把 Chroma 的 **cosine 距离** 转成项目约定的**相似度**（越大越相似）。

        为什么是 `1 - distance`：`hnsw:space=cosine` 返回 `d = 1 - cos(a, b)`，
        0 表示方向完全相同、2 表示完全相反，所以它**越小越相似**。
        项目契约（以及 BM25 融合、`VECTOR_MIN_SCORE` 闸门）要求 score
        **越大越相似**，于是 `score = 1 - d`，即余弦相似度本身。

        ⚠️ 直接把 distance 当 score 用不会报错，只会让「最相似的排到最后」——
        检索看起来还能出结果，但全是错的。这是本文件最需要看清的一行。
        """
        return 1.0 - float(distance)

    def _to_hits(self, result: Mapping[str, Any]) -> list[VectorHit]:
        """把 Chroma 的「按 query 分组的二维结果」摊平成一维 `VectorHit` 列表。"""
        ids = _first_row(result.get("ids"))
        distances = _first_row(result.get("distances"))
        documents = _first_row(result.get("documents"))
        metadatas = _first_row(result.get("metadatas"))

        hits: list[VectorHit] = []
        for index, raw_id in enumerate(ids):
            distance = _as_float(_at(distances, index), _UNKNOWN_DISTANCE)
            metadata = _at(metadatas, index)
            hits.append(
                VectorHit(
                    id=str(raw_id),
                    score=self._distance_to_score(distance),
                    content=_as_str(_at(documents, index)),
                    metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                )
            )
        return hits

    # ------------------------------------------------------------ 删除/统计
    def delete(
        self,
        *,
        kb_id: int,
        doc_id: int | None = None,
        vector_ids: Sequence[str] | None = None,
    ) -> int:
        """删除向量；三种组合的语义见 `vectorstore.base` 的模块 docstring。"""
        if doc_id is None and vector_ids is None:
            # 清整个 KB：直接 drop 集合，比逐条删快几个数量级
            previous = self._count_existing(kb_id)
            self.reset(kb_id=kb_id)
            return previous

        collection = self._existing_collection(kb_id)
        if collection is None:
            return 0

        if doc_id is not None and vector_ids is not None:
            # 交集：先取该文档的 id，再和调用方给的 id 求交
            existing = set(self.list_vector_ids(kb_id=kb_id, doc_id=doc_id))
            targets = sorted(existing.intersection(vector_ids))
            return self._delete_ids(collection, kb_id, targets)

        if vector_ids is not None:
            # 只给 id：直接交给 Chroma，它会忽略不存在的 id 并返回真实删除数
            targets = list(dict.fromkeys(vector_ids))
            return self._delete_ids(collection, kb_id, targets)

        if doc_id is None:
            # 上面的分支已经覆盖了所有组合，这里只是让类型收敛的兜底
            return 0
        try:
            result = collection.delete(where={"doc_id": int(doc_id)})
        except Exception as exc:
            raise VectorStoreError(
                f"删除文档向量失败（kb_id={kb_id}, doc_id={doc_id}）：{type(exc).__name__}: {exc}"
            ) from exc
        return _deleted_count(result, fallback=0)

    def _delete_ids(self, collection: Collection, kb_id: int, targets: Sequence[str]) -> int:
        """按 id 删除；空列表要提前返回（Chroma 对空 ids 抛 ValueError，实测）。"""
        if not targets:
            return 0
        try:
            result = collection.delete(ids=list(targets))
        except Exception as exc:
            raise VectorStoreError(
                f"按 id 删除向量失败（kb_id={kb_id}, 共 {len(targets)} 个 id）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return _deleted_count(result, fallback=len(targets))

    def count(self, *, kb_id: int | None = None) -> int:
        """统计向量条数；`kb_id=None` 表示把所有权重集合加起来。"""
        if kb_id is not None:
            return self._count_existing(kb_id)
        return sum(int(collection.count()) for collection in self._iter_collections())

    def _count_existing(self, kb_id: int) -> int:
        collection = self._existing_collection(kb_id)
        if collection is None:
            return 0
        try:
            return int(collection.count())
        except Exception as exc:
            raise VectorStoreError(
                f"统计 Chroma 集合条数失败（kb_id={kb_id}）：{type(exc).__name__}: {exc}"
            ) from exc

    def list_vector_ids(self, *, kb_id: int, doc_id: int | None = None) -> list[str]:
        """只取 id，不拉向量（`include=[]`）—— 一致性校验会全量调它。"""
        collection = self._existing_collection(kb_id)
        if collection is None:
            return []
        doc_filter: dict[str, Any] | None = {"doc_id": int(doc_id)} if doc_id is not None else None
        try:
            result = collection.get(where=_to_chroma_where(doc_filter), include=[])
        except Exception as exc:
            raise VectorStoreError(
                f"列出向量 id 失败（kb_id={kb_id}, doc_id={doc_id}）：{type(exc).__name__}: {exc}"
            ) from exc
        # ⚠️ `get()` 的 ids 是**一维**列表，而 `query()` 的是二维（按 query 分组）。
        # 用错 helper 的后果是「静默返回空列表」，所以这里单独用 _flat_str_list。
        return _flat_str_list(result.get("ids"))

    def reset(self, *, kb_id: int | None = None) -> None:
        """清空集合；集合不存在时静默跳过（Chroma 会抛 NotFoundError）。"""
        if kb_id is not None:
            self._collections.pop(kb_id, None)
            self._drop(collection_name(kb_id))
            return
        names = [collection.name for collection in self._iter_collections()]
        self._collections.clear()
        for name in names:
            self._drop(name)

    def _drop(self, name: str) -> None:
        """删除集合；「本来就不存在」不算错误（删除是幂等的）。"""
        with self._lock:
            try:
                self._client.delete_collection(name)
            except (NotFoundError, ValueError, KeyError) as exc:
                logger.debug("删除不存在的 Chroma 集合，忽略", collection=name, error=str(exc))
            except Exception as exc:
                raise VectorStoreError(
                    f"删除 Chroma 集合失败（{name}）：{type(exc).__name__}: {exc}"
                ) from exc

    # ---------------------------------------------------------------- 健康
    def health(self) -> dict[str, Any]:
        """`/health` 用的自述信息；**不抛异常**（运维接口不能自己 500）。"""
        info: dict[str, Any] = {
            "backend": self.backend,
            "path": str(self._path),
            "dim": self.dim,
            "kb_collections": 0,
            "total_vectors": 0,
        }
        try:
            collections = self._iter_collections()
            info["kb_collections"] = len(collections)
            info["total_vectors"] = sum(int(collection.count()) for collection in collections)
        except Exception as exc:  # noqa: BLE001
            info["error"] = f"{type(exc).__name__}: {exc}"
        return info


def _to_chroma_where(where: Mapping[str, Any] | None) -> Where | None:
    """把上层的过滤条件收敛成 chroma 的 `Where` 类型。

    纯粹是给 mypy 用的类型收敛（`Where` 是 `Dict[...]` 别名，而 dict 不变型，
    直接把 `dict[str, Any]` 传进去会被判为不兼容）；数据本身已经在
    `base.validate_where` 里校验过，这里不改动任何内容。
    """
    if not where:
        return None
    return cast("Where", dict(where))


def _first_row(rows: object) -> list[object]:
    """取「按 query 分组」结果的第一行（我们一次只查一个 query）。

    Chroma 的 **query** 返回值形如 `{"ids": [[...]], "distances": [[...]]}`，
    外层列表对应 query 个数，所以必须取 `[0]`；直接当一维用会拿到嵌套列表。
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        return []
    first = rows[0]
    if not isinstance(first, (list, tuple)):
        return []
    return list(first)


def _flat_str_list(rows: object) -> list[str]:
    """把 `get()` 返回的**一维** id 列表转成 `list[str]`。

    `get()` 和 `query()` 的返回形状不同：前者是 `{"ids": ["1:0", "1:1"]}`，
    后者是 `{"ids": [["1:0", "1:1"]]}`。混用两个 helper 不会报错，
    只会静默拿到空列表 —— 实测踩过，所以分成两个函数并各自写清形状。
    """
    if not isinstance(rows, (list, tuple)):
        return []
    return [str(item) for item in rows]


def _at(values: Sequence[object], index: int) -> object:
    """安全下标：Chroma 理论上不返回参差列表，但不要因为越界把检索搞崩。"""
    if 0 <= index < len(values):
        return values[index]
    return None


def _as_float(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _deleted_count(result: object, *, fallback: int) -> int:
    """Chroma 的 delete 返回 `{"deleted": N}` 而不是整数（实测），这里抹平差异。"""
    if isinstance(result, Mapping):
        deleted = result.get("deleted")
        if isinstance(deleted, int):
            return deleted
    if isinstance(result, int):
        return result
    return fallback


__all__ = ["ChromaVectorStore"]
