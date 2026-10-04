"""向量库抽象层：`VectorItem` / `VectorHit` / `VectorStore` 协议与共用工具。

**一句话定位**：定义「向量库」这件事的最小契约，让上层不关心底层是 Chroma
还是内存实现。

**在整条链路中的位置**：`ingest` 用 `upsert` 写入切分块的向量，`retrieve` 用
`query` 取回候选，删文档/删 KB 用 `delete` 清理，`KBService.stats() 的 consistent 字段 + scripts/smoke_pipeline.py 的一致性断言`
用 `count` / `list_vector_ids` 比对 MySQL 与向量库是否一致，`/health` 用 `health()`。

**语义约定（两个实现必须完全一致，否则「换后端 = 改行为」）**：

1. `query` 的 `score` **一律「越大越相似」**。Chroma 的 cosine 空间返回的是
   **距离**（越小越相似），实现里必须做 `1 - distance`（见 `chroma_store`）。
2. `kb_id=None` 传给 `count` / `reset` 表示「全部 KB」，其余方法必须给 kb_id。
3. 集合按 KB 隔离，命名 `knowflow_kb_{kb_id}`：删 KB 就等于 drop 集合，最干净。
4. `delete` 支持三种组合：只给 kb_id（清整个 KB）、kb_id+doc_id（清一个文档）、
   kb_id+vector_ids（精确删）；同时给 doc_id 与 vector_ids 时取**交集**。
5. `metadata` 的值**不允许是 None**：实测 Chroma 写入时会**静默丢弃** None 值的键
   （写完 `get()` 回来那个键就没了）。所以两个实现统一在 `sanitize_metadata`
   里丢弃 None，保证返回的 metadata 一模一样。
   ⚠️ 下游取值请统一写 `metadata.get("page_no")`，不要写 `metadata["page_no"]`。

**关键设计取舍**：

- `VectorItem` / `VectorHit` 都是 `frozen=True`：写入向量库的唯一入口是 `upsert`，
  拿到 hit 之后就地改 metadata 会让两个实现行为分叉（Chroma 的结果是反序列化
  出来的副本，改了也不会传回库）。
- `where` 支持范围严格对齐 Chroma 实测行为，并在**进门时**校验；不支持就抛
  `VectorStoreError`，绝不静默忽略 —— 静默忽略会变成「同一份代码换个后端结果就
  不同」，属于最难查的一类 bug。注意 Chroma 的硬性限制：**每层只能有一个条件**，
  多条件必须写成 `{"$and": [{"doc_id": 1}, {"page_no": 2}]}`；
  `$contains` 只对列表值生效，本项目 metadata 只有标量，所以直接拒绝。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

KB_COLLECTION_PREFIX: Final[str] = "knowflow_kb_"

# metadata 里必须由调用方提供、且缺了就会在「回表 JOIN / 按文档删除」时炸掉的键。
REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = ("doc_id", "doc_name", "chunk_index")
# 语义上允许为空的键（PDF 才有页码，纯文本没有小节路径）。
OPTIONAL_METADATA_KEYS: Final[tuple[str, ...]] = (
    "page_no",
    "section_path",
    "ext",
    "created_at",
)
# Chroma 只接受这四种标量做 metadata（传 list/dict 会直接报错）。
SUPPORTED_METADATA_TYPES: Final[tuple[type, ...]] = (str, int, float, bool)
# Chroma 支持的比较算子（实测：其余的会在它自己的校验里报 ValueError）。
_COMPARISON_OPERATORS: Final[frozenset[str]] = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
)
_LOGICAL_OPERATORS: Final[frozenset[str]] = frozenset({"$and", "$or"})
# 实测结论：Chroma 的 $contains / $not_contains **只对列表值**生效，
# 对标量字符串（即使全等）永远返回空结果。本项目 metadata 只有标量
# （sanitize_metadata 拒绝 list/dict），所以这两个算子在这里必然查不到东西，
# 与其静默查空，不如明确拒绝并告诉调用方该用什么。
_LIST_ONLY_OPERATORS: Final[frozenset[str]] = frozenset({"$contains", "$not_contains"})
# Chroma 要求 where 的**每一层都只有一个算子**：{"a": 1, "b": 2} 会直接报
# "Expected where to have exactly one operator"。多条件必须写成 $and。
_WHERE_LEVEL_KEYS: Final[int] = 1
_LOGICAL_MIN_TERMS: Final[int] = 2


class VectorStoreError(Exception):
    """向量库操作失败（不支持的条件、维度不符、后端异常）。

    上层捕获它并转成 HTTP 502 `UPSTREAM_ERROR` / 500 `INTERNAL_ERROR`，
    所以消息必须能直接定位问题（哪个 kb、哪个算子、期望维度多少）。
    """


@dataclass(slots=True, frozen=True)
class VectorItem:
    """一个待写入的向量（= 一个 chunk）。

    `id` 约定为 `f"{doc_id}:{chunk_index}"`，与 MySQL `chunks.vector_id` 一一对应，
    这是「向量库 ↔ 关系库」的一致性锚点（双向可查）。
    """

    id: str
    vector: list[float]
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class VectorHit:
    """一条检索结果。

    `score` 约定**越大越相似**（cosine 相似度，范围约 `[-1, 1]`）；
    上层用它跟 `VECTOR_MIN_SCORE` 比、用它做融合排序。
    """

    id: str
    score: float
    content: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """向量库统一契约。"""

    backend: str
    """`chroma` | `memory`，`/health` 的 `vector_backend` 用它。"""

    dim: int
    """集合维度，必须等于 `EMBEDDING_DIM`。"""

    def upsert(self, *, kb_id: int, items: Sequence[VectorItem]) -> int:
        """幂等写入（同 id 覆盖），返回写入/覆盖的条数。"""
        ...

    def query(
        self,
        *,
        kb_id: int,
        vector: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """检索 top_k，`score` 越大越相似。"""
        ...

    def delete(
        self,
        *,
        kb_id: int,
        doc_id: int | None = None,
        vector_ids: Sequence[str] | None = None,
    ) -> int:
        """删除向量，返回删除条数（语义见模块 docstring 第 4 条）。"""
        ...

    def count(self, *, kb_id: int | None = None) -> int:
        """统计向量条数；`kb_id=None` 表示全部 KB。"""
        ...

    def list_vector_ids(self, *, kb_id: int, doc_id: int | None = None) -> list[str]:
        """只取 id（不拉向量），用于一致性校验与按文档清理。"""
        ...

    def reset(self, *, kb_id: int | None = None) -> None:
        """清空；`kb_id=None` 表示清空全部 KB。"""
        ...

    def health(self) -> dict[str, Any]:
        """给 `/health` 用的自述信息（真实后端、路径、集合数、向量数、维度）。"""
        ...


def collection_name(kb_id: int) -> str:
    """KB 对应的集合名：`knowflow_kb_{kb_id}`。

    按 KB 隔离而不是用「一个集合 + metadata 过滤」的原因：
    删除 KB 变成 drop 集合（干净的 O(1) 操作，而不是扫全库删几万条），
    并且从根本上杜绝「过滤条件写错导致串库」这种隐私事故。
    """
    return f"{KB_COLLECTION_PREFIX}{kb_id}"


def sanitize_metadata(kb_id: int, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """把调用方给的 metadata 规范成两个后端都能接受、且**完全一致**的形式。

    做四件事：

    1. **强制覆盖 `kb_id`**：以方法入参为准。KB 隔离是检索正确性的底线，
       调用方写错 kb_id 时不能让错值进库（过滤条件依赖它）。
    2. **校验必含键**：`doc_id` / `doc_name` / `chunk_index` 缺失直接抛错。
       它们是回表 JOIN 与按文档删除的依据，缺了会在检索之后才炸，很难定位。
    3. **丢掉值为 None 的键**：这是 Chroma 的实测行为（写入时静默丢弃），
       内存实现必须跟着丢，否则两个后端返回的 metadata 一个有一个没有。
    4. **标量类型收敛**：`doc_id` / `chunk_index` / `page_no` 强制 `int`。
       Chroma 的 `where` 比较是**强类型**的，传字符串 `"3"` 去比 `{"doc_id": 3}`
       会一条都查不到而且不报错 —— 这类静默查空必须在写入口拦住。
    """
    missing = [key for key in REQUIRED_METADATA_KEYS if metadata.get(key) is None]
    if missing:
        raise VectorStoreError(
            f"向量 metadata 缺少必含键 {missing}（契约要求 {REQUIRED_METADATA_KEYS}），"
            f"现有键：{sorted(metadata)}"
        )

    result: dict[str, Any] = {"kb_id": int(kb_id)}
    for key, value in metadata.items():
        if value is None:
            continue
        if not isinstance(value, SUPPORTED_METADATA_TYPES):
            raise VectorStoreError(
                f"metadata[{key!r}] 的类型 {type(value).__name__} 不被向量库支持"
                f"（只支持 str/int/float/bool）"
            )
        result[str(key)] = value

    result["doc_id"] = int(metadata["doc_id"])
    result["chunk_index"] = int(metadata["chunk_index"])
    result["doc_name"] = str(metadata["doc_name"])
    if "page_no" in result:
        result["page_no"] = int(result["page_no"])
    for text_key in ("section_path", "ext", "created_at"):
        if text_key in result:
            result[text_key] = str(result[text_key])
    return result


def validate_where(where: Mapping[str, Any] | None) -> None:
    """校验 Mongo 风格过滤表达式，不支持就抛**清晰**错误。

    为什么必须显式校验而不是「透传给后端」：Chroma 只支持固定的几个算子，
    传 `$regex` 会抛它自己的 ValueError（消息里全是 Chroma 内部术语）；
    而内存实现如果不校验就会**静默忽略**这个条件，返回一堆不该返回的结果。
    同一份代码换后端结果不同是最难查的 bug，所以两个后端进门先跑同一个校验器。

    规则全部来自对 chromadb 1.5.9 的实测（报错原文记在注释里）：

    1. **每一层只允许一个键**：`{"doc_id": 1, "page_no": 2}` 在 Chroma 里抛
       `Expected where to have exactly one operator`。多条件要写成
       `{"$and": [{"doc_id": 1}, {"page_no": 2}]}`；
    2. `$and` / `$or` 的值必须是**至少两个**条件的列表；
    3. 一个键后面只能跟**一个**算子：`{"page_no": {"$gte": 1, "$lte": 5}}` 非法，
       要写成 `{"$and": [{"page_no": {"$gte": 1}}, {"page_no": {"$lte": 5}}]}`；
    4. `$contains` / `$not_contains` 明确拒绝：它们只对列表值生效，
       而本项目 metadata 只有标量，传了必然静默查空。
    """
    if where is None:
        return
    if not isinstance(where, Mapping):
        raise VectorStoreError(f"where 必须是 dict，收到 {type(where).__name__}")
    if len(where) != _WHERE_LEVEL_KEYS:
        raise VectorStoreError(
            f"where 每层只允许一个条件（Chroma 的硬性限制），收到 {sorted(where)}；"
            '多条件请写成 {"$and": [{"k1": v1}, {"k2": v2}]}'
        )

    for key, expected in where.items():
        if key in _LOGICAL_OPERATORS:
            if not isinstance(expected, (list, tuple)) or len(expected) < _LOGICAL_MIN_TERMS:
                raise VectorStoreError(
                    f"{key} 的值必须是至少 {_LOGICAL_MIN_TERMS} 个条件的列表"
                    f"（Chroma 的硬性限制），收到 {type(expected).__name__}"
                )
            for sub in expected:
                validate_where(sub)
            continue
        if key.startswith("$"):
            raise VectorStoreError(
                f"不支持的逻辑算子 {key}（只支持 {'/'.join(sorted(_LOGICAL_OPERATORS))}）"
            )
        if expected is None:
            # 实测：Chroma 对 {"page_no": None} 直接抛 ValueError，无法按 None 过滤。
            # 这里提前拦住，避免「内存能查、Chroma 报错」的诡异差异。
            raise VectorStoreError(
                f"where[{key!r}] 的值是 None：向量库不支持按 None 过滤"
                "（None 值的 metadata 键在写入时已被丢弃）"
            )
        if isinstance(expected, Mapping):
            if len(expected) != _WHERE_LEVEL_KEYS:
                raise VectorStoreError(
                    f"where[{key!r}] 只能跟一个算子（Chroma 的硬性限制），"
                    f"收到 {sorted(expected)}；多个算子是两个条件，请用 $and 组合"
                )
            unknown = [
                op
                for op in expected
                if op not in _COMPARISON_OPERATORS and op not in _LIST_ONLY_OPERATORS
            ]
            if unknown:
                raise VectorStoreError(
                    f"不支持的过滤算子 {unknown}（支持：{'/'.join(sorted(_COMPARISON_OPERATORS))}）"
                )
            list_only = [op for op in expected if op in _LIST_ONLY_OPERATORS]
            if list_only:
                raise VectorStoreError(
                    f"过滤算子 {list_only} 只对「列表型」metadata 生效，"
                    "而本项目 metadata 只有标量（str/int/float/bool），"
                    "用它必然静默查空。字符串包含匹配请用关键词检索（BM25），"
                    "精确匹配请用 $eq"
                )
            if any(op in {"$in", "$nin"} for op in expected):
                operand = next(iter(expected.values()))
                if not isinstance(operand, (list, tuple)):
                    raise VectorStoreError(
                        f"where[{key!r}] 的 $in/$nin 取值必须是列表，收到 {type(operand).__name__}"
                    )
            continue
        if not isinstance(expected, SUPPORTED_METADATA_TYPES):
            raise VectorStoreError(f"where[{key!r}] 的取值类型 {type(expected).__name__} 不被支持")


def metadata_matches(metadata: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """内存后端的 `where` 求值器，语义对齐 Chroma（含 `$and` / `$or` 嵌套）。"""
    if not where:
        return True
    for key, expected in where.items():
        if key in _LOGICAL_OPERATORS:
            results = (metadata_matches(metadata, sub) for sub in expected)
            if key == "$and" and not all(results):
                return False
            if key == "$or" and not any(results):
                return False
            continue
        # 键不存在（值为 None 被丢弃）时取到 None，与任何条件比较都不成立，
        # 这正是 Chroma 的行为：按一个不存在的键过滤 = 查不到东西。
        if not _value_matches(metadata.get(key), expected):
            return False
    return True


def _value_matches(actual: Any, expected: Any) -> bool:
    """单个键的比较（`expected` 可能是标量，也可能是算子 dict）。"""
    if not isinstance(expected, Mapping):
        return bool(actual == expected)

    for operator, operand in expected.items():
        if operator == "$eq" and actual != operand:
            return False
        if operator == "$ne" and actual == operand:
            return False
        if operator in {"$gt", "$gte", "$lt", "$lte"} and not _compare(actual, operand, operator):
            return False
        if operator == "$in" and actual not in operand:
            return False
        if operator == "$nin" and actual in operand:
            return False
    return True


def _compare(actual: Any, operand: Any, operator: str) -> bool:
    """数值/字符串比较；类型不可比时返回 False（而不是抛 TypeError）。"""
    if actual is None:
        return False
    try:
        if operator == "$gt":
            return bool(actual > operand)
        if operator == "$gte":
            return bool(actual >= operand)
        if operator == "$lt":
            return bool(actual < operand)
        return bool(actual <= operand)
    except TypeError:
        return False


def matches_delete_filter(
    *,
    item_id: str,
    metadata: Mapping[str, Any],
    doc_id: int | None,
    vector_ids: Collection[str] | None,
) -> bool:
    """`delete` 的三种过滤组合（两个后端共用，保证语义一致）：

    - 只给 `vector_ids` → 按 id 精确删；
    - 只给 `doc_id` → 按 `metadata["doc_id"]` 删；
    - 两个都给 → **取交集**（即「这篇文档里指定的那几个块」）。

    `vector_ids` 建议传 `set`（这里是 `in` 判断，list 会是 O(n)）。
    """
    if vector_ids is not None and item_id not in vector_ids:
        return False
    if doc_id is None:
        return True
    return int(metadata.get("doc_id", -1)) == int(doc_id)


__all__ = [
    "KB_COLLECTION_PREFIX",
    "OPTIONAL_METADATA_KEYS",
    "REQUIRED_METADATA_KEYS",
    "SUPPORTED_METADATA_TYPES",
    "VectorHit",
    "VectorItem",
    "VectorStore",
    "VectorStoreError",
    "collection_name",
    "matches_delete_filter",
    "metadata_matches",
    "sanitize_metadata",
    "validate_where",
]
