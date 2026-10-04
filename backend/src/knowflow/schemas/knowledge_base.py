"""知识库契约：创建 / 更新 / 输出 / 统计。

字段来源：契约文档 5.2 节。

**为什么这里要重复一遍 `chunk_overlap < chunk_size` 的校验**：
`core/config.py` 已经对全局默认值做过这个校验，但接口允许按 KB 覆盖这两个值
（切分粒度必须能按 KB 调：技术文档和 FAQ 的最优粒度不一样）。覆盖值只在
请求体里出现，不经过 `Settings`，所以这里是唯一的拦截点。漏了它，
`chunk_overlap >= chunk_size` 会让递归切分器无法收敛（无限切分直到内存爆）。

**`doc_count` / `chunk_count` 是派生字段**：它们在 ORM 上不存在，由 service 层
聚合查询后连同模型一起塞进 dict，再 `model_validate`。schema 里不写任何数据库
访问逻辑——schema 一旦能查库，就会有人从 api 层顺手调它，最终变成"在序列化阶段
触发 N+1 查询"。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowflow.schemas.common import UtcDatetime

__all__ = [
    "KBCreateRequest",
    "KBOut",
    "KBStatsOut",
    "KBUpdateRequest",
]

# 切分参数的上/下限与 `core/config.py` 的 `chunk_size` / `chunk_overlap` 保持一致。
# 不一致的话，接口能收下一个配置体系拒绝的值，同一套参数在两个入口行为不同。
_CHUNK_SIZE_GE = 50
_CHUNK_SIZE_LE = 8000
_CHUNK_OVERLAP_GE = 0
_CHUNK_OVERLAP_LE = 4000


class KBCreateRequest(BaseModel):
    """创建知识库（契约 5.2：`{name, description?, chunk_size?, chunk_overlap?}`）。

    `chunk_size` / `chunk_overlap` 传 `None` 表示"用服务端配置默认值"——
    这样默认值只在 `core/config.py` 定义一处，前端不必知道 600/120。
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        min_length=1,
        max_length=128,
        description="知识库名称，全局唯一（重名返回 409 KB_NAME_CONFLICT）",
    )
    description: str | None = Field(default=None, max_length=512, description="描述，可选")
    chunk_size: int | None = Field(
        default=None,
        ge=_CHUNK_SIZE_GE,
        le=_CHUNK_SIZE_LE,
        description="子块目标字符数，留空用服务端默认值",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=_CHUNK_OVERLAP_GE,
        le=_CHUNK_OVERLAP_LE,
        description="相邻子块重叠字符数，必须小于 chunk_size，留空用服务端默认值",
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> KBCreateRequest:
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})，"
                "否则递归切分器无法收敛"
            )
        return self


class KBUpdateRequest(BaseModel):
    """更新知识库（PATCH）。**所有字段可选**，只改传了的字段。

    为什么用 PATCH 语义而不是"整体覆盖"：前端编辑弹窗只提交用户改过的字段，
    未提交的字段保持原值。若按 PUT 处理，没传的字段会被重置成默认值，
    用户只改一个名字却把 chunk_size 改回 600 —— 这是很容易发生的静默数据损坏。
    所以 service 层必须用 `model_dump(exclude_unset=True)` 取增量。
    """

    model_config = ConfigDict(from_attributes=True)

    name: str | None = Field(
        default=None, min_length=1, max_length=128, description="新名称，重名返回 409"
    )
    description: str | None = Field(default=None, max_length=512, description="新描述")
    chunk_size: int | None = Field(
        default=None, ge=_CHUNK_SIZE_GE, le=_CHUNK_SIZE_LE, description="新的子块字符数"
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=_CHUNK_OVERLAP_GE,
        le=_CHUNK_OVERLAP_LE,
        description="新的重叠字符数，必须小于 chunk_size",
    )
    is_active: bool | None = Field(default=None, description="是否启用；停用后该 KB 不参与检索")

    @model_validator(mode="after")
    def _check_overlap(self) -> KBUpdateRequest:
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})"
            )
        return self


class KBOut(BaseModel):
    """知识库输出（契约 5.2 节的 `KBOut`，字段顺序照抄）。

    `doc_count` / `chunk_count` 不是 `knowledge_bases` 表的列，由 service 传入。
    没有它们，KB 列表页要为每个 KB 单独查一次文档数（N+1）。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="知识库 ID")
    name: str = Field(description="知识库名称")
    description: str | None = Field(default=None, description="描述")
    owner_id: int = Field(description="创建者用户 ID")
    embedding_provider: str = Field(description="创建时快照的向量化提供方：local / hash / api")
    embedding_model: str = Field(description="创建时快照的向量化模型名")
    embedding_dim: int = Field(description="向量维度，与模型绑定，换模型必须重建 KB")
    chunk_size: int = Field(description="子块目标字符数")
    chunk_overlap: int = Field(description="相邻子块重叠字符数")
    is_active: bool = Field(description="是否启用")
    doc_count: int = Field(default=0, ge=0, description="未删除文档数（派生字段）")
    chunk_count: int = Field(default=0, ge=0, description="切片总数（派生字段）")
    created_at: UtcDatetime = Field(default=None, description="创建时间，UTC ISO8601 带 Z")

    @classmethod
    def from_orm_with(
        cls,
        orm_obj: Any,
        *,
        doc_count: int = 0,
        chunk_count: int = 0,
    ) -> KBOut:
        """ORM 对象 + 派生计数 → `KBOut`。

        刻意做成显式命名参数而不是 `**extra`：多传一个拼错的字段名
        （`doc_counts=`）会静默失效并回落到 0，前端就会显示"0 篇文档"。
        显式参数在调用点就会报 `TypeError`。
        """
        return cls.model_validate(
            {
                "id": orm_obj.id,
                "name": orm_obj.name,
                "description": orm_obj.description,
                "owner_id": orm_obj.owner_id,
                "embedding_provider": orm_obj.embedding_provider,
                "embedding_model": orm_obj.embedding_model,
                "embedding_dim": orm_obj.embedding_dim,
                "chunk_size": orm_obj.chunk_size,
                "chunk_overlap": orm_obj.chunk_overlap,
                "is_active": orm_obj.is_active,
                "doc_count": doc_count,
                "chunk_count": chunk_count,
                "created_at": orm_obj.created_at,
            }
        )


class KBStatsOut(BaseModel):
    """知识库一致性自检（契约 5.2）。

    `consistent = (chunk_count == vector_count)` 是**把静默降级变可见**的核心：
    写入向量库失败时如果只吞掉异常，检索会莫名少召回，而页面上一切正常。
    把这个布尔值暴露出来，前端画红点，问题当场可见。
    """

    model_config = ConfigDict(from_attributes=True)

    kb_id: int = Field(description="知识库 ID")
    doc_count: int = Field(default=0, ge=0, description="未删除文档总数")
    ready_doc_count: int = Field(default=0, ge=0, description="状态为 READY 的文档数")
    chunk_count: int = Field(default=0, ge=0, description="MySQL 中的切片总数")
    vector_count: int = Field(default=0, ge=0, description="向量库中的向量总数")
    bm25_doc_count: int = Field(default=0, ge=0, description="BM25 索引覆盖的切片数")
    total_chars: int = Field(default=0, ge=0, description="切片字符数合计")
    total_tokens: int = Field(default=0, ge=0, description="切片 token 数合计")
    consistent: bool = Field(
        default=True,
        description="chunk_count 是否等于 vector_count；为 false 说明向量库写入有丢失",
    )
