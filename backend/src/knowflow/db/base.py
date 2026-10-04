"""SQLAlchemy 声明式基类与公共 Mixin。

`NAMING_CONVENTION` 的作用经常被低估：不写它，Alembic 自动生成的迁移里
约束名会由数据库随机生成（MySQL 上叫 `PRAGMA` 风格的名字），
导致「想改一个索引」时无从下手，也**无法可靠地 autogenerate 出 DROP**。
显式命名约定让迁移可预测、可 review。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from knowflow.db.types import DateTime6, utcnow

NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """全项目唯一的声明式基类。

    所有模型都必须继承它 —— 否则 Alembic 的 `target_metadata` 扫不到，
    autogenerate 会生成一个「把所有表都删掉」的迁移（真实踩过的坑）。
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """创建/更新时间。用 Python 侧默认值而不是 `server_default`，
    这样测试里 `session.flush()` 之后对象上立刻能读到值，不用 refresh 回查。"""

    created_at: Mapped[datetime] = mapped_column(DateTime6, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime6, default=utcnow, onupdate=utcnow, nullable=False
    )


class SoftDeleteMixin:
    """软删除。

    为什么软删：`message_citations` 可能引用已被删除文档的 chunk，
    物理删除会让引用悬空、历史会话的引用点开就是 404。
    软删后 chunk 仍在，引用依然可回溯（这也是「删了文档但历史答案还能核对出处」的原因）。
    """

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime6, default=None, nullable=True)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def mark_deleted(self) -> None:
        self.deleted_at = utcnow()


__all__ = ["NAMING_CONVENTION", "Base", "SoftDeleteMixin", "TimestampMixin"]
