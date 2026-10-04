"""跨数据库可用的列类型与时间工具。

**为什么要 `.with_variant`**：本项目生产用 MySQL（要 `MEDIUMTEXT` 才能存下长 chunk），
但测试用 SQLite（不落盘、快、CI 无需起数据库）。
直接写 `MEDIUMTEXT` 在 SQLite 上会建表失败，所以定义一份"逻辑类型"，
让 SQLAlchemy 按方言选具体实现 —— 这样**同一套模型定义在两种数据库上都能建表**，
测试才能真正覆盖生产模型。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from sqlalchemy import BigInteger, DateTime, SmallInteger, Text
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.types import TypeEngine

# 主键/外键都用 BIGINT（MySQL 侧 UNSIGNED，SQLite 无此概念自动退回普通 BIGINT）。
# 出现在这里而不是每个模型里重复写，是为了保证**主键与所有外键类型完全一致**——
# 类型不一致时 MySQL 会拒绝建外键，而 SQLite 不报错（测试通过、生产建表失败，
# 这类"测试假通过"最难查）。
#
# ⚠ SQLite 侧必须显式映射成 INTEGER：SQLite 只把「声明类型恰好是 INTEGER」的
# PRIMARY KEY 当作 rowid 别名来自动分配主键。声明成 BIGINT 时建表照样成功，
# 但 INSERT 不带 id 会直接报 `NOT NULL constraint failed: users.id` ——
# 于是"同一套模型定义在 SQLite 上也能跑测试"这个承诺在**写入**时才崩，
# 而 SQLite 单测恰好全靠写入（见 db/session.py 的 StaticPool 说明）。
BigIntType: Final[TypeEngine[int]] = (
    BigInteger()
    .with_variant(mysql.BIGINT(unsigned=True), "mysql")
    .with_variant(sqlite.INTEGER(), "sqlite")
)

# 小整数（反馈评分 -1/0/1 等）
SmallIntType: Final[TypeEngine[int]] = SmallInteger()

# 长文本：MySQL 用 MEDIUMTEXT（最大 16MB，足够放整篇文档的 chunk 合并）；
# 其它方言退回 TEXT（SQLite 的 TEXT 本身无长度上限）。
LongText: Final[TypeEngine[str]] = Text().with_variant(mysql.MEDIUMTEXT(), "mysql")

# 高精度时间：DATETIME(6) 保留微秒，排序稳定（同秒内多条记录不会乱序）。
# MySQL 的 DATETIME 不带时区，所以本项目的约定是：**一律存 UTC 的 naive datetime**，
# 序列化时补上 "Z"。这条约定写在 `docs/01-数据库与接口契约.md` 第一节。
DateTime6: Final[TypeEngine[datetime]] = DateTime().with_variant(mysql.DATETIME(fsp=6), "mysql")


def utcnow() -> datetime:
    """当前 UTC 时间（naive）。

    用 `datetime.now(UTC).replace(tzinfo=None)` 而不是已废弃的 `utcnow()`，
    同时保证写进库的值一定是 naive UTC，读出来不会带时区歧义。
    """
    return datetime.now(UTC).replace(tzinfo=None)


def to_iso_z(value: datetime | None) -> str | None:
    """把库里的 naive UTC datetime 序列化成带 `Z` 的 ISO8601 字符串。

    这是唯一的时间出口，所有 schema 都调它 —— 避免有的地方输出 `+00:00`、
    有的地方输出本地时间，前端解析要写两套。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "BigIntType",
    "DateTime6",
    "LongText",
    "SmallIntType",
    "to_iso_z",
    "utcnow",
]
