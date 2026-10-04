"""数据访问层：SQLAlchemy 基类/会话/模型。

刻意**不在这个 `__init__.py` 里 import 模型**：模型之间存在 relationship 交叉引用，
在这个位置聚集导入很容易制造循环 import。需要模型的代码请显式
`from knowflow.db.models import Document`。
"""

from knowflow.db.base import Base

__all__ = ["Base"]
