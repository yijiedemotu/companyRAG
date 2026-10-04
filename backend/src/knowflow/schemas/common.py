"""公共契约：分页、错误信封、时间序列化与零散小模型。

本模块只放**被两个以上业务模块共用**的东西。放这里的判断标准很简单：
如果删掉它，是否会有两个以上的文件需要各自复制一份？是，就放这里。

三个关键决定：

1. **`UtcDatetime` 是唯一的时间字段类型**。全项目的时间出口只有一个
   （`knowflow.db.types.to_iso_z`），schema 层不允许再写第二个序列化逻辑。
   如果每个 `*Out` 都自己写 `field_serializer`，迟早出现有的输出 `+00:00`、
   有的输出本地时间，前端就得写两套解析——这类"看起来能跑"的不一致最难查。
2. **`ErrorEnvelope` 与 `KnowFlowError.to_dict()` 逐字对齐**。契约文档第一节
   规定了所有非 2xx 的响应体形状，这里的模型只用于 OpenAPI 文档展示，
   实际序列化仍由 `exceptions.py` 决定；两者字段必须一模一样，
   否则 `/docs` 上的说明就是错的（比没有文档更糟）。
3. **`Page.pages` 在 `total=0` 时必须是 0**。契约文档给的样例就是
   `{"items": [], "total": 0, "page": 1, "size": 20, "pages": 0}`。
   用 `ceil(total/size) or 1` 那种写法会得到 1，前端分页器会显示"第 1/1 页"
   却没有任何数据。边界条件在这里收口，业务层不用再想。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar, get_args

from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from knowflow.db.models import MODE_AGENT, MODE_RAG
from knowflow.db.types import to_iso_z

__all__ = [
    "ErrorBody",
    "ErrorEnvelope",
    "MessageOut",
    "OkOut",
    "Page",
    "PageParams",
    "SessionMode",
    "SimpleMessageOut",
    "T",
    "UtcDatetime",
]

T = TypeVar("T")

#: 会话/对话模式：`rag` = 单轮检索增强，`agent` = 带判定、改写与自省的多轮图。
#: 放在 common 而不是 chat：`conversations.mode` 与 `ChatRequest.mode` 是同一个概念，
#: 取值必须完全一致（`conversations.mode` 决定回放历史时用哪条链路）。
SessionMode = Literal["rag", "agent"]

# 取值来源是 ORM 常量（`db/models/chat.py`），但 `Literal[...]` 里**必须写字面量**：
# 试过 `Literal[MODE_AGENT]`（即使把常量标注成 `Final[str]`），mypy 一律报
# "Parameter 1 of Literal[...] is invalid"。这是 mypy 的硬限制，不是风格问题。
# 为了不让"两处取值"变成漂移隐患，下面用一条 import 期断言把它钉住：
# 以后谁改了 ORM 常量而忘了改 schema，第一次 import 就会直接炸，
# 而不是等到线上出现一个无法反序列化的 mode。
assert get_args(SessionMode) == (MODE_RAG, MODE_AGENT), (
    "SessionMode 与 ORM 常量不一致：请同步 schemas/common.py 与 db/models/chat.py"
)

# 时间字段的统一类型。
#
# `datetime | None` 作为**入参**（校验/解析）依然可用，因为 PlainSerializer
# 只改变序列化方向；换成输出时统一走 `to_iso_z()`，得到 `2026-01-01T00:00:00.000Z`。
UtcDatetime = Annotated[
    datetime | None,
    PlainSerializer(to_iso_z, return_type=str | None),
]
"""`datetime | None` 的统一序列化类型：输出 ISO8601 且带 `Z`。"""


class PageParams:
    """分页查询参数（FastAPI 依赖）。

    用类而不是 `BaseModel`：这些值来自 query string，用 `Annotated[PageParams, Depends()]`
    注入后 FastAPI 会在 `/docs` 上渲染出两个真正的查询参数输入框
    （page / size），而不是一个 JSON body——分页参数放 body 里是反直觉的。

    上限 `le=100` 来自契约文档第一节：`size` 默认 20、最大 100。
    没有上限时，前端一句 `size=100000` 就能让 MySQL 全表扫描把库拖死。

    **用法必须写 `Annotated[PageParams, Depends()]`**（`Depends()` 不带参数）：

        @router.get("/kbs")
        async def list_kbs(params: Annotated[PageParams, Depends()], ...): ...

    写成 `Depends(PageParams)` 也能跑，但语义完全不同——FastAPI 会把
    `PageParams` 整个当成一个**依赖函数**去调用并推导参数，`/docs` 上呈现成
    一个 body 参数，而不是两个查询参数输入框。实测确认过（FastAPI 0.142）。
    """

    def __init__(
        self,
        page: int = Query(1, ge=1, description="页码，从 1 开始"),
        size: int = Query(20, ge=1, le=100, description="每页条数，默认 20，最大 100"),
    ) -> None:
        self.page = page
        self.size = size

    @property
    def offset(self) -> int:
        """SQL `OFFSET`。分页到 SQL 的唯一换算点，避免各处手算 `(page-1)*size`。"""
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"PageParams(page={self.page}, size={self.size}, offset={self.offset})"


class Page(BaseModel, Generic[T]):
    """统一分页信封，见契约文档第一节。

    `pages` 由 `create()` 计算而**不是**由调用方传入：让调用方算，
    早晚有一个接口漏算或者用 `math.ceil` 和整除各写一遍。
    """

    model_config = ConfigDict(from_attributes=True)

    items: list[T] = Field(default_factory=list, description="当前页数据")
    total: int = Field(default=0, ge=0, description="满足条件的总条数")
    page: int = Field(default=1, ge=1, description="当前页码，从 1 开始")
    size: int = Field(default=20, ge=1, description="每页条数")
    pages: int = Field(default=0, ge=0, description="总页数，total=0 时为 0")

    @classmethod
    def create(cls, items: list[T], total: int, page: int, size: int) -> Page[T]:
        """构造分页响应。

        边界处理（都必须显式判断，只靠 `ceil` 会出错）：
        - `total <= 0` → `pages = 0`（空列表不是"第 1 页"，是"没有页"）；
        - `size <= 0` → `pages = 0` 且 `size` 归一到 1（除零会直接抛
          `ZeroDivisionError`；正常路径下 `PageParams` 已挡住 size<1，
          但 service 层可能用 `size=0` 表示"不分页"，这时不该 500）；
        - `page <= 0` → 归一到 1（模型的 `ge=1` 约束会直接拒绝 0，
          与其让一个"只要总数"的内部调用方踩到 500，不如在这里归一）。
        """
        safe_total = max(total, 0)
        pages = 0 if safe_total == 0 or size <= 0 else math.ceil(safe_total / size)
        return cls(
            items=items,
            total=safe_total,
            page=max(page, 1),
            size=max(size, 1),
            pages=pages,
        )


class ErrorBody(BaseModel):
    """错误信封的内层对象，字段与 `KnowFlowError.to_dict()["error"]` 完全一致。"""

    model_config = ConfigDict(from_attributes=True)

    code: str = Field(description="稳定的机器可读错误码，如 KB_NOT_FOUND（前端按它分支）")
    message: str = Field(description="给人看的中文错误信息（前端不要匹配这个文本）")
    detail: Any = Field(default=None, description="可选的额外上下文，结构随错误码而定")
    request_id: str | None = Field(default=None, description="请求 ID，排查问题时提供给运维")
    timestamp: str | None = Field(default=None, description="错误发生时刻，ISO8601 带 Z")


class ErrorEnvelope(BaseModel):
    """所有非 2xx 响应的统一外层结构。"""

    model_config = ConfigDict(from_attributes=True)

    error: ErrorBody = Field(description="错误详情")


class SimpleMessageOut(BaseModel):
    """一句提示性文案。用于没有实体可返回、但需要给人反馈的接口。

    原名 `MessageOut`，改名原因：`knowflow.schemas` 顶层导出的 `MessageOut`
    必须是会话里的聊天消息（[conversation.MessageOut](conversation.py)，含
    citations / tokens），因为业务代码 99% 想要的是那个。两者同名会让
    `from knowflow.schemas import MessageOut` 静默拿到错的那个
    （拿到 `{message: str}` 后在 `.citations` 上 AttributeError，报错点离原因很远）。

    兼容：本模块保留 `MessageOut = SimpleMessageOut` 别名，老引用不会断。
    """

    model_config = ConfigDict(from_attributes=True)

    message: str = Field(description="提示文案")


#: 向后兼容别名（原 `common.MessageOut` 的名字）。
#: 注意顶层 `knowflow.schemas.MessageOut` 现在指向**会话消息**，不是这个。
MessageOut = SimpleMessageOut


class OkOut(BaseModel):
    """`{"ok": true, "message": ...}`。

    DELETE 接口按契约返回 204（无响应体），但调试/管理接口有时希望
    明确回一句人话，这类接口统一用它，避免随手返回 `{"result": "done"}` 这种
    没有 schema 的裸字典。
    """

    model_config = ConfigDict(from_attributes=True)

    ok: bool = Field(default=True, description="操作是否成功")
    message: str = Field(default="ok", description="提示文案")
