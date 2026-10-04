"""会话、消息、引用与反馈契约。

字段来源：契约文档 5.7 节。

`MessageOut` 里这一组字段一个都不能省——它们是"回答为什么长这样"的全部线索：

- `refusal`：这条是拒答还是真答案。前端靠它决定气泡样式；
  也是质量指标（拒答率）的唯一来源，靠匹配文案判断是不可靠的。
- `retrieval_rounds`：agent 模式改写重查了几轮。
- `trace_id`：点进 `/obs/traces/{trace_id}` 就能看到那次请求的完整瀑布图。
  历史消息里保留它，意味着**两周后仍能解释当时的答案**，这是本项目排障能力的根。

`cost_usd` 用 `float` 而不是 `Decimal`：JSON 里它就是个数字，
用 `Decimal` 会在 `model_dump(mode="json")` 时变成字符串（前端要做 `parseFloat`，
忘了就得到 `NaN`），得不偿失。精度损失对"展示花了多少钱"这个用途可以忽略。
"""

from __future__ import annotations

from typing import Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from knowflow.db.models import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
)
from knowflow.schemas.common import Page, SessionMode, UtcDatetime

__all__ = [
    "CitationOut",
    "ConversationCreateRequest",
    "ConversationOut",
    "FeedbackOut",
    "FeedbackRequest",
    "MessageOut",
    "MessagePage",
    "MessageRole",
]

#: 消息角色，与 `messages.role` 列的取值一致。
#: `Literal[...]` 只接受字面量（mypy 硬限制），下面的断言负责守卫一致性。
MessageRole = Literal["user", "assistant", "system", "tool"]

assert get_args(MessageRole) == (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_TOOL), (
    "MessageRole 与 db/models/chat.py 的角色常量不一致"
)

#: 反馈评分：1 有用 / -1 没用 / 0 中立（契约第三节 `feedback.rating`）。
FeedbackRating = Literal[-1, 0, 1]


class CitationOut(BaseModel):
    """答案里第 `rank` 个 `[n]` 对应的出处（契约 5.7 的 `citations[]`）。

    引用**落到 `chunk_id`** 而不是复制原文：历史答案的 `[1]` 点开能回到真实切片
    （含页码、小节路径），文档软删后引用依然可回溯。

    `doc_name` / `page_no` / `section_path` **不由 ORM 直接提供**（`message_citations`
    表只有 `chunk_id` / `doc_id`），它们**由路由层 JOIN `chunks` + `documents` 填充**
    （见 `api/routes/conversations.py`）：这样做的好处是历史引用在文档被改名/移动后
    依然显示**当前**的文件名，而不是当初写进库里的那个快照。
    三个字段都可选，前端缺省时用自己的兜底文案（`片段 #<chunk_id>`）。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="引用记录 ID")
    message_id: int = Field(description="所属消息 ID")
    chunk_id: int | None = Field(
        default=None,
        description="对应切片 ID；切片被物理删除时为 null（引用行保留，指针失效）",
    )
    doc_id: int | None = Field(default=None, description="所属文档 ID")
    rank: int = Field(ge=1, description="引用序号，从 1 开始，对应正文里的 [n]")
    score: float = Field(default=0.0, description="相关度得分，越大越相似")
    snippet: str = Field(default="", description="命中片段摘要，最长 512 字符")
    doc_name: str | None = Field(
        default=None, description="文档文件名；由路由层 JOIN documents 填充"
    )
    page_no: int | None = Field(
        default=None, description="页码，仅 PDF 有值；由路由层 JOIN chunks 填充"
    )
    section_path: str | None = Field(
        default=None, description="所属小节路径；由路由层 JOIN chunks 填充"
    )


class MessageOut(BaseModel):
    """一条消息（契约 5.7 节的 `MessageOut`，字段顺序照抄）。

    这是 `from knowflow.schemas import MessageOut` 拿到的模型（会话聊天消息，
    含 citations / tokens / trace_id）。

    **同名的另一个是 `common.SimpleMessageOut`**（原 `common.MessageOut`）：
    那个只有 `{message: str}`，是一句提示文案，用于没有实体可返回的接口。
    需要它请显式写 `from knowflow.schemas import SimpleMessageOut`，
    不要用 `common.MessageOut`（那个名字只是兼容别名，容易拿错）。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="消息 ID")
    conversation_id: int = Field(description="所属会话 ID")
    role: MessageRole = Field(description="角色：user / assistant / system / tool")
    content: str = Field(description="消息正文")
    mode: str | None = Field(default=None, description="生成时的对话模式：rag 或 agent")
    model: str | None = Field(default=None, description="生成时使用的模型名")
    prompt_tokens: int = Field(default=0, ge=0, description="输入 token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 token 数")
    cost_usd: float = Field(default=0.0, ge=0, description="本条消息的费用（美元）")
    latency_ms: int | None = Field(default=None, description="生成耗时（毫秒）")
    refusal: bool = Field(default=False, description="是否因知识库无相关内容而拒答")
    retrieval_rounds: int = Field(default=0, ge=0, description="本次检索轮数")
    trace_id: str | None = Field(
        default=None, description="链路追踪 ID，可跳转 /obs/traces/{trace_id} 看全过程"
    )
    citations: list[CitationOut] = Field(
        default_factory=list, description="引用列表，rank 与正文 [n] 一一对应"
    )
    created_at: UtcDatetime = Field(default=None, description="创建时间，UTC ISO8601 带 Z")


class MessagePage(Page[MessageOut]):
    """`GET /conversations/{id}/messages` 的分页响应：`Page[MessageOut]`。

    **为什么是子类而不是 `MessagePage = Page["MessageOut"]` 这种别名**：
    别名里的字符串是 `ForwardRef`，Pydantic 会拿**定义 `Page` 的那个模块**
    （`schemas/common.py`）的命名空间去解析它，而那里也有一个 `MessageOut`
    （提示文案，现名 `SimpleMessageOut`，旧名仍为别名），于是
    `Page["MessageOut"]` 会静默解析成 `Page[SimpleMessageOut]`——
    接口看起来能跑，返回的 items 却是提示文案。实测确认过。
    子类没这个坑（泛型参数写的是真类），而且能在 `/docs` 上得到一个
    有名字的 `MessagePage` schema。
    """

    model_config = ConfigDict(from_attributes=True)

    # `Page.create` 是 `@classmethod` 返回 `Page[T]`，在子类上调用时 mypy 会把
    # 返回类型推成基类的 `Page[MessageOut]`（泛型 Self 推导的已知限制），
    # 于是"路由声明的 `-> MessagePage`"与"实际返回 `Page[MessageOut]`"被判为不兼容。
    # 覆盖一次把返回类型收窄到 `MessagePage`：运行时行为与基类**完全一致**
    # （基类实现里只用到 `cls(...)`，传进来的就是 `MessagePage`），
    # 所以这里用 `cast` 只是把 mypy 已经推不出来的事实**写给类型检查器看**，
    # 不是在做类型欺骗。代价是 1 行 cast，换来 mypy 0 error。
    @classmethod
    def create(  # type: ignore[override]
        cls, items: list[MessageOut], total: int, page: int, size: int
    ) -> MessagePage:
        """与 `Page.create` 同语义，只是把返回类型钉成 `MessagePage`。"""
        return cast("MessagePage", super().create(items=items, total=total, page=page, size=size))


# 契约不因改名而漂移：把"分页里装的到底是什么"钉死。
assert get_args(MessagePage.model_fields["items"].annotation) == (MessageOut,), (
    "MessagePage.items 必须是 list[MessageOut]（会话消息），"
    f"实际是 {MessagePage.model_fields['items'].annotation!r}"
)


class ConversationOut(BaseModel):
    """会话（契约 5.7）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="会话 ID")
    kb_id: int | None = Field(default=None, description="绑定的知识库 ID；null 表示全局问答")
    user_id: int = Field(description="所属用户 ID")
    title: str = Field(description="会话标题，默认「新会话」")
    mode: SessionMode = Field(default="agent", description="会话模式：rag 或 agent")
    message_count: int = Field(
        default=0, ge=0, description="消息条数（冗余计数，避免列表页 N+1 查询）"
    )
    created_at: UtcDatetime = Field(default=None, description="创建时间，UTC ISO8601 带 Z")
    updated_at: UtcDatetime = Field(default=None, description="最后更新时间；会话列表按它倒序排列")


class ConversationCreateRequest(BaseModel):
    """新建会话（契约 5.7：`{kb_id?, title?}`）。"""

    model_config = ConfigDict(from_attributes=True)

    kb_id: int | None = Field(default=None, ge=1, description="绑定的知识库 ID；null = 全局问答")
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="会话标题；留空由服务端生成（如用首个问题摘要）",
    )


class FeedbackRequest(BaseModel):
    """提交反馈（契约 5.7：`{rating, comment?}`）。

    加了一条契约没写但必要的规则：**评分为 -1（没用）时要求写明原因**。
    点踩是改进检索质量最重要的信号，只有"踩了"而不知道为什么，
    这份数据没法用来定位问题（是召回错了？还是答案编了？）。
    """

    model_config = ConfigDict(from_attributes=True)

    rating: FeedbackRating = Field(description="评分：1 = 有用 / -1 = 没用 / 0 = 中立")
    comment: str | None = Field(
        default=None, max_length=512, description="补充说明；rating=-1 时必填"
    )

    @model_validator(mode="after")
    def _require_comment_on_downvote(self) -> FeedbackRequest:
        if self.rating == -1 and not (self.comment or "").strip():
            raise ValueError("点踩（rating=-1）时必须填写 comment，否则无法定位问题")
        return self


class FeedbackOut(BaseModel):
    """反馈结果（契约 5.7）。回显 `message_id` 让前端能精确更新那一条气泡的状态。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="反馈记录 ID")
    message_id: int = Field(description="被评价的消息 ID")
    user_id: int = Field(description="提交者用户 ID")
    rating: FeedbackRating = Field(description="评分：1 有用 / -1 没用 / 0 中立")
    comment: str | None = Field(default=None, description="补充说明")
    created_at: UtcDatetime = Field(default=None, description="提交时间，UTC ISO8601 带 Z")
