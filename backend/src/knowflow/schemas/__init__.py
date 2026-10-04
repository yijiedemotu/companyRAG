"""Pydantic 契约层：**接口字段的唯一真相来源**。

字段定义全部来自 `docs/01-数据库与接口契约.md` 第五节「REST 接口清单」，
本包不做任何数据库访问、不写业务规则——只负责三件事：

1. **校验入参**：把不合法的请求挡在业务代码之前（`chunk_overlap >= chunk_size`
   这类规则在 schema 层就报 422，而不是等到切分器把内存吃满）；
2. **约束出参**：响应模型是一道防泄漏闸门（`UserOut` 里没有 `password_hash`，
   粗心的 `model_validate(user)` 也泄不出去），同时决定 `/docs` 上显示什么；
3. **统一序列化**：时间一律走 `common.UtcDatetime` → `to_iso_z()` 输出带 `Z`，
   全项目只有一个时间格式，前端只写一套解析。

用法：

    from knowflow.schemas import ChatRequest, ChatResponse, Page

命名约定（契约文档第一节）：请求 `XxxRequest`、响应 `XxxOut`、分页 `Page[T]`。
"""

from __future__ import annotations

from knowflow.schemas.auth import (
    ACCOUNT_ROLES,
    LoginRequest,
    RegisterRequest,
    TokenOut,
    UserOut,
)
from knowflow.schemas.chat import (
    SSE_EVENT_NAMES,
    ChatRequest,
    ChatResponse,
    DoneEvent,
    EndEvent,
    ErrorEvent,
    GradingItem,
    MetaEvent,
    ReflectEvent,
    SourceOut,
    SourcesEvent,
    TokenEvent,
    ToolEvent,
    TraceEvent,
    UsageOut,
)

# `MessageOut` 重名的处理方式（顶层这个名字 = **会话消息**，不是提示文案）：
#
# 顶层 `knowflow.schemas.MessageOut` 指向契约 5.7 的会话消息（含 citations），
# 因为业务代码 99% 想要的是那个；common 里那个「一句提示文案」已重命名为
# `SimpleMessageOut`（旧名 `common.MessageOut` 保留为兼容别名）。
#
# 为什么不靠两个 `import MessageOut` 的先后顺序：ruff 的 isort 会按字母序重排，
# 谁在前谁在后完全取决于模块名，改一次 import 就可能**静默把 MessageOut 换回
# 提示文案**（Pydantic 不报错，只在前端访问 `.citations` 时才炸）。
# 所以这里只在一个地方绑定它：`from ...conversation import MessageOut as MessageOut`
# （`X as X` 是声明式重导出，静态检查器认它是本模块的公开 API），
# common 的同名名字则以 `_CommonMessageOut` 形式留存，**两条路不会打架**。
from knowflow.schemas.common import (
    ErrorBody,
    ErrorEnvelope,
    MessageOut as _CommonMessageOut,
    OkOut,
    Page,
    PageParams,
    SessionMode,
    SimpleMessageOut,
    UtcDatetime,
)

# 本行就是上面说的"唯一绑定处"：`X as X` = 声明式重导出，静态检查器认它是
# 本模块的公开 API（因此 `from knowflow.schemas import MessageOut` 有类型）。
from knowflow.schemas.conversation import (
    CitationOut,
    ConversationCreateRequest,
    ConversationOut,
    FeedbackOut,
    FeedbackRequest,
    MessageOut as MessageOut,
    MessagePage,
    MessageRole,
)
from knowflow.schemas.document import (
    ChunkOut,
    DocumentOut,
    DocumentStatus,
)
from knowflow.schemas.evaluation import (
    AblationGroup,
    AblationResponse,
    CaseIn,
    CompareResponse,
    DatasetCreateRequest,
    DatasetOut,
    EvalCaseOut,
    EvalCaseResultOut,
    EvalMetrics,
    EvalRunOut,
    RunCreateRequest,
    RunMode,
    RunStatus,
)
from knowflow.schemas.knowledge_base import (
    KBCreateRequest,
    KBOut,
    KBStatsOut,
    KBUpdateRequest,
)
from knowflow.schemas.observability import (
    ByModelOut,
    ByModeOut,
    DBHealthOut,
    HealthOut,
    LatencyOut,
    ObsStatsOut,
    PoolHealthOut,
    QualityOut,
    TimelinePointOut,
    TokenStatsOut,
    TraceDetailOut,
    TraceOut,
    TraceSpanOut,
)
from knowflow.schemas.search import (
    SearchDebug,
    SearchGate,
    SearchHit,
    SearchRequest,
    SearchResponse,
)

# 绑定错了就会静默拿错模型（`{message: str}` 那个），所以这里自证一次。
# 用 import 期断言而不是写测试：一旦有人改坏，第一次 import 就炸，不必等 CI。
assert MessageOut is not _CommonMessageOut, (
    "knowflow.schemas.MessageOut 必须是 conversation.MessageOut（会话消息），"
    "不能是 common 那个提示文案"
)
assert "citations" in MessageOut.model_fields, (
    "knowflow.schemas.MessageOut 缺少 citations，拿到的不是契约 5.7 的会话消息"
)
assert "citations" not in SimpleMessageOut.model_fields, (
    "SimpleMessageOut 不该有 citations（它只是一句提示文案）"
)

# 全局按字母序排列（ruff 的 RUF022 保证），因此同一业务模块的模型可能不挨着。
# 要按模块浏览请直接看上面的 import 分组。
__all__ = [
    # auth
    "ACCOUNT_ROLES",
    # chat（含 SSE 事件）
    "SSE_EVENT_NAMES",
    # evaluation
    "AblationGroup",
    "AblationResponse",
    # observability
    "ByModeOut",
    "ByModelOut",
    "CaseIn",
    "ChatRequest",
    "ChatResponse",
    # document
    "ChunkOut",
    # conversation
    "CitationOut",
    "CompareResponse",
    "ConversationCreateRequest",
    "ConversationOut",
    "DBHealthOut",
    "DatasetCreateRequest",
    "DatasetOut",
    "DocumentOut",
    "DocumentStatus",
    "DoneEvent",
    "EndEvent",
    # common
    "ErrorBody",
    "ErrorEnvelope",
    "ErrorEvent",
    "EvalCaseOut",
    "EvalCaseResultOut",
    "EvalMetrics",
    "EvalRunOut",
    "FeedbackOut",
    "FeedbackRequest",
    "GradingItem",
    "HealthOut",
    # knowledge_base
    "KBCreateRequest",
    "KBOut",
    "KBStatsOut",
    "KBUpdateRequest",
    "LatencyOut",
    "LoginRequest",
    # conversation：会话聊天消息（契约 5.7），不是提示文案那个
    "MessageOut",
    "MessagePage",
    "MessageRole",
    "MetaEvent",
    "ObsStatsOut",
    "OkOut",
    "Page",
    "PageParams",
    "PoolHealthOut",
    "QualityOut",
    "ReflectEvent",
    "RegisterRequest",
    "RunCreateRequest",
    "RunMode",
    "RunStatus",
    # search
    "SearchDebug",
    "SearchGate",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "SessionMode",
    # common（SimpleMessageOut = 原 common.MessageOut，一句提示文案）
    "SimpleMessageOut",
    "SourceOut",
    "SourcesEvent",
    "TimelinePointOut",
    "TokenEvent",
    "TokenOut",
    "TokenStatsOut",
    "ToolEvent",
    "TraceDetailOut",
    "TraceEvent",
    "TraceOut",
    "TraceSpanOut",
    "UsageOut",
    "UserOut",
    "UtcDatetime",
]
