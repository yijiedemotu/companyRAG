"""对话契约：`POST /chat`、`POST /chat/stream` 与 9 个 SSE 事件。

字段来源：契约文档 5.5（对话）与 5.6（SSE 事件流）。

本模块是整个项目最核心的契约文件，有三个决定值得写在前面：

1. **`ChatResponse` 字段与契约 5.5 节逐字一致**，一个都不多、一个都不少。
   前端的类型定义是照着契约手写的，多一个字段会让人以为"后端会返回这个"，
   少一个字段会让前端拿到 `undefined` 然后在渲染时崩(`refusal` 少一个就会
   把拒答气泡画成正常答案——这是最严重的一类静默错误)。
2. **`DoneEvent` 就是 `ChatResponse` 的别名**，不是复制一份字段。
   契约 5.6 明确规定 `done` 帧的 data 是「`ChatResponse` 的完整字段」。
   复制一份字段定义，两边迟早漂移（改了 ChatResponse 忘了改 DoneEvent，
   流式和非流式返回的结构就不一样了，前端要写两套解析）。
3. **`usage` / `cost_usd` / `latency_ms` / `model` / `offline` 必须返回**：
   这是「这个项目不骗人」的另一条硬规则——离线降级（没配 API Key）时
   `offline=true` 且 `model` 说明实际用了什么，用户一眼能看出答案不是大模型写的，
   而不是被一个看似正常的答案骗过去。
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from knowflow.core.config import ChatMode, get_settings

__all__ = [
    "SSE_EVENT_NAMES",
    "ChatRequest",
    "ChatResponse",
    "DoneEvent",
    "EndEvent",
    "ErrorEvent",
    "GradingItem",
    "MetaEvent",
    "ReflectEvent",
    "SourceOut",
    "SourcesEvent",
    "TokenEvent",
    "ToolEvent",
    "TraceEvent",
    "UsageOut",
]

# 默认值取服务端配置，避免"接口默认 5、服务端默认 3"这类不一致。
_DEFAULTS = get_settings()


class UsageOut(BaseModel):
    """token 用量（契约 5.5：`{prompt_tokens, completion_tokens, total_tokens}`）。"""

    model_config = ConfigDict(from_attributes=True)

    prompt_tokens: int = Field(default=0, ge=0, description="输入 token 数")
    completion_tokens: int = Field(default=0, ge=0, description="输出 token 数")
    total_tokens: int = Field(default=0, ge=0, description="合计 token 数")


class SourceOut(BaseModel):
    """答案引用的一个来源（契约 5.5 的 `sources[]`）。"""

    model_config = ConfigDict(from_attributes=True)

    rank: int = Field(ge=1, description="引用序号，从 1 开始，对应答案正文里的 [n]")
    chunk_id: int = Field(description="切片 ID，点击可回到原文")
    doc_id: int = Field(description="所属文档 ID")
    doc_name: str = Field(description="文档文件名，前端直接展示")
    page_no: int | None = Field(default=None, description="页码，仅 PDF 有值")
    section_path: str | None = Field(default=None, description="所属小节路径")
    score: float = Field(default=0.0, description="相关度得分，越大越相似")
    snippet: str = Field(default="", description="命中片段摘要")


class GradingItem(BaseModel):
    """一条「这块内容是否真的回答了问题」的判定（契约 5.5 的 `grading[]`）。

    这是 Agent 模式的关键动作：检索到的东西**不一定相关**，先判定再生成，
    比"喂进去让它自己忽略"可靠得多（模型很擅长被无关上下文带偏）。
    """

    model_config = ConfigDict(from_attributes=True)

    chunk_id: int = Field(description="被评估的切片 ID")
    relevant: bool = Field(description="是否与问题相关")
    reason: str = Field(default="", description="判定理由，给人看的短句")


class ChatRequest(BaseModel):
    """问答请求（契约 5.5）。`/chat` 与 `/chat/stream` 共用同一个请求体。"""

    model_config = ConfigDict(from_attributes=True)

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="用户问题，1–4000 字符",
    )
    kb_id: int | None = Field(
        default=None,
        ge=1,
        description="限定检索的知识库 ID；null = 全局检索（跨所有启用的 KB）",
    )
    conversation_id: int | None = Field(
        default=None,
        ge=1,
        description="已有会话 ID；null = 新建会话（响应里会返回真实的 conversation_id）",
    )
    mode: ChatMode = Field(
        default="agent",
        description="对话模式：rag（单轮检索即答）或 agent（带相关性判定/改写/自省）",
    )
    top_k: int = Field(
        default=_DEFAULTS.top_k,
        ge=1,
        le=50,
        description="送给模型的知识片段条数，1–50",
    )
    use_rerank: bool = Field(
        default=_DEFAULTS.rerank_enabled,
        description="是否对召回结果重排（更准，但多一次模型调用/耗时）",
    )
    use_memory: bool = Field(
        default=True,
        description="是否带上历史对话记忆；关闭则视为单轮问答",
    )


class ChatResponse(BaseModel):
    """问答响应（契约 5.5 节，字段逐字对应）。

    所有 `usage` / `cost_usd` / `latency_ms` / `model` / `offline` 字段都是
    **必填**（有默认值但一定会被写入）：前端要把它们展示在答案下方，
    缺失会直接渲染成空白，看起来像后端出了 bug。
    """

    model_config = ConfigDict(from_attributes=True)

    conversation_id: int = Field(description="会话 ID；请求未指定时是新建的会话 ID")
    message_id: int = Field(description="本次助手消息的 ID，反馈接口要它")
    trace_id: str = Field(description="本次请求的链路追踪 ID，可在 /obs/traces 查到全过程")
    answer: str = Field(default="", description="答案正文，引用标记形如 [1]")
    sources: list[SourceOut] = Field(
        default_factory=list, description="引用来源，rank 与正文 [n] 一一对应"
    )
    refusal: bool = Field(
        default=False,
        description="是否因知识库无相关内容而拒答；true 时 answer 是说明文案而非答案",
    )
    retrieval_rounds: int = Field(
        default=0, ge=0, description="实际检索轮数；agent 模式改写后可能 > 1"
    )
    reflect_passed: bool = Field(
        default=True, description="自省是否通过；false 表示答案有未被支撑的内容"
    )
    grading: list[GradingItem] = Field(
        default_factory=list, description="逐块相关性判定明细，agent 模式且有开启时非空"
    )
    rewritten_queries: list[str] = Field(
        default_factory=list,
        description="本轮的查询改写结果（含原始问题），用于解释为什么召回了这些片段",
    )
    usage: UsageOut = Field(default_factory=UsageOut, description="token 用量")
    cost_usd: float = Field(default=0.0, ge=0, description="本次调用费用（美元）")
    latency_ms: int = Field(default=0, ge=0, description="端到端耗时（毫秒）")
    model: str = Field(default="", description="实际使用的模型名；离线或纯检索时为说明性取值")
    offline: bool = Field(
        default=False,
        description="是否处于离线模式（未配置 API Key）；true 时不骗人，前端要显著提示",
    )


# --------------------------------------------------------------------------------------
# SSE 事件模型（契约 5.6）
# --------------------------------------------------------------------------------------
# 帧格式 `event: <name>\ndata: <json>\n\n`，事件名由后端产帧与测试共用一份常量，
# 避免测试里手写字符串拼错（"tokken" 这种错字不会报错，只会让前端什么都不显示）。
# --------------------------------------------------------------------------------------

#: 9 个 SSE 事件名，顺序即契约 5.6 的流顺序。
SSE_EVENT_NAMES: Final[tuple[str, ...]] = (
    "meta",
    "trace",
    "tool",
    "sources",
    "reflect",
    "token",
    "done",
    "error",
    "end",
)


class MetaEvent(BaseModel):
    """`meta`：第一帧，前端靠它建气泡并知道这次用的什么模式/模型。"""

    model_config = ConfigDict(from_attributes=True)

    conversation_id: int = Field(description="会话 ID（新建时是刚创建的 ID）")
    mode: ChatMode = Field(description="本次对话模式：rag 或 agent")
    trace_id: str = Field(description="链路追踪 ID")
    model: str = Field(default="", description="本次要用的模型名")
    offline: bool = Field(default=False, description="是否离线模式")
    embedding_mode: str = Field(
        default="",
        description="向量化实际模式，降级时形如 hash(fallback:local_load_failed)",
    )


class TraceEvent(BaseModel):
    """`trace`：图节点开始/结束，前端据此画「思考过程」。"""

    model_config = ConfigDict(from_attributes=True)

    node: str = Field(description="节点名，如 analyze/retrieve/grade/rewrite/generate/reflect")
    status: str = Field(default="ok", description="节点状态：ok 或 error")
    duration_ms: int = Field(default=0, ge=0, description="节点耗时（毫秒）")
    detail: Any | None = Field(
        default=None, description="节点附加信息（如候选条数），结构随节点而定"
    )


class ToolEvent(BaseModel):
    """`tool`：工具（检索）调用，让用户看到"它在查什么、查到几条"。"""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(description="工具名，如 search_knowledge_base")
    args: Any | None = Field(default=None, description="调用参数（已截断），如 query/top_k")
    result_count: int = Field(default=0, ge=0, description="返回条数")
    duration_ms: int = Field(default=0, ge=0, description="调用耗时（毫秒）")


class SourcesEvent(BaseModel):
    """`sources`：引用来源，**先于 token 推送**，让用户立刻有反馈而不是干等。"""

    model_config = ConfigDict(from_attributes=True)

    sources: list[SourceOut] = Field(default_factory=list, description="引用来源列表")


class ReflectEvent(BaseModel):
    """`reflect`：自省结果。`action` 说明下一步动作，前端可展示"正在改写重查"。

    `unsupported` 的元素是**对象**（`{"sentence": "...", "reason": "..."}` 这类形状，
    由判官模型输出、原样透传给前端），不是纯字符串。契约 5.6 只写了 `[...]`，
    这里按 Agent 实际产出的结构定成 `list[dict]` 并容忍裸字符串——
    两者用校验器统一成同一种形状，避免前端要为两种数据写两套渲染。

    **`action` 只用于前端展示，不做取值校验**（类型就是 `str`，不是 `Literal`）。
    这是刻意的：它是本帧里最可能新增取值（如 `abort`）的字段，
    一旦用 `Literal` 校验，后端加一个新动作就会让**整帧 422**，
    而 SSE 一帧失败意味着用户只能看到半截答案——一个展示字段不该有这种权力。
    """

    model_config = ConfigDict(from_attributes=True)

    round: int = Field(default=1, ge=1, description="第几轮自省，从 1 开始")
    passed: bool = Field(description="本轮是否通过")
    unsupported: list[dict[str, Any]] = Field(
        default_factory=list,
        description="答案中未被检索内容支撑的陈述，每项形如 {sentence, reason}",
    )
    action: str = Field(
        default="",
        description=(
            "下一步动作，当前已知取值 accept / regenerate；"
            "**仅用于前端展示，后端可自由新增取值，不做校验**"
        ),
    )

    @field_validator("unsupported", mode="before")
    @classmethod
    def _normalize_unsupported(cls, value: Any) -> Any:
        """把裸字符串包装成 `{"sentence": ...}`，非字符串非字典的项丢掉。

        宽松处理的原因：判官模型输出的是自由 JSON，历史版本/不同模型可能给出
        `["某句话缺依据"]` 这种数组。**因为一帧格式不符就让整条 SSE 流报错断掉，
        代价远大于宽容处理**——用户已经看到答案了，不该在收尾时炸。
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [{"sentence": value}]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            normalized: list[dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append({"sentence": item})
            return normalized
        return []


class TokenEvent(BaseModel):
    """`token`：增量文本。前端逐帧追加即可，不要整段替换。"""

    model_config = ConfigDict(from_attributes=True)

    text: str = Field(default="", description="本次增量文本片段")


#: `done` 帧的 data 就是 `ChatResponse` 的完整字段（契约 5.6）。
#: 用别名而不是新建模型：字段定义只有一处，永远不会和 /chat 的返回漂移。
DoneEvent = ChatResponse


class ErrorEvent(BaseModel):
    """`error`：出错帧，之后必然跟一帧 `end`。字段与统一错误信封对齐。"""

    model_config = ConfigDict(from_attributes=True)

    code: str = Field(description="错误码，与统一错误信封的 code 一致")
    message: str = Field(description="中文错误信息")
    request_id: str | None = Field(default=None, description="请求 ID，便于排查")


class EndEvent(BaseModel):
    """`end`：关闭流。始终是最后一帧，前端收到它就收尾（无论成功或出错）。"""

    model_config = ConfigDict(from_attributes=True)
