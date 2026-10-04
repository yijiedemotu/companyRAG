"""全部 Prompt 模板集中在这一处。

**为什么不散落在各模块**：Prompt 是这个项目的"产品逻辑"，改一句话就可能让
幻觉率翻倍。集中放置才能被 review、被 diff、被测试引用（测试会断言
"防幻觉那句必须还在"）。散落各处的 Prompt 最终会互相矛盾。

模板设计的三条规则（每条都有踩坑背景）：

1. **给编号、要求引用**：上下文用 `[1] [2]` 编号，要求模型在句末标 `[n]`。
   这样答案可溯源，也让"没依据的句子"在结构上就显眼（前端能高亮没有引用的句子）。
2. **明确"资料不足"的出口**：必须告诉模型**怎么拒答**。
   只写"不许编"是不够的——模型会退而求其次去编一个"看起来安全"的答案。
   所以给出固定话术，并让链路能识别它（`REFUSAL_MARKERS`）。
3. **把格式约束写成可解析的样子**：需要 JSON 的地方要求"只输出 JSON"，
   并给出**带字段的示例**。经验：给示例比写十行规则有效。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from knowflow.llm.base import ChatMessage

# --------------------------------------------------------------------------------------
# 拒答标记：链路靠它判断"模型是否说了没有依据"
# 注意：判定必须**同时**看这几个特征，单看某一句话会误判
# （例如模型答"知识库中没有提到年假，但根据差旅制度…"这种半拒答）。
# --------------------------------------------------------------------------------------
REFUSAL_MARKERS: Final[tuple[str, ...]] = (
    "知识库中没有",
    "资料中没有",
    "没有找到相关",
    "未找到相关",
    "无法从提供的资料",
    "提供的信息不足以",
    "暂无相关信息",
)

REFUSAL_ANSWER: Final[str] = (
    "知识库中没有找到与该问题相关的内容，因此我无法给出有依据的回答。"
    "建议换一种问法，或确认相关文档已经上传。"
)

# --------------------------------------------------------------------------------------
# 1. RAG 问答
# --------------------------------------------------------------------------------------
RAG_SYSTEM_PROMPT: Final[
    str
] = """你是一个严谨的企业知识库问答助手。你的唯一信息来源是用户提供的「参考资料」。

必须遵守的规则：
1. 只依据参考资料回答，**不得使用你自己的先验知识**补充任何事实。
2. 每一条事实性陈述后面必须标注来源编号，格式为 [1]、[2]（可多个： [1][3]）。
3. 如果参考资料不足以回答问题，直接回复：「知识库中没有找到与该问题相关的内容，因此我无法给出有依据的回答。建议换一种问法，或确认相关文档已经上传。」**不要猜测，不要部分编造。**
4. 如果参考资料之间互相矛盾，要指出矛盾，并分别标注来源，不要替用户做选择。
5. 回答简洁、直接，不要复述问题，不要写「根据参考资料」这类开场白。
6. 使用与提问相同的语言回答。"""

CONTEXT_HEADER: Final[str] = "以下是参考资料（每条以 [编号] 开头）："
QUESTION_HEADER: Final[str] = "用户问题："


def format_contexts(contexts: Sequence[tuple[int, str, str]]) -> str:
    """把检索结果拼成带编号的上下文块。

    参数是 `(编号, 出处描述, 正文)` 三元组。
    出处描述会出现在模型眼前——**这一步很关键**：让模型看到
    「员工报销制度 > 差旅报销标准」这种小节路径，它能判断该引用哪一段，
    引用准确率明显高于只给正文。
    """
    blocks: list[str] = []
    for rank, source, text in contexts:
        blocks.append(f"[{rank}] （出处：{source}）\n{text.strip()}")
    return "\n\n".join(blocks)


def build_rag_messages(
    *,
    question: str,
    contexts: Sequence[tuple[int, str, str]],
    history: Sequence[ChatMessage] | None = None,
    extra_instruction: str | None = None,
) -> list[ChatMessage]:
    """组装 RAG 问答的消息列表。

    `extra_instruction` 是 Self-RAG 反思后重试时用的：第一次生成没有通过
    引用校验时，把「上一轮有句子缺少引用支撑」这句追加进去，让模型修正。
    **这是"自我纠正"能生效的关键**——不告诉模型错在哪，重试只是重新掷骰子。
    """
    system = RAG_SYSTEM_PROMPT
    if extra_instruction:
        system = f"{system}\n\n本次额外要求（上一轮未通过校验）：\n{extra_instruction}"

    messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]
    if history:
        # 历史只保留 user/assistant 两种角色，避免把上一轮的 system 重复塞进来
        messages.extend(m for m in history if m.role in ("user", "assistant"))
    messages.append(
        ChatMessage(
            role="user",
            content=f"{CONTEXT_HEADER}\n\n{format_contexts(contexts)}\n\n{QUESTION_HEADER}{question}",
        )
    )
    return messages


# --------------------------------------------------------------------------------------
# 1b. 无需检索时的直答（闲聊 / 系统自身能力询问）
# --------------------------------------------------------------------------------------
DIRECT_SYSTEM_PROMPT: Final[str] = """你是一个企业知识库问答助手。

当前这个问题不需要查询知识库（属于闲聊或询问系统自身能力）。请直接、简洁地回答。
如果用户在问你能做什么：说明你可以基于已上传的企业文档回答问题，并会给出原文出处。
不要编造任何关于公司制度、产品、流程的具体事实——那些必须走检索。"""


def build_direct_messages(
    *, question: str, history: Sequence[ChatMessage] | None = None
) -> list[ChatMessage]:
    """`analyze` 判定不需要检索时走这条路（省一次检索、也避免无意义的引用）。"""
    messages: list[ChatMessage] = [ChatMessage(role="system", content=DIRECT_SYSTEM_PROMPT)]
    if history:
        messages.extend(m for m in history if m.role in ("user", "assistant"))
    messages.append(ChatMessage(role="user", content=question))
    return messages


# --------------------------------------------------------------------------------------
# 2. CRAG：检索结果相关性判定（grade 节点）
# --------------------------------------------------------------------------------------
GRADING_SYSTEM_PROMPT: Final[
    str
] = """你是一个检索质量评审员。给你一个问题和若干条检索到的资料片段，你要判断每条片段「是否包含回答该问题所需的信息」。

判断标准（按重要性排序）：
1. 片段必须直接涉及问题所问的主题，而不是只在字面上有共同词；
2. 「主题相关但缺少具体数值/结论」算不相关（例如问"住宿标准多少"，片段只讲"差旅费需审批"）；
3. 宁可判为不相关，也不要为了让答案好看而放行——放行一条无用片段会让模型编出错误答案。

只输出 JSON，不要任何解释文字，格式如下：
{"items": [{"id": 1, "relevant": true, "reason": "直接给出住宿标准数值"}, {"id": 2, "relevant": false, "reason": "只提到报销流程，没有标准数值"}]}"""


def build_grading_messages(
    *, question: str, contexts: Sequence[tuple[int, str]]
) -> list[ChatMessage]:
    """`contexts` 是 `(编号, 正文)`。"""
    body = "\n\n".join(f"[{rank}] {text.strip()}" for rank, text in contexts)
    return [
        ChatMessage(role="system", content=GRADING_SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"问题：{question}\n\n待判定资料：\n\n{body}"),
    ]


# --------------------------------------------------------------------------------------
# 2b. 重排（rerank）
# --------------------------------------------------------------------------------------
RERANK_SYSTEM_PROMPT: Final[
    str
] = """你是一个检索结果重排器。给你一个问题和若干条候选资料片段，请按「对回答该问题的有用程度」给每条打 0-10 分。

打分标准：
10 = 直接包含问题的答案（数值、结论、明确条款）
 7 = 高度相关，是答案所在的章节，但需要结合其它片段
 4 = 主题相关，但对回答这个问题帮助有限
 1 = 只是字面上有共同词，主题不同
 0 = 完全无关，或与问题问的方面相反（例如问"住宿标准"却给了"餐饮标准"）

只输出 JSON，不要解释文字，格式：
{"scores": [{"id": 1, "score": 9}, {"id": 2, "score": 2}]}"""


def build_rerank_messages(*, query: str, contexts: Sequence[tuple[int, str]]) -> list[ChatMessage]:
    """`contexts` 是 `(编号, 正文)`。用片段正文而不是父块，省 token。"""
    body = "\n\n".join(f"[{rank}] {text.strip()}" for rank, text in contexts)
    return [
        ChatMessage(role="system", content=RERANK_SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"问题：{query}\n\n候选资料：\n\n{body}"),
    ]


# --------------------------------------------------------------------------------------
# 3. 查询改写（rewrite 节点）
# --------------------------------------------------------------------------------------
REWRITE_SYSTEM_PROMPT: Final[
    str
] = """你是一个检索查询优化器。用户的问题在关键词检索上效果不好，请把它改写成更适合检索的形式。

改写策略（按需选择，可以同时用）：
1. 去掉口语化成分（"请问一下那个…"）和指代词（"它"、"那个"→补全为具体名词）；
2. 补充同义词与近义表达（"房费"→"住宿费 住宿标准"）；
3. 如果问题需要多个信息源，拆成 1-3 个子问题，每个子问题独立成行；
4. 保留原始问题里的专有名词、错误码、型号、数字（这些是不可替换的锚点）。

只输出 JSON，格式：
{"queries": ["改写后的查询1", "改写后的查询2"], "reason": "一句话说明改了什么"}"""


def build_rewrite_messages(
    *, question: str, history_summary: str | None = None, attempt: int = 1
) -> list[ChatMessage]:
    """`attempt` 从 1 开始。第 2 次改写时会提示模型"上次改写仍不够"，让它换策略。"""
    hint = ""
    if attempt >= 2:
        hint = (
            "\n\n注意：这是第 2 次改写，上一次改写后仍然没有检索到有用资料。"
            "请换一种策略——优先拆分成更小的子问题，或改用更宽泛的上位词。"
        )
    user = f"原始问题：{question}"
    if history_summary:
        user += f"\n\n对话上文（用于补全指代词）：{history_summary}"
    return [
        ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT + hint),
        ChatMessage(role="user", content=user),
    ]


# --------------------------------------------------------------------------------------
# 4. 问题分析（analyze 节点）
# --------------------------------------------------------------------------------------
ANALYZE_SYSTEM_PROMPT: Final[
    str
] = """你是一个问答路由器。判断用户的问题是否需要在企业知识库里检索资料。

需要检索：涉及公司制度、产品文档、业务数据、流程规范等私有信息的问题。
不需要检索：纯闲聊、纯数学计算、纯代码语法、以及明确不需要外部信息的问题。

只输出 JSON，格式：
{"needs_retrieval": true, "queries": ["用于检索的查询"], "question_type": "factual", "reason": "一句话理由"}

`question_type` 取值：factual（事实型）/ multi_hop（需要多跳推理）/ chitchat（闲聊）/ meta（询问系统自身能力）。"""


def build_analyze_messages(
    *, question: str, history_summary: str | None = None
) -> list[ChatMessage]:
    user = f"用户问题：{question}"
    if history_summary:
        user += f"\n\n对话上文：{history_summary}"
    return [
        ChatMessage(role="system", content=ANALYZE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


# --------------------------------------------------------------------------------------
# 5. Self-RAG：答案引用支撑检查（reflect 节点）
# --------------------------------------------------------------------------------------
REFLECT_SYSTEM_PROMPT: Final[
    str
] = """你是一个答案审核员。给你「参考资料」和「待审核答案」，检查答案里每一条事实性陈述是否都能在参考资料中找到支撑。

判定规则：
1. 逐句检查。只对**包含事实主张**的句子判定（寒暄、过渡句不算）；
2. 句末有 [n] 且 [n] 对应的资料确实支持该句 → supported；
3. 句末没有引用编号，或有引用但资料不支持 → unsupported；
4. 数值、日期、金额必须与资料**完全一致**，不一致就是 unsupported（这是最常见的错误类型）。

只输出 JSON，格式：
{"passed": true, "unsupported": [{"sentence": "原句", "reason": "资料中未出现该数值"}], "overall": "一句话总评"}

`passed` 为 true 当且仅当 `unsupported` 为空。"""


def build_reflect_messages(
    *, answer: str, contexts: Sequence[tuple[int, str]]
) -> list[ChatMessage]:
    body = "\n\n".join(f"[{rank}] {text.strip()}" for rank, text in contexts)
    return [
        ChatMessage(role="system", content=REFLECT_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"参考资料：\n\n{body}\n\n待审核答案：\n\n{answer}",
        ),
    ]


# --------------------------------------------------------------------------------------
# 6. 评测用判官（LLM as judge）
# --------------------------------------------------------------------------------------
FAITHFULNESS_SYSTEM_PROMPT: Final[
    str
] = """你是评测判官。评估「答案」在多大程度上由「参考资料」支撑。

输出 0 到 1 之间的一个小数：
1.0 = 每一句事实主张都能在资料中找到依据；
0.5 = 一半左右有依据；
0.0 = 完全无依据（或答案在编造）。

只输出 JSON：{"score": 0.0, "reason": "一句话理由"}"""

ANSWER_RELEVANCE_SYSTEM_PROMPT: Final[
    str
] = """你是评测判官。评估「答案」在多大程度上回答了「问题」。

考虑：是否正面回答了问题、是否覆盖了问题要求的全部要点、有没有跑题。
即使答案没有依据（可能编造），只要它正面回答了问题，相关性也可以高——依据问题由 faithfulness 指标负责。

输出 0 到 1 之间的一个小数。只输出 JSON：{"score": 0.0, "reason": "一句话理由"}"""


def build_judge_messages(
    *, kind: str, question: str, answer: str, contexts: Sequence[tuple[int, str]] = ()
) -> list[ChatMessage]:
    """`kind` 取 `faithfulness` 或 `answer_relevance`。"""
    if kind == "faithfulness":
        system = FAITHFULNESS_SYSTEM_PROMPT
        body = "\n\n".join(f"[{rank}] {text.strip()}" for rank, text in contexts)
        user = f"问题：{question}\n\n参考资料：\n\n{body}\n\n答案：\n\n{answer}"
    elif kind == "answer_relevance":
        system = ANSWER_RELEVANCE_SYSTEM_PROMPT
        user = f"问题：{question}\n\n答案：\n\n{answer}"
    else:
        raise ValueError(f"未知的判官类型: {kind!r}")
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


# --------------------------------------------------------------------------------------
# 7. 离线（无 API Key）抽取式回答用的提示片段
# --------------------------------------------------------------------------------------
OFFLINE_NOTICE: Final[str] = (
    "（离线模式：未配置大模型 API Key，以下答案由检索结果直接抽取生成，"
    "仅用于验证链路是否连通，不代表真实生成质量。）"
)


def detect_refusal(answer: str) -> bool:
    """判断一段回答是否属于"拒答"。

    规则：命中任一 `REFUSAL_MARKERS` 且答案很短（< 200 字）才判为拒答。
    加长度条件是因为长答案里出现"没有找到相关"可能只是在描述某个流程——
    只按关键词判会把正常答案误标成拒答，污染拒答率指标。
    """
    text = answer.strip()
    if not text or len(text) > 200:
        return False
    return any(marker in text for marker in REFUSAL_MARKERS)


__all__ = [
    "ANALYZE_SYSTEM_PROMPT",
    "ANSWER_RELEVANCE_SYSTEM_PROMPT",
    "CONTEXT_HEADER",
    "DIRECT_SYSTEM_PROMPT",
    "FAITHFULNESS_SYSTEM_PROMPT",
    "GRADING_SYSTEM_PROMPT",
    "OFFLINE_NOTICE",
    "QUESTION_HEADER",
    "RAG_SYSTEM_PROMPT",
    "REFUSAL_ANSWER",
    "REFUSAL_MARKERS",
    "REFLECT_SYSTEM_PROMPT",
    "RERANK_SYSTEM_PROMPT",
    "REWRITE_SYSTEM_PROMPT",
    "build_analyze_messages",
    "build_direct_messages",
    "build_grading_messages",
    "build_judge_messages",
    "build_rag_messages",
    "build_reflect_messages",
    "build_rerank_messages",
    "build_rewrite_messages",
    "detect_refusal",
    "format_contexts",
]
