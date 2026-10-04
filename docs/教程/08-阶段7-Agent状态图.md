# 阶段 7：Agent 状态图

> **本阶段目标**：把控制流**画成图**——6 个节点、4 条条件边，模型只在节点内部做判断；每个节点都带降级分支。
> **预计耗时**：150 分钟
> **前置阶段**：阶段 6
> **本阶段产出**：`agent/` 七个模块（2,065 行）+ 可回溯的 checkpointer + 双重截断的会话记忆

## 0. 为什么先做这一步

因为**控制流归谁，决定了系统能不能被调试**（`docs/00` 第 1.1 节）。

| 形态 | 控制流由谁决定 | 适用 | 本项目 |
| --- | --- | --- | --- |
| 固定 RAG 链 | 代码写死（先检索→再生成） | 单跳事实型问答，占 80% 流量 | 作为**快路径**保留（`build_fixed_chain`） |
| Agent（工具调用） | 模型决定调不调、调几次 | 需要多跳、需要算术/查库 | 图的主力路径 |
| **Workflow（状态图）** | **代码定义骨架，模型只填节点内部** | 需要可控、可观测、可断言 | **本项目的 Agent 就是 LangGraph 状态图** |

> 面试话术：**「我没有把控制流交给模型，我把控制流画成了图；
> 模型只在图的每个节点内部做判断。这样既有自适应的好处，又保证每一步可观测、可兜底、可测试。」**

第二个理由：**这一步是"服务层能不能薄"的前提**。阶段 8 的 `ChatService`
只是"给图喂一个 `session`、把结果落库"——**图本身必须先能脱离数据库独立跑**。
本阶段的所有验证都是这么做的（`VECTOR_BACKEND=memory` + `MockChatModel`，不碰 MySQL）。

第三个理由：**降级分支不是洁癖，是必需**。`agent/nodes.py` 开头写得很直白：

```python
# backend/src/knowflow/agent/nodes.py L17-L19
**所有节点都有"模型抽风也要能继续"的降级分支**。这不是防御性编程洁癖：
LLM 输出 JSON 失败是常态（实测约 2%~5%），如果每次失败都中断整个问答，
用户体验会是"每 30 次提问就报一次错"。正确的做法是降级 + 记录 + 继续。
```

**"约 2%~5%"这个数字请记住**：它意味着如果你只测一遍，几乎永远测不到降级分支；
而它在上千次提问里会发生几十次。**降级分支是线上分支，不是防御性代码。**

---

## 1. 要解决的问题

不做这一步（或做错了），你会遇到的具体失败现象：

| 现象 | 真实原因 |
| --- | --- |
| 第二轮提问的引用里混进了第一轮的内容 | `thread_id` 用了 `conversation_id`，而 `candidates` 是**累加** reducer |
| "过滤掉不相关的片段"这个操作**完全无效** | `grade` 返回了过滤后的 `candidates`，被 reducer 又合并回来（LangGraph 最大的坑） |
| prompt 里出现两个引用编号指向同一段原文 | 多轮检索重复捞到同一个 chunk，`candidates` 没用去重 reducer |
| token 成本少算了后面几轮 | `usage` 用覆盖语义而不是累加 |
| 一次提问变成 20 次模型调用（烧钱） | 工具循环没有硬上限 |
| 用本地小模型时 Agent 直接报错 | 模型不支持 function calling，却没有降级路径 |
| 流式接口每次都发 `error` 帧，`done` 永远不出现 | **contextvars 跨线程**：`Token was created in a different Context`（E6.5） |
| 会话列表的消息数比实际少 | 用 `threading.Lock` 而不是 `SELECT ... FOR UPDATE`，多进程下失效 |
| 第 20 轮提问的 prompt 是第 1 轮的 20 倍 | 记忆只按轮数截断，用户粘一篇 3000 字就爆 |
| 前端显示一个"空气泡" | 图跑完却没有答案（某节点静默失败），返回了空字符串 |
| 排障时不知道"那次请求走到了哪一步" | 没有 checkpointer，图的中间状态全丢 |
| 明明不该检索的闲聊也去查库了 | `analyze` 的 JSON 解析失败时**默认不检索**（方向反了） |

---

## 2. 动手实现

### 2.1 目标

```
                     ┌──────────────┐
      START ────────►│   analyze    │  判断问题类型 + 是否需要检索 + 生成查询
                     └──────┬───────┘
                            │ needs_retrieval?
                 ┌──────────┴───────────┐
                no                     yes
                 │                       ▼
                 │              ┌──────────────┐
                 │     ┌───────►│   retrieve   │  工具循环（模型驱动，有硬上限）
                 │     │        └──────┬───────┘
                 │     │               ▼
                 │     │        ┌──────────────┐
                 │     │        │    grade     │  CRAG：逐条判相关性 → relevant_ids
                 │     │        └──────┬───────┘
                 │     │               │ 不足且 rewrite_round < AGENT_MAX_REWRITE
                 │     └── rewrite ◄───┤
                 │                     │ 足够
                 ▼                     ▼
              ┌────────────────────────────────┐
              │           generate             │  只依据 [n] 编号上下文作答
              └───────────────┬────────────────┘
                              ▼
                       ┌──────────────┐
                       │   reflect    │  Self-RAG：检查引用支撑
                       └──────┬───────┘
                              │ 不通过且还有次数
                              └────────► 回到 generate
                              │ 通过
                              ▼
                             END
```

**状态（`AgentState`）是这张图的形状**，读它比读 `graph.py` 更快看清 Agent 能做什么：

```python
# backend/src/knowflow/agent/state.py L56-L107（节选）
class AgentState(TypedDict, total=False):
    """LangGraph 的状态。`total=False` 表示节点可以只返回自己改动的字段。"""
    # ---------------- analyze ----------------
    needs_retrieval: bool
    queries: list[str]  # 本轮要检索的查询（覆盖语义）
    all_queries: Annotated[list[str], operator.add]  # 历史所有查询（累加语义）
    # ---------------- retrieve ----------------
    candidates: Annotated[list[Candidate], merge_candidates]
    retrieval_rounds: int
    gate_passed: bool
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    # ---------------- grade（CRAG） ----------------
    relevant_ids: list[str]
    # ---------------- generate ----------------
    answer: str
    usage: Annotated[dict[str, int], merge_usage]
    # ---------------- reflect（Self-RAG） ----------------
    reflect_passed: bool
    unsupported: list[dict[str, Any]]
```

### 2.2 关键代码 1：reducer 决定"节点返回值是覆盖还是累加"

```python
# backend/src/knowflow/agent/state.py L1-L22（节选）
"""Agent 状态定义（LangGraph 的 State）。

三个关键设计：

1. **reducer 决定"节点返回值是覆盖还是累加"**
   - `queries` 用默认（覆盖）：每轮改写只关心"这一轮用什么查"；
   - `all_queries` 用 `operator.add`（累加）：trace 与调试要看完整的查询历史；
   - `candidates` 用自定义 reducer（按 `vector_id` 去重取高分）：
     多轮检索会重复捞到同一 chunk，累加会让 prompt 里出现重复引用 `[3]` 和 `[7]`
     指向同一段原文 —— 既浪费 token 又让引用编号失去意义。

2. **刻意不做"过滤式"写入**：`grade` 节点**不**返回过滤后的 candidates，
   而是返回 `relevant_ids`，由 `generate` 去筛。
   原因：如果用 reducer 累加，过滤结果会被下一轮的累加又"合并回来"
   （reducer 是 `left + right`，你没法通过返回更少的元素来删东西）。
   这是 LangGraph 最容易踩的坑之一，写在注释里避免后人重犯。

3. **usage 用合并 reducer**：generate 可能被 reflect 打回重试，
   两次调用的 token 都要计入成本，不能覆盖。
"""
```

自定义 reducer 长这样（**`total=False` + 自定义 reducer 是这套设计的全部机关**）：

```python
# backend/src/knowflow/agent/state.py L33-L42
def merge_candidates(
    left: list[Candidate] | None, right: list[Candidate] | None
) -> list[Candidate]:
    """按 `vector_id` 去重合并，保留分数更高的那条，按分数降序返回。"""
    by_id: dict[str, Candidate] = {}
    for candidate in list(left or []) + list(right or []):
        existing = by_id.get(candidate.vector_id)
        if existing is None or candidate.score > existing.score:
            by_id[candidate.vector_id] = candidate
    return sorted(by_id.values(), key=lambda c: (-c.score, c.vector_id))
```

**"保留分数更高的那条"而不是"保留先来的"**：第二轮检索可能因为改写了查询，
对同一 chunk 打出**更低**的分（也可能是更高）。保留高分更符合直觉，
也保证了**同一份输入下结果可复现**。

| 字段 | reducer | 为什么 |
| --- | --- | --- |
| `queries` | 默认（覆盖） | 每轮只需要"这一轮用什么查" |
| `all_queries` | `operator.add` | 调试/前端要展示完整的查询历史 |
| `candidates` | `merge_candidates` | 多轮会重复捞到同一 chunk，必须去重取高分 |
| `tool_calls` | `operator.add` | trace 要把每次工具调用都留下 |
| `grading` | `operator.add` | 每轮的判定结果都要留下（可解释性） |
| `usage` | `merge_usage` | generate 会被 reflect 打回重试，token 必须累加 |
| `errors` | `operator.add` | 错误不能因为后来的成功被抹掉 |

### 2.3 关键代码 2：为什么 `grade` 返回 `relevant_ids` 而不是过滤后的 `candidates`

这是**本阶段最容易踩、也最值得讲的一个坑**。

假设你"顺手"让 `grade` 返回过滤后的候选：

```python
# ❌ 错误写法（本项目刻意不这么做）
return {"candidates": [c for c in candidates if is_relevant(c)]}
```

按 reducer 的定义，`candidates` 的新值是 `merge_candidates(旧候选, 你返回的候选)`——
**也就是"并集"**。你返回得再少，旧候选**依然在 left 里**，过滤等于没做。
更糟的是：**它不报错**。你只会看到"不相关的片段还是进了 prompt"。

正确做法是**把"判定结果"和"候选集合"分开存**，由需要它们的人去筛：

```python
# backend/src/knowflow/agent/nodes.py L781-L793
def select_contexts(state: AgentState) -> list[Candidate]:
    """按 `relevant_ids` 过滤候选（grade 判定为相关的优先），保持分数序。"""
    candidates = list(state.get("candidates") or [])
    if not candidates:
        return []
    relevant_ids = set(state.get("relevant_ids") or [])
    selected = [c for c in candidates if c.vector_id in relevant_ids] if relevant_ids else []
    if not selected:
        # grade 全判不相关时不留空：退回分数最高的前几条，
        # 让 generate 有机会基于它们回答（并在 reflect 阶段被检查引用）。
        selected = candidates[: max(1, min(3, len(candidates)))]
    selected.sort(key=lambda c: (-c.score, c.vector_id))
    return selected
```

**"全判不相关也退回前 3 条"这个兜底很关键**：如果 `grade` 一个都不放行，
`generate` 会因为"没有上下文"直接短路拒答——而 `grade` 本身是**可能判错的**。
退几条给它、再让 `reflect` 检查引用，比直接拒答更稳。

> 🔻 **这条经验可以一般化**：在 LangGraph 里，**reducer 是累加时，
> "少返回一些元素"不能表达"删除"**。要学会区分"我算出来的结果"（放普通字段）
> 与"可以累积的证据"（放 reducer 字段）。本项目 `grading` / `relevant_ids`
> 就是"算出来的结果"，`candidates` 是"累积的证据"。

### 2.4 关键代码 3：六个节点的输入 → 决策 → 输出

```python
# backend/src/knowflow/agent/nodes.py L6-L15（源码里的表，直接抄）
| 节点 | 干什么 | 写回 state |
| --- | --- | --- |
| `analyze` | 判断要不要检索；生成检索用查询 | needs_retrieval / queries / question_type |
| `retrieve` | 模型驱动工具循环（或降级为直接检索） | candidates / gate_* / tool_calls / retrieval_rounds |
| `grade` | CRAG：逐条判断召回是否相关 | relevant_ids / grading |
| `rewrite` | 召回不足时改写查询（有次数上限） | queries / rewrite_round |
| `generate` | 组装上下文生成答案（流式推 token） | answer / refusal / usage |
| `reflect` | Self-RAG：检查答案是否有引用支撑 | reflect_passed / unsupported / reflect_instruction |
```

**节点 ① `analyze`**：解析失败时**默认检索**（方向很重要）：

```python
# backend/src/knowflow/agent/nodes.py L209-L214（节选）
            needs = decision.get("needs_retrieval")
            if not isinstance(needs, bool):
                # 降级：拿不到判断就**默认检索**。理由：多检索一次的成本
                # （几十毫秒 + 一点 token）远小于"该查没查"导致的错误答案。
                needs = True
            queries = _normalize_queries(decision.get("queries"), fallback=question)
```

**"默认检索"而不是"默认不检索"**：两种降级的代价不对称——
多查一次的代价是几十毫秒；该查没查的代价是**一个错误答案**。
**不对称时选危害小的那一边**（和阶段 3 的"宁可少一行数据也不要把数据当列名"是同一种判断）。

`_normalize_queries` 保证**永远不返回空列表**（取不到就用原问题兜底），
并且去重、最多 3 条。**"永远不返回空"是这类函数的通用要求**：
空查询会让这一轮检索静默地什么都没做。

### 2.5 关键代码 4：工具循环与两个硬上限

```python
# backend/src/knowflow/agent/nodes.py L51-L54
# 工具循环的最大轮数：防止模型无限调工具（每轮都是一次 API 调用 = 钱和时间）
MAX_TOOL_ITERATIONS = 3
# 一次工具循环里最多执行多少次 kb_search
MAX_SEARCH_CALLS = 4
```

```python
# backend/src/knowflow/agent/nodes.py L340-L356（节选）
def _run_tool_loop(...) -> list[ToolResult]:
    """模型驱动的工具循环。

    **与固定链的本质区别**：这里模型看到第一批检索结果后，
    可以决定"信息不够，换个词再查一次"或者"够了，不用再查"。
    循环上限 `MAX_TOOL_ITERATIONS` 是硬约束 —— 没有它，一个爱查的模型
    能把一次提问变成 20 次 API 调用。
    """
```

循环结构里有两个**必须理解的设计**：

```python
# backend/src/knowflow/agent/nodes.py L374-L391（节选）
    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        # 第一轮强制先执行 analyze 给出的查询 —— 保证"至少检索一次"，
        # 也避免模型在第一轮就空手回答（那等于绕过了 RAG）。
        if iteration == 1 and queries:
            for query in queries[:MAX_SEARCH_CALLS]:
                result = _invoke_search(...)
                results.append(result)
                search_calls += 1
                messages.append(ChatMessage(role="assistant", content=f"（调用 kb_search：{query}）"))
                messages.append(ChatMessage(role="tool", content=result.to_tool_message()))

        if search_calls >= MAX_SEARCH_CALLS:
            break
```

1. **第一轮"强制先检索一次"**：如果不强制，一个偷懒的模型可能直接空手回答
   ——那就等于**绕过了整个 RAG**，而系统还以为自己检索过了。
2. **两个上限是"或"关系**（轮数 ≤ 3 **且** 检索次数 ≤ 4）：
   模型可以在第 1 轮里要求调 4 次 `kb_search`，那就直接到顶；
   也可以每轮调 1 次、跑满 3 轮。两种路径都由硬约束封住。

工具返回给模型的文本**必须有长度上限**：

```python
# backend/src/knowflow/agent/tools.py L58-L60
    def to_tool_message(self) -> str:
        """回灌给模型的工具结果文本。**必须有长度上限**：
        一次检索 5 条父块可能上万字，全塞回去会让下一轮 prompt 爆掉。"""
```

实现里是 `c.context_text[:1200]`（L66）。**为什么上限放在这里而不是"少返回几条"**：
条数少了模型看不到全貌；字符截断则保证"每条都有、只是略短"。

### 2.6 关键代码 5：不支持工具时的优雅降级

```python
# backend/src/knowflow/agent/nodes.py L267-L292（节选）
            if deps.chat_model.supports_tools():
                used_tool_loop = True
                results = _run_tool_loop(deps, question=..., queries=..., kb_ids=..., ...)
            else:
                # ---------- 优雅降级：模型不支持工具调用 ----------
                # 直接用 analyze 产出的查询逐条检索，把"工具调用"退化成"固定检索"。
                # 能力下降（不能根据中间结果改查询），但功能完整、且行为可预测。
                logger.info("agent.tools_unsupported", model=deps.chat_model.model)
                results = [
                    deps.tools.invoke(
                        TOOL_KB_SEARCH,
                        {"query": query, "top_k": top_k},
                        kb_ids=kb_ids, top_k=top_k, use_rerank=use_rerank,
                    )
                    for query in queries[:MAX_SEARCH_CALLS]
                ]
```

**这段代码是"离线模式也能跑全图"的关键**：`MockChatModel.supports_tools()` 返回 `False`，
于是检索退化成"按 analyze 给的查询逐条查"——**能力下降但功能完整、行为可预测**。
本会话跑图时会看到一行 `agent.tools_unsupported model=offline-mock` 的日志，
它就是这条分支被走到的证据。

**这也是"工具描述（description）就是产品代码"的地方**：

```python
# backend/src/knowflow/agent/tools.py L8-L14（节选）
工具设计的经验（这一条比实现更重要）：
**工具描述（description）决定调用准确率。**
写"搜索知识库"模型会乱调、乱传参；写成
"在企业知识库中做混合检索（语义+关键词）。当问题涉及公司制度、产品文档、
业务数据时使用；参数 query 应使用名词短语而不是整句问句"
——调用准确率会明显提升。所以本文件的 `description` 是**产品代码**，
不是注释，改它要像改 Prompt 一样谨慎。
```

`kb_search` 的 schema 里还有两个刻意的约束：`query` 描述里明确"用名词短语"、
`top_k` 有 `minimum: 1 / maximum: 10`（在协议层就封住了模型的乱传参）。

### 2.7 关键代码 6：`grade` 为什么用 LLM，以及它的三重降级

```python
# backend/src/knowflow/agent/nodes.py L462-L472（节选）
        if not candidates:
            return {"relevant_ids": [], "grading": [], "grading_skipped": False}

        # 候选不多时直接全放行：省一次模型调用。
        # 3 条以内基本都在 top_k 里，逐条判定的收益小于成本。
        if len(candidates) <= 3:
            return {
                "relevant_ids": [c.vector_id for c in candidates],
                "grading": [],
                "grading_skipped": True,
            }
```

**`grade` 有三条路径**（这就是"降级分支是线上分支"的具体形态）：

| 条件 | 策略 | 记在哪 |
| --- | --- | --- |
| 候选 ≤ 3 条 | 全放行（省一次调用） | `grading_skipped=True` |
| `AGENT_GRADING_ENABLED=false` | 启发式（向量分 OR 覆盖率） | `strategy="heuristic(disabled)"` |
| LLM 调用失败 / JSON 解析失败 | 启发式兜底 | `strategy="heuristic(llm_failed)"` |
| 正常 | LLM 逐条判定 | `strategy="llm"` |

启发式兜底只有 5 行，但"最后一根稻草"那条注释很值得读：

```python
# backend/src/knowflow/agent/nodes.py L564-L576（节选）
def _heuristic_relevance(candidates: Sequence[Candidate], cfg: Settings) -> list[str]:
    """零成本相关性判定：向量分或关键词覆盖率任一达标就算相关。"""
    relevant: list[str] = []
    for candidate in candidates:
        vector_ok = (candidate.vector_score or 0.0) >= cfg.vector_min_score
        keyword_ok = candidate.keyword_coverage >= cfg.keyword_min_coverage
        if vector_ok or keyword_ok:
            relevant.append(candidate.vector_id)
    # 都没达标但确实召回了内容时，至少留最高分那条 ——
    # 全丢会让"检索到了一点东西"变成"完全拒答"，通常更糟。
    if not relevant and candidates:
        relevant = [candidates[0].vector_id]
    return relevant
```

**"为什么 grade 用 LLM 而不是复用检索分数"**（`docs/00` 第 1.2 节的四个设计点之一）：

> 检索分数是「和 query 的字面/语义相似度」，不等于「能不能回答这个问题」。
> 实测里经常出现「高分片段答的是另一个问题」。LLM 判定多花一次便宜的小模型调用，
> 但能拦掉一整类错误。（代价：多一次调用、多 300~800ms，所以做成可开关 `AGENT_GRADING_ENABLED`。）

### 2.8 关键代码 7：`rewrite` 为什么有上限、`reflect` 为什么不重检索

**`rewrite` 有次数上限**（`AGENT_MAX_REWRITE=2`），而且第二次会提示模型"换策略"：

```python
# backend/src/knowflow/llm/prompts.py L204-L209
    hint = ""
    if attempt >= 2:
        hint = (
            "\n\n注意：这是第 2 次改写，上一次改写后仍然没有检索到有用资料。"
            "请换一种策略——优先拆分成更小的子问题，或改用更宽泛的上位词。"
        )
```

> **为什么 `rewrite` 要有次数上限？** 避免"死循环烧钱"。上限 2 次：第一次改写查询，
> 第二次拆子问题；还没料就直接走 `generate` + **明确告知"知识库中没有足够信息"**
> ——宁可拒答也不编。（`docs/00` 第 1.2 节）

`rewrite` 解析失败时也**不抛异常**，直接用原问题兜底：

```python
# backend/src/knowflow/agent/nodes.py L599-L609（节选）
            queries = [question]
            reason = ""
            try:
                result = deps.chat_model.complete(messages, temperature=0.0)
                obj = extract_json_object(result.text) or {}
                queries = _normalize_queries(obj.get("queries"), fallback=question)
```

**`reflect` 不重新检索**（职责分离）：

> **为什么 `reflect` 不重新检索？** 职责分离：`grade` 管「料找对没」，
> `reflect` 管「话有没有依据」。混在一起会出现「越反思越跑偏」。（`docs/00` 第 1.2 节）

`reflect` 解析失败时**默认通过**：

```python
# backend/src/knowflow/agent/nodes.py L821-L826（节选）
                obj = extract_json_object(result.text)
                if obj is None:
                    # 解析失败时**默认通过**：宁可漏掉一次纠正，
                    # 也不要因为判官输出格式问题让用户多等一轮生成。
                    passed = True
                    unsupported = []
```

**注意这两个"默认值"的方向是相反的**：`analyze` 失败**默认检索**（多做无害），
`reflect` 失败**默认通过**（少做无害）。判断依据是同一条：
**降级到"代价更小的一边"**。`reflect` 误判"通过"只是漏掉一次优化；
误判"不通过"会让用户多等一整轮生成（而且可能永远循环）。

重试时要把"错在哪"明确告诉模型，否则重试等于重新掷骰子：

```python
# backend/src/knowflow/agent/nodes.py L846-L853（节选）
        if not passed and unsupported:
            # 把"错在哪"明确告诉模型，否则重试等于重新掷骰子。
            sentences = "；".join(str(item.get("sentence", ""))[:80] for item in unsupported[:3])
            instruction = (
                f"上一轮回答中以下句子缺少参考资料支撑：{sentences}。"
                "请重新作答：每一句事实性陈述都必须以 [n] 标注来源；"
                "如果某条信息在参考资料中确实不存在，请直接删除该句，不要改写或猜测。"
            )
```

### 2.9 关键代码 8：四条条件边是**纯函数**，所以能单测

```python
# backend/src/knowflow/agent/graph.py L16-L18
**每个条件边的判据都写成小函数**（`route_after_*`），而不是内联 lambda：
条件边是"业务规则"，需要被单独测试（"give 一个 gate 没过且还有改写次数的 state，
应该走 rewrite"）。内联 lambda 没法测。
```

四条判据（**每一条的每个分支都值得背**）：

```python
# backend/src/knowflow/agent/graph.py L64-L117（节选，四条边）
def route_after_analyze(state: AgentState, cfg: Settings) -> str:
    """不需要检索就直接生成（闲聊 / 询问系统能力）。"""
    if state.get("finished"):
        return END
    return NODE_RETRIEVE if state.get("needs_retrieval", True) else NODE_GENERATE


def route_after_grade(state: AgentState, cfg: Settings) -> str:
    """召回不足且还有改写次数 → 改写查询再查一次；否则去生成（可能是拒答）。"""
    if state.get("finished"):
        return END
    if not state.get("candidates"):
        return NODE_GENERATE
    round_no = int(state.get("rewrite_round") or 0)
    if round_no >= cfg.agent_max_rewrite:
        return NODE_GENERATE
    if not state.get("gate_passed", False):
        return NODE_REWRITE
    if not state.get("relevant_ids"):
        return NODE_REWRITE
    return NODE_GENERATE


def route_after_generate(state: AgentState, cfg: Settings) -> str:
    """只有"确实检索过 + 有上下文 + 没拒答 + 还有反思次数"才去反思。

    注意几个跳过条件都是有理由的：
    - 直答（没检索）：没有引用可校验；
    - 拒答：本来就没编东西，再校验一次纯浪费；
    - 没有候选：reflect 拿不到参考资料，判定必然不可靠。
    """
    if state.get("finished") and not state.get("answer"):
        return END
    if not state.get("needs_retrieval", True):
        return END
    if state.get("refusal"):
        return END
    if cfg.agent_max_reflect <= 0:
        return END
    if not state.get("candidates"):
        return END
    if int(state.get("reflect_round") or 0) >= cfg.agent_max_reflect:
        return END
    return NODE_REFLECT


def route_after_reflect(state: AgentState, cfg: Settings) -> str:
    """反思不通过就带着"哪里缺引用"再生成一次；次数用完就接受现状。"""
    if state.get("reflect_passed", True):
        return END
    if int(state.get("reflect_round") or 0) > cfg.agent_max_reflect:
        return END
    return NODE_GENERATE
```

**"拒答不反思"这一条特别值得讲**：拒答的答案是**固定话术**，
它本来就"没有编东西"，再花一次模型调用去校验是纯浪费。
这就是 `route_after_generate` 里 `if state.get("refusal"): return END` 的意义——
**它同时是省 token 和"拒答路径不调生成模型"的第二个保证**。

装配代码里只有一层 lambda（把 `cfg` 注入判据，判据本身仍然是纯函数）：

```python
# backend/src/knowflow/agent/graph.py L163-L175（节选）
    # 条件边用 lambda 把 cfg 注入判据函数（判据本身是独立可测的纯函数）
    builder.add_conditional_edges(
        NODE_ANALYZE,
        lambda state: route_after_analyze(state, cfg),
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_GENERATE: NODE_GENERATE, END: END},
    )
    builder.add_edge(NODE_RETRIEVE, NODE_GRADE)
    builder.add_conditional_edges(
        NODE_GRADE,
        lambda state: route_after_grade(state, cfg),
        {NODE_REWRITE: NODE_REWRITE, NODE_GENERATE: NODE_GENERATE, END: END},
    )
```

节点名用常量而不是字符串字面量（`NODE_ANALYZE = "analyze"`），
注释里给了理由：**"用常量避免拼写错误导致 LangGraph 运行期才报错"**。
条件边的目标集合也必须列全，否则 LangGraph 会在**运行期**才报错。

### 2.10 关键代码 9：checkpointer 为什么用 `trace_id` 当 `thread_id`

```python
# backend/src/knowflow/agent/graph.py L20-L31（节选）
**关于 checkpointer（重要决定）**：
- `thread_id` 用 **trace_id**（一次请求一个 thread），不是 conversation_id。

为什么不是 conversation_id？因为 `AgentState` 里的 `candidates` / `all_queries`
用的是**累加 reducer**。如果同一会话共用一个 thread，第二轮的状态会与第一轮
合并 —— 上一轮的候选片段会混进这一轮的上下文，引用编号也全乱。
而"多轮记忆"这件事本项目已经有更可控的实现：**从 MySQL 读消息 + 双重截断**
（见 `agent/memory.py`），那才是产品语义上的记忆。

那 checkpointer 还留着做什么？**可回溯**：每个 trace 的图状态被持久化，
排障时能看到"那次请求在图里走到了哪一步、每个节点的中间产出是什么"。
```

**这段话读十遍都不为过**：它把"记忆"这件事**拆成了两个不同的东西**：

| 需求 | 实现 | 语义 |
| --- | --- | --- |
| **多轮对话记忆**（"那二线城市呢？"） | MySQL 的 `messages` + 双重截断 | **产品语义**上的记忆 |
| **图状态可回溯**（排障） | checkpointer（`thread_id=trace_id`） | **工程语义**上的追溯 |

用 checkpointer 做多轮记忆看起来"更省事"，但它会把**上一轮的候选片段**带进这一轮，
污染引用编号——**这是一个功能看起来正常、结果却是错的 bug**。

checkpointer 本身也带降级（**可观测性坏了不该让问答不可用**）：

```python
# backend/src/knowflow/agent/graph.py L119-L146（节选）
def build_checkpointer(cfg: Settings) -> Any | None:
    """按配置构造 checkpointer。

    `sqlite` 用文件持久化（服务重启后仍能回溯历史 trace 的图状态）；
    `memory` 只在进程内。任何构造失败都返回 None（**不阻断服务启动**：
    可观测性坏了不该让问答不可用）。
    """
    ...
            # check_same_thread=False：LangGraph 可能在别的线程里读写 checkpoint
            conn = sqlite3.connect(str(cfg.checkpoint_path), check_same_thread=False)
```

🔻 **一个真实的版本提示**（本会话实测）：在 `langgraph 1.2.12` 下把
`Candidate` 这种自定义对象写进 checkpoint 时，会打印一条警告：

```
[warning] Deserializing unregistered type knowflow.retrieval.types.Candidate from checkpoint.
This will be blocked in a future version. Set LANGGRAPH_STRICT_MSGPACK=true to block now,
or add to allowed_msgpack_modules to allow explicitly: [('knowflow.retrieval.types', 'Candidate')]
```

**当前不影响功能**（反序列化照常工作），但**未来的 LangGraph 版本会默认拦下**。
要提前适配就设置 `LANGGRAPH_STRICT_MSGPACK=true` 并显式登记
`allowed_msgpack_modules`——**这条警告值得写进 roadmap**。

### 2.11 关键代码 10：会话记忆的"双重截断"与数据库级并发安全

```python
# backend/src/knowflow/agent/memory.py L1-L19（节选）
"""会话记忆：双重截断 + 数据库级并发安全。

**为什么必须截断**（两个都要，取更严者）：

| 只按轮数截断 | 只按字符截断 |
| --- | --- |
| 用户粘贴一篇 3000 字的需求后，10 轮就是 3 万字，照样爆上下文 | 如果每轮都很短（"嗯"、"继续"），会保留上百轮，语义上早已跑题 |

所以实现是「**从最新往回装，装不下就停**」：
既限制轮数（`MEMORY_MAX_TURNS`），也限制总字符（`MEMORY_MAX_CHARS`），
任何一个到顶就停止回溯。这样成本上限是可算的：
`prompt_token ≈ 记忆字符 + 上下文预算`，不会随对话轮数线性增长。

**为什么用 `with_for_update()` 而不是 Python 的 `threading.Lock`**：
Python 锁只在单进程内有效。生产上多 worker / 多副本时，
两个请求同时写同一会话会丢掉一次 `message_count` 自增，
表现为"会话列表显示的消息数比实际少" —— 难查且不影响功能，属于最烦的一类 bug。
`SELECT ... FOR UPDATE` 把并发控制交给数据库，跨进程也正确。
"""
```

实现只有 26 行，**"至少保留最后一轮"这个条件很容易写漏**：

```python
# backend/src/knowflow/agent/memory.py L52-L65（节选）
    for message in reversed(messages):
        if message.role not in MEMORY_ROLES:
            continue
        cost = len(message.content)
        # 至少保留最后一轮：否则会出现"有记忆配置但完全没记忆"的情况，
        # 多轮问题（"那二线城市呢？"）直接失效。
        if kept and (turns >= max_turns or used_chars + cost > max_chars):
            break
        if message.role == "user":
            turns += 1
        kept.append(message)
        used_chars += cost

    kept.reverse()
    return kept
```

三个细节：
1. **`if kept and (...)`** 而不是 `if ...`——`kept` 非空才允许 `break`，
   保证**至少留下最后一条**；不写这个条件，`max_chars=1` 时会返回空记忆；
2. **`turns` 只在 `user` 消息上自增**——"一轮"= 一问一答，用 assistant 数会翻倍；
3. **`kept.reverse()`**——从后往前装、返回前必须转回**时间正序**，否则 prompt 里
   对话顺序是反的（模型会看到"答案在问题前面"）。

数据库读取时**故意多取一点再截断**：

```python
# backend/src/knowflow/agent/memory.py L88-L94（节选）
    """从 MySQL 读取并截断会话历史。

    **为什么要多取一些再截断**（`limit = max_turns * 3`）：
    截断是按"字符预算从后往前装"，如果 SQL 只取最后 `max_turns` 条，
    当最后几条恰好都很长时，能装下的轮数会比配置少很多。
    多取一点（有上限，不会失控）让截断逻辑自己决定实际保留多少。
    """
```

写消息时用行锁把"写消息 + 会话计数自增"变成**原子操作**：

```python
# backend/src/knowflow/agent/memory.py L127-L134（节选）
    conversation = session.get(Conversation, conversation_id, with_for_update=True)
    if conversation is None:
        raise ValueError(f"会话 {conversation_id} 不存在")

    message = Message(conversation_id=conversation_id, role=role, content=content, **fields)
    session.add(message)
    conversation.message_count = (conversation.message_count or 0) + 1
```

`MEMORY_ROLES = ("user", "assistant")` 排除了 tool / system：
**tool 消息是中间产物**（已经体现在最终答案里），**system 是每轮重新拼的**。

### 2.12 关键代码 11：`RecorderBox`——contextvars 跨线程的真实 bug

这是**本项目最值得讲的一个 bug**（E6.5），完整记录在源码注释里：

```python
# backend/src/knowflow/agent/nodes.py L60-L80（节选）
class RecorderBox:
    """一次请求的 `TraceRecorder` 持有者（显式传递，不用 contextvars）。

    **为什么不用 contextvars**（这是踩出来的，值得写清楚）：

    我们最初用模块级 `ContextVar` 绑定 recorder，非流式路径完全正常。
    但**流式路径必然报错**：

        ValueError: <Token var=<ContextVar name='current_recorder'>> was created
                    in a different Context

    原因是 Starlette 迭代**同步生成器**时会用 `iterate_in_threadpool`，
    **每取一个元素就是一次 `run_in_threadpool(next, ...)`** ——
    也就是每个 `next()` 可能落在不同的线程、拿着**各自的 context 副本**上。
    而 `ContextVar.set()` 返回的 token 只能在**同一个 context** 里 `reset()`。
    生成器第 1 次 yield 时 set，第 N 次 yield 后退出 `with` 时 reset → 直接抛异常。

    流式恰好是本项目的主路径，所以这个问题必须从根上解决：
    **把 recorder 挂在依赖对象上显式传递**，不依赖任何隐式上下文。
    """
```

**为什么这个 bug 特别值得记**（E6.5 的原话）：

> 它只在**流式**路径出现，而流式恰好是这个项目的主路径；
> `pytest` 用 `ASGITransport` 打请求时**不会**触发这种分次线程调度，
> 所以 **204 个单测全绿也照样漏掉它** —— 只有真起 uvicorn 打一次流式接口才会暴露。

**教训**："单元测试全绿"证明不了"流式路径是对的"。
`contextvars` 在"同步生成器 + 线程池"的组合下**语义是坏的**，
要用**显式传参**（依赖注入）替代隐式上下文。

配套的两个工具函数也各有一个要点：

```python
# backend/src/knowflow/agent/nodes.py L136-L143（节选）
    recorder = deps.recorder_box.value or _get_recorder()
    if recorder is None:
        with nullcontext():
            yield
        return
    with recorder.span(name, span_type=span_type, input=payload):
        yield
```

**"取不到 recorder 就用 `nullcontext()`"**：让 `agent/` 能脱离可观测模块运行
（脚本、单测）。**可观测是横切关注点，不该成为业务代码的硬依赖。**

```python
# backend/src/knowflow/agent/nodes.py L155-L163（节选）
def emit(payload: dict[str, Any]) -> None:
    """推一个 SSE 事件。**任何异常都不能影响主流程**（前端掉线不该让问答失败）。"""
```

**"前端掉线不该让问答失败"**：SSE 是**单向**推送，客户端断开时 `writer()` 会抛异常；
如果不吞掉，用户刷新一下页面就会让整个回答生成中断，而 `messages` 表里
已经落了一条 user message——**留下一条永远不会有回答的提问**。

### 2.13 关键代码 12：`runner` 的收尾与"空答案兜底"

```python
# backend/src/knowflow/agent/runner.py L1-L14（节选）
"""Agent 运行器：把 LangGraph 的图包装成"一次问答"。

**为什么需要这一层**：图的输入输出是"状态字典"，而 API 层需要的是
"答案 + 引用 + 用量 + 反思结果"。这一层做映射与收尾，好处是：

- 图的形状可以被替换（`agent` 全图 / `rag` 快路径），而 API 代码零改动；
- 非流式（`POST /chat`）与流式（`POST /chat/stream`）**共用同一套逻辑**，
  避免"流式和非流式行为不一致"这个经典 bug（用户会发现两种接口答得不一样）。

流式实现的关键取舍：用 `stream_mode=["custom", "updates"]` **同时**收两类事件 ——
`custom` 是节点主动推的（token / sources / tool / reflect），
`updates` 是 LangGraph 报的状态增量。用后者自己累积出最终状态，
这样**即使没有 checkpointer 也能拿到完整结果**（不依赖额外存储）。
"""
```

**"流式与非流式共用同一套逻辑"** 是个容易做错的工程点：
如果两条路各写一遍，一定会漂移（最常见的是**流式没有引用、非流式有**）。

收尾处有三件事必须做对：

```python
# backend/src/knowflow/agent/runner.py L259-L278（节选）
        except Exception as exc:  # noqa: BLE001 - 流一旦开始就不能再抛给 HTTP 层了
            logger.error("agent.stream_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            yield {"event": "error", "code": "AGENT_ERROR", "message": f"生成过程中出错：{type(exc).__name__}"}
            outcome = self._to_outcome(accumulated)
            yield {"event": "done", **outcome.done_payload()}
            yield {"event": "end"}
            return

        outcome = self._to_outcome(accumulated)
        if not outcome.answer and not outcome.refusal:
            # 图跑完却没有答案：说明某个节点静默失败了。**不能返回空字符串**，
            # 那会让前端显示一个空气泡，用户以为界面卡了。
            outcome.answer = "抱歉，本次生成没有产出内容。请重试或换一种问法。"
            outcome.refusal = False
        yield {"event": "done", **outcome.done_payload()}
        yield {"event": "end"}
```

三件事：
1. **流已经开始后，异常不能再抛给 HTTP 层**——HTTP 头早就发出去了，
   只能发 `error` 帧。而且**发完 `error` 仍要补 `done` 与 `end`**，
   否则前端的 SSE 解析器会以为连接被截断；
2. **空答案不许返回空字符串**：换成一句明确的话术。
   前端显示一个"空气泡"是最糟的失败形态——用户以为界面卡了，而不是"这次没生成出来"；
3. **`sources` 复用了 `select_contexts`**（`runner.py` L159-L161）：
   **"展示的引用"和"喂给模型的上下文"必须是同一份**，否则用户点开引用会发现
   内容对不上答案。

---

## 3. 怎么验证我做对了

### 验证 1：四条条件边的判据（本会话真实执行）

条件边是**纯函数**，所以可以零成本、零依赖地逐个分支验证：

```powershell
cd D:\Projects\pythonProjects\knowflow\backend
$env:PYTHONPATH='D:\Projects\pythonProjects\knowflow\backend\src'
$env:EMBEDDING_PROVIDER='hash'; $env:VECTOR_BACKEND='memory'
$py='D:\Projects\pythonProjects\knowflow\.venv\Scripts\python.exe'
@'
from knowflow.core.config import get_settings
from knowflow.agent.graph import (route_after_analyze, route_after_grade,
                                  route_after_generate, route_after_reflect)
cfg = get_settings()
print("cfg: agent_max_rewrite =", cfg.agent_max_rewrite, " agent_max_reflect =", cfg.agent_max_reflect,
      " grading_enabled =", cfg.agent_grading_enabled, " fusion =", cfg.fusion)
print()
print("analyze  不需要检索 ->", route_after_analyze({"needs_retrieval": False}, cfg))
print("analyze  需要检索   ->", route_after_analyze({"needs_retrieval": True}, cfg))
print("grade    无候选     ->", route_after_grade({"candidates": []}, cfg))
print("grade    闸门没过   ->", route_after_grade({"candidates": [1], "gate_passed": False, "rewrite_round": 0}, cfg))
print("grade    改写用尽   ->", route_after_grade({"candidates": [1], "gate_passed": False, "rewrite_round": 2}, cfg))
print("grade    相关为空   ->", route_after_grade({"candidates": [1], "gate_passed": True, "relevant_ids": [], "rewrite_round": 0}, cfg))
print("grade    都满足     ->", route_after_grade({"candidates": [1], "gate_passed": True, "relevant_ids": ["1:0"], "rewrite_round": 0}, cfg))
print("generate 拒答       ->", route_after_generate({"refusal": True, "needs_retrieval": True, "candidates": [1]}, cfg))
print("generate 直答       ->", route_after_generate({"needs_retrieval": False, "candidates": [], "answer": "hi"}, cfg))
print("generate 正常       ->", route_after_generate({"needs_retrieval": True, "candidates": [1], "answer": "a", "reflect_round": 0}, cfg))
print("reflect  通过       ->", route_after_reflect({"reflect_passed": True}, cfg))
print("reflect  不通过     ->", route_after_reflect({"reflect_passed": False, "reflect_round": 1}, cfg))
'@ | & $py -
```

**本会话真实输出**：

```
cfg: agent_max_rewrite = 2  agent_max_reflect = 1  grading_enabled = True  fusion = rrf

analyze  不需要检索 -> generate
analyze  需要检索   -> retrieve
grade    无候选     -> generate
grade    闸门没过   -> rewrite
grade    改写用尽   -> generate
grade    相关为空   -> rewrite
grade    都满足     -> generate
generate 拒答       -> __end__
generate 直答       -> __end__
generate 正常       -> reflect
reflect  通过       -> __end__
reflect  不通过     -> generate
```

**判断标准**（逐条都有"为什么"）：
- ✅ `grade 无候选 → generate`（不是 rewrite）：**没捞到东西时改写查询通常也捞不到**，
  不如直接进 generate 走拒答，省一次模型调用；
- ✅ `grade 改写用尽 → generate`：`rewrite_round=2` 已达 `AGENT_MAX_REWRITE=2`，
  **必须有这个出口，否则就是死循环**；
- ✅ `grade 闸门没过 → rewrite`、`grade 相关为空 → rewrite`：两种"召回不足"都改写；
- ✅ `generate 拒答 → __end__`（`END` 的字符串值是 `__end__`）：
  **拒答不反思**，省一次调用（§2.9）；
- ✅ `reflect 不通过 → generate`：带着 `reflect_instruction` 重新生成一次。

### 验证 2：`candidates` 的 reducer 去重取高分（本会话真实执行）

```powershell
@'
from knowflow.agent.state import merge_candidates
from knowflow.retrieval.types import Candidate
def C(vid, score):
    c = Candidate(vector_id=vid, doc_id=1, doc_name="员工报销制度.md", content="x", context_text="x")
    c.score = score
    return c
m = merge_candidates([C("1:0", 0.5), C("1:1", 0.4)], [C("1:0", 0.9), C("1:2", 0.3)])
print("合并结果 =", [(c.vector_id, c.score) for c in m])
'@ | & $py -
```

**本会话真实输出**：

```
合并结果 = [('1:0', 0.9), ('1:1', 0.4), ('1:2', 0.3)]
```

**判断标准**：
- ✅ `1:0` **只出现一次**，且用的是**更高**的分数 `0.9`（不是先来的 `0.5`）；
- ✅ 三路独有的候选**一个都没丢**（`1:1`、`1:2` 都在）；
- ✅ 结果按分数降序。
- ⚠ 如果这里出现两个 `1:0`，你的 prompt 里就会有 `[3]` 和 `[7]` 指向同一段原文
  ——**引用编号失去意义，而且不报错**。

### 验证 3：会话记忆的双重截断（本会话真实执行）

```powershell
@'
from knowflow.agent.memory import truncate_history, MEMORY_ROLES
from knowflow.llm.base import ChatMessage
msgs = []
for i in range(1, 6):
    msgs.append(ChatMessage(role="user", content=f"问题{i}"))
    msgs.append(ChatMessage(role="assistant", content="答" * 3000))
print("max_turns=6 max_chars=4000 ->", len(truncate_history(msgs, max_turns=6, max_chars=4000)), "条")
print("max_turns=2 max_chars=100000 ->", [m.content[:6] for m in truncate_history(msgs, max_turns=2, max_chars=100000)])
print("max_chars=1（至少保留最后一轮） ->", len(truncate_history(msgs, max_turns=6, max_chars=1)), "条")
print("tool/system 不进记忆:", MEMORY_ROLES)
'@ | & $py -
```

**本会话真实输出**：

```
max_turns=6 max_chars=4000 -> 2 条
max_turns=2 max_chars=100000 -> ['问题4', '答答答答答答', '问题5', '答答答答答答']
max_chars=1（至少保留最后一轮） -> 1 条
tool/system 不进记忆: ('user', 'assistant')
```

**判断标准**：
- ✅ 第一行：5 轮 × 3000 字的 assistant，`max_chars=4000` 只装得下 **1 轮**
  （`2 条` = 1 问 + 1 答）——**字符预算是更严的那个约束**；
- ✅ 第二行：放宽字符后，`max_turns=2` 精确留下**最后两轮**（问题 4/5）+ 顺序**正序**；
- ✅ 第三行：`max_chars=1` 时仍然保留 **1 条**（至少保留最后一轮，
  否则"那二线城市呢？"这类指代问题直接失效）；
- ✅ `MEMORY_ROLES` 只有 `user`/`assistant`。

### 验证 4：整图离线端到端（本会话真实执行，不碰 MySQL、不下载模型）

```powershell
cd D:\Projects\pythonProjects\knowflow\backend
$env:PYTHONPATH='D:\Projects\pythonProjects\knowflow\backend\src'
$env:EMBEDDING_PROVIDER='hash'; $env:VECTOR_BACKEND='memory'; $env:AGENT_CHECKPOINT_BACKEND='memory'
$py='D:\Projects\pythonProjects\knowflow\.venv\Scripts\python.exe'
@'
from knowflow.core.config import get_settings
from knowflow.embeddings.hash_embedder import HashEmbedder
from knowflow.vectorstore.memory_store import InMemoryVectorStore
from knowflow.vectorstore.base import VectorItem
from knowflow.retrieval.bm25 import BM25Registry
from knowflow.retrieval.engine import HybridRetriever
from knowflow.agent.tools import ToolRegistry
from knowflow.agent.nodes import AgentDeps
from knowflow.agent.graph import build_agent_graph, build_fixed_chain
from knowflow.agent.runner import AgentRunner
from knowflow.llm.mock import MockChatModel

cfg = get_settings()
embedder = HashEmbedder(cfg); store = InMemoryVectorStore(cfg); bm25 = BM25Registry()
chunks = [
    {"vector_id": "1:0", "chunk_id": 1, "doc_id": 1, "doc_name": "员工报销制度.md", "chunk_index": 0,
     "content": "一线城市住宿标准为每晚600元。", "parent_content": "员工报销制度 > 差旅报销标准：一线城市住宿标准为每晚600元。",
     "page_no": None, "section_path": "员工报销制度 > 差旅报销标准"},
    {"vector_id": "1:1", "chunk_id": 2, "doc_id": 1, "doc_name": "员工报销制度.md", "chunk_index": 1,
     "content": "二线城市住宿标准为每晚400元。", "parent_content": "员工报销制度 > 差旅报销标准：二线城市住宿标准为每晚400元。",
     "page_no": None, "section_path": "员工报销制度 > 差旅报销标准"},
]
store.upsert(kb_id=1, items=[VectorItem(id=c["vector_id"], vector=embedder.embed_documents([c["content"]])[0],
    content=c["content"], metadata={"doc_id": c["doc_id"], "doc_name": c["doc_name"], "chunk_index": c["chunk_index"],
    "chunk_id": c["chunk_id"], "page_no": c["page_no"], "section_path": c["section_path"]}) for c in chunks])
bm25.add_chunks(1, chunks)
model = MockChatModel(cfg)
retriever = HybridRetriever(embedder=embedder, vector_store=store, bm25=bm25, chat_model=model, settings=cfg)
deps = AgentDeps(settings=cfg, chat_model=model, judge_model=model, retriever=retriever,
                 tools=ToolRegistry(retriever=retriever))
runner = AgentRunner(agent_graph=build_agent_graph(deps), fixed_graph=build_fixed_chain(deps),
                     settings=cfg, model_name=model.model)
for q in ["一线城市住宿标准是多少", "今天天气怎么样"]:
    out = runner.run(question=q, kb_id=1, mode="agent")
    print(f"Q: {q}")
    print(f"   refusal={out.refusal} retrieval_rounds={out.retrieval_rounds} reflect_passed={out.reflect_passed} "
          f"needs_retrieval={out.needs_retrieval} question_type={out.question_type}")
    print(f"   usage={out.usage}  sources={len(out.sources)}  tool_calls={len(out.tool_calls)}  vector_error={out.vector_error}")
    print(f"   answer[:70]={out.answer[:70]!r}")
'@ | & $py -
```

**本会话真实输出**：

```
Q: 一线城市住宿标准是多少
   refusal=False retrieval_rounds=1 reflect_passed=False needs_retrieval=True question_type=factual
   usage={'prompt_tokens': 1948, 'completion_tokens': 604, 'llm_calls': 4}  sources=1  tool_calls=1  vector_error=None
   answer[:70]='（离线模式：未配置大模型 API Key，以下答案由检索结果直接抽取生成，仅用于验证链路是否连通，不代表真实生成质量。）\n\n一线城市住宿标准'
Q: 今天天气怎么样
   refusal=True retrieval_rounds=1 reflect_passed=True needs_retrieval=True question_type=factual
   usage={'prompt_tokens': 332, 'completion_tokens': 110, 'llm_calls': 1}  sources=0  tool_calls=1  vector_error=None
   answer[:70]='知识库中没有找到与该问题相关的内容，因此我无法给出有依据的回答。建议换一种问法，或确认相关文档已经上传。'
```

**判断标准**（这条验证一次覆盖了本阶段四个设计点）：
- ✅ 有答案那一问：`llm_calls = 4` = analyze + grade + generate + reflect
  → **`usage` 的累加 reducer 生效**（覆盖的话只会看到 1 次调用的用量）；
- ✅ 无关那一问：**`llm_calls = 1`**（只有 analyze），`sources = 0`，
  答案是**固定话术** → 这就是 E6 第 [7] 项断言的复现：
  `该 trace 的 span: ['analyze', 'retrieve']`、
  `拒答路径下没有调用生成模型（无 generate.llm span）→ generate.llm span = 0`；
- ✅ `tool_calls = 1` 且日志里有 `agent.tools_unsupported model=offline-mock`
  → **§2.6 的优雅降级分支被走到了**（离线 Mock 不支持 function calling）；
- ⚠ 这一问 `reflect_passed=False` 是正常的（Mock 的答案里有句子没带引用）；
  **`reflect` 只会在还有次数时把流程打回 `generate` 一次**，
  所以 `retrieval_rounds` 仍然是 1、答案照样返回。

### 验证 5：`thread_id` 用 `trace_id` 而不是 `conversation_id`（本会话真实执行）

**这是本阶段最有说服力的一次实测**：用同一个 `thread_id` 连跑两轮，
亲眼看"上一轮的状态混进这一轮"。

```powershell
cd D:\Projects\pythonProjects\knowflow\backend
$env:PYTHONPATH='D:\Projects\pythonProjects\knowflow\backend\src'
$py='D:\Projects\pythonProjects\knowflow\.venv\Scripts\python.exe'
@'
import tempfile, pathlib
from knowflow.core.config import Settings
from knowflow.embeddings.hash_embedder import HashEmbedder
from knowflow.vectorstore.memory_store import InMemoryVectorStore
from knowflow.vectorstore.base import VectorItem
from knowflow.retrieval.bm25 import BM25Registry
from knowflow.retrieval.engine import HybridRetriever
from knowflow.agent.tools import ToolRegistry
from knowflow.agent.nodes import AgentDeps
from knowflow.agent.graph import build_agent_graph, build_checkpointer
from knowflow.agent.runner import AgentRunner
from knowflow.llm.mock import MockChatModel

cfg = Settings(database_url="sqlite://", data_dir=pathlib.Path(tempfile.mkdtemp()),
               agent_checkpoint_backend="sqlite", embedding_provider="hash", vector_backend="memory")
embedder = HashEmbedder(cfg); store = InMemoryVectorStore(cfg); bm25 = BM25Registry()
chunks = [
  {"vector_id": "1:0", "chunk_id": 1, "doc_id": 1, "doc_name": "员工报销制度.md", "chunk_index": 0,
   "content": "一线城市住宿标准为每晚600元。", "parent_content": "一线城市住宿标准为每晚600元。"},
  {"vector_id": "1:1", "chunk_id": 2, "doc_id": 1, "doc_name": "考勤与休假管理办法.md", "chunk_index": 0,
   "content": "年假天数按司龄计算。", "parent_content": "年假天数按司龄计算。"},
]
store.upsert(kb_id=1, items=[VectorItem(id=c["vector_id"], vector=embedder.embed_documents([c["content"]])[0],
   content=c["content"], metadata={"doc_id": c["doc_id"], "doc_name": c["doc_name"], "chunk_index": c["chunk_index"]}) for c in chunks])
bm25.add_chunks(1, chunks)
model = MockChatModel(cfg)
retriever = HybridRetriever(embedder=embedder, vector_store=store, bm25=bm25, chat_model=model, settings=cfg)
deps = AgentDeps(settings=cfg, chat_model=model, judge_model=model, retriever=retriever,
                 tools=ToolRegistry(retriever=retriever))
graph = build_agent_graph(deps, checkpointer=build_checkpointer(cfg))
runner = AgentRunner(agent_graph=graph, settings=cfg, model_name=model.model)

o1 = runner.run(question="一线城市住宿标准是多少", kb_id=1, mode="agent", thread_id="conversation-1")
o2 = runner.run(question="年假有几天", kb_id=1, mode="agent", thread_id="conversation-1")
print("【同一 thread_id 跑两轮（模拟 thread_id=conversation_id）】")
print("  第 1 轮 all_queries =", o1.rewritten_queries)
print("  第 2 轮 all_queries =", o2.rewritten_queries, "  <- 上一轮的查询还在")
print("  第 2 轮 sources =", [(s["rank"], s["doc_name"]) for s in o2.sources])

o3 = runner.run(question="一线城市住宿标准是多少", kb_id=1, mode="agent", thread_id="trace-aaa")
o4 = runner.run(question="年假有几天", kb_id=1, mode="agent", thread_id="trace-bbb")
print("【每请求一个 thread_id（本项目用法）】")
print("  请求 A all_queries =", o3.rewritten_queries)
print("  请求 B all_queries =", o4.rewritten_queries, "  <- 只有本次的查询")
print("  请求 B sources =", [(s["rank"], s["doc_name"]) for s in o4.sources])
'@ | & $py -
```

**本会话真实输出**：

```
【同一 thread_id 跑两轮（模拟 thread_id=conversation_id）】
  第 1 轮 all_queries = ['一线城市住宿标准是多少']
  第 2 轮 all_queries = ['一线城市住宿标准是多少', '年假有几天']   <- 上一轮的查询还在
  第 2 轮 sources = [(1, '员工报销制度.md')]   <- 上一轮的候选也混进来了
【每请求一个 thread_id（本项目用法）】
  请求 A all_queries = ['一线城市住宿标准是多少']
  请求 B all_queries = ['年假有几天']   <- 只有本次的查询
  请求 B sources = [(1, '考勤与休假管理办法.md')]
```

**判断标准**（这就是 §2.10 那段设计说明的实证）：
- ✅ 同一 `thread_id` 时，第 2 轮的 `all_queries` **累加**了上一轮的查询
  （`operator.add` reducer + 共享 checkpoint 的直接后果）；
- ✅ 更严重的是 **`sources`**：问"年假有几天"，引用却指向
  `员工报销制度.md`——**上一轮的候选被 `merge_candidates` 合并回来了**。
  用户看到的是"有出处"的错误引用，**而整个链路不报任何错**；
- ✅ 换成"每请求一个 `thread_id`"后，两轮完全隔离
  （`all_queries` 只有本次的，`sources` 指向正确文档）；
- ✅ 顺带验证了 `build_checkpointer(cfg)` 在 `sqlite` 后端下能建出
  `{data_dir}/agent_checkpoints.sqlite`。

⚠ **这段输出里还有一条 langgraph 的警告**（本会话真实出现，不是错误）：

```
[warning] Deserializing unregistered type knowflow.retrieval.types.Candidate from checkpoint.
This will be blocked in a future version. Set LANGGRAPH_STRICT_MSGPACK=true to block now,
or add to allowed_msgpack_modules to allow explicitly: [('knowflow.retrieval.types', 'Candidate')]
```

当前版本只是警告；**未来版本会默认拦下**（§2.10 已说明）。

### 验证 6：SSE 事件顺序与"流式/非流式一致"（对应 E6，⚠ 需要起 HTTP 服务）

```powershell
cd D:\Projects\pythonProjects\knowflow
.\.venv\Scripts\python.exe backend\scripts\smoke_http.py
```

**预期输出**（E6.3 原始记录，真起 uvicorn + 手写解析 SSE 分帧）：

```
  PASS = 70
  FAIL = 0
RESULT: ALL PASS
```

其中与本阶段直接相关的关键断言（E6 第 [10] 项）：

```
事件序列: ['meta','trace','trace','trace','tool','trace','trace','trace','trace',
           'sources','token'×15,'trace','trace','reflect','trace','trace',
           'sources','token'×16,'trace','done','end']
token 帧数 = 32，拼接长度 = 252
[PASS] 首帧是 meta
[PASS] 末帧是 end
[PASS] token 帧数量 > 3（确实在流式）
[PASS] sources 在 token 之前
[PASS] 事件顺序 meta → … → done → end
```

**判断标准**：
- ✅ **`sources` 必须在 `token` 之前**：前端要先拿到引用列表，才能在做流式渲染时
  给 `[n]` 加角标。这是 §2.13 里 `emit({"event": "sources", ...})` 写在
  `_generate_text` 之前的原因；
- ✅ **首帧 `meta`、末帧 `end`**——`runner` 保证的帧序（§2.13）；
- ✅ `reflect` 帧出现在**两次 `sources` 之间**：说明 `generate → reflect → generate`
  的重试路径真的走了一次（E6 那一轮的答案第一次没通过引用校验）。

⚠ **本会话未重跑这条命令**（它需要真实起服务、连 MySQL 与文件目录），
上面是 E6/E6.3 的原始记录。另外 E6.5 的 `contextvars` bug **只有真起 uvicorn
打一次流式接口才会暴露**（`pytest` 的 `ASGITransport` 不触发分次线程调度）。

### 验证 7：`rag` 快路径 vs `agent` 全图（对应 E6，⚠ 未重跑）

```
[11] rag 快路径 vs agent 全图（离线 Mock 下的相对对比）
      mode=rag    总耗时=  317ms retrieval_rounds=1 reflect=True  tokens=1704+255
      mode=agent  总耗时=  691ms retrieval_rounds=3 reflect=False tokens=5433+739
```

（E6 原始记录。**离线 Mock 下的耗时只反映"多跑了几次模型调用"这件事的量级，
不代表真实模型下的差距**——真实差距要看模型延迟。）

**这张对比就是 `build_fixed_chain` 存在的理由**：

```python
# backend/src/knowflow/agent/graph.py L190-L196（节选）
def build_fixed_chain(deps: AgentDeps) -> Any:
    """ "快路径"：只有 analyze → retrieve → generate，不做 grade/rewrite/reflect。

    存在的意义是**延迟对比**：评测里用 `rag` 模式跑一遍，能得到
    "自我纠正换来了多少准确率、多花了多少毫秒"的真实数字。
    这也是面试里回答"你的 Agent 比固定链好在哪"的依据。
    """
```

**"你的 Agent 比固定链好在哪"** 这个问题必须用**数字**回答，不能用形容词。
本阶段提供了造数字的工具（`rag` / `agent` 两条路），阶段 10/11 用它产出指标。

### 验证 8：回归测试（本会话真实执行）

```powershell
cd D:\Projects\pythonProjects\knowflow\backend
$env:EMBEDDING_PROVIDER='hash'; $env:VECTOR_BACKEND='memory'
..\.venv\Scripts\python.exe -m pytest -q
```

**本会话真实输出**：

```
220 passed, 1 warning in 4.31s
```

**口径说明**：E1.5 记录的是 **`204 passed, 1 warning in 4.63s`**；
本会话是 **220**（多出的是当前工作区新增的 `tests/test_tokenizer_parity.py` 的 16 个用例）。
⚠ 而**保护本阶段的三个测试文件
（`test_agent_graph.py`：四个条件边判据、拒答短路、反思重试；
`test_memory.py`：双重截断、消息落库与计数自增；`test_services_sqlite.py`：全流程）
都不在当前工作区**（见 README §0）。
所以本阶段的"四个判据"与"双重截断"这两条承诺，
本会话是用 §3 验证 1 与验证 3 **手写脚本**验证的。

---

## 4. 常见错误

| 错误信息 | 原因 | 解决 |
| --- | --- | --- |
| 第二轮提问引用了第一轮的内容（不报错） | `thread_id` 用了 `conversation_id`，而 `candidates` 是累加 reducer | `thread_id` 用 `trace_id`；多轮记忆走 MySQL + 双重截断 |
| `grade` 过滤掉的不相关片段**又回来了** | reducer 是 `left + right`，返回更少的元素删不掉东西 | 返回 `relevant_ids`，由 `generate` 调 `select_contexts` 去筛 |
| prompt 里两个引用编号指向同一段原文 | `candidates` 没有去重 reducer | `merge_candidates` 按 `vector_id` 去重取高分 |
| token 成本少算了几轮 | `usage` 用覆盖语义 | `Annotated[dict, merge_usage]`；`ChatUsage.merge()` |
| `ValueError: <Token var=<ContextVar name='current_recorder'>> was created in a different Context` | 用 contextvars 传 recorder，而 Starlette 逐元素在线程池里迭代同步生成器（E6.5） | 改成显式传递（`RecorderBox` → `AgentDeps` → `RequestServices`） |
| 流式每次都发 `error` 帧、`done` 永远不出现 | 同上（E6.5 的现象） | 同上；**这条只有真起 uvicorn 打流式才会暴露** |
| 一次提问变成 20 次模型调用 | 工具循环没有上限 | `MAX_TOOL_ITERATIONS=3` + `MAX_SEARCH_CALLS=4` |
| 本地小模型一用 Agent 就报错 | 模型不支持 function calling，没有降级路径 | `supports_tools()` 为 `False` 时退化成"按查询逐条检索" |
| 工具返回的父块把下一轮 prompt 撑爆 | `to_tool_message()` 没有长度上限 | `context_text[:1200]` |
| 会话列表的消息数比实际少 | 用了 `threading.Lock`，多进程下失效 | `session.get(Conversation, id, with_for_update=True)` |
| 第 20 轮 prompt 是第 1 轮的 20 倍 | 只按轮数截断（用户可能粘 3000 字） | 轮数 + 字符双上限，取更严者，**至少保留最后一轮** |
| 记忆里出现了"答案在问题前面" | 从后往前装之后忘了 `reverse()` 回去 | `kept.reverse()`（时间正序） |
| 前端显示一个"空气泡" | 图跑完没有答案，返回了空字符串 | `runner` 兜底换成"抱歉，本次生成没有产出内容……" |
| 用户刷新页面导致回答中断、留下无回答的提问 | `emit()` 的异常冒出去了 | `emit` 内 `try/except` 吞掉（前端掉线不该让问答失败） |
| 明明不该检索的闲聊也去查库 | `analyze` 解析失败时默认不检索 | 默认 `needs_retrieval=True`（多做无害） |
| 判官输出格式坏了，用户多等一整轮 | `reflect` 解析失败时默认"不通过" | 解析失败默认 **`passed=True`**（少做无害） |
| 条件边写在 lambda 里，改错只能在线上发现 | 内联 lambda 没法单测 | 抽成 `route_after_*` 纯函数 + `add_conditional_edges` 里注入 `cfg` |
| `Deserializing unregistered type ... from checkpoint` | checkpoint 里有自定义对象（`Candidate`） | 当前是警告；未来版本要设 `LANGGRAPH_STRICT_MSGPACK=true` + `allowed_msgpack_modules` |
| checkpointer 建不起来导致服务起不来 | 把可观测性的失败当成了致命错误 | `build_checkpointer` 任何异常都返回 `None` |

---

## 5. 自测问题

1. `queries` 用覆盖、`all_queries` 用累加——如果两个都改成累加，会坏在哪？
2. 为什么 `grade` 返回 `relevant_ids` 而不是过滤后的 `candidates`？
   请用"reducer 是 `left + right`"这句话解释，并说明这个 bug 为什么不报错。
3. `select_contexts` 为什么在 `relevant_ids` 全空时要退回前 3 条？直接拒答不是更安全吗？
4. 四条条件边里，哪一条是"省一次模型调用"的设计？为什么拒答不需要反思？
5. `analyze` 解析失败默认"要检索"、`reflect` 解析失败默认"通过"——
   这两个默认值的方向为什么相反？判断依据是什么？
6. 工具循环的两个上限（轮数与检索次数）为什么要同时存在？只留一个会怎样？
7. 为什么 `thread_id` 用 `trace_id` 而不是 `conversation_id`？
   用 `conversation_id` 会导致引用错到什么程度（用 §3 验证 5 的输出说）？
8. `SELECT ... FOR UPDATE` 比 `threading.Lock` 好在哪里？不换的话现象是什么（为什么难查）？
9. `RecorderBox` 为什么能解决 contextvars 的跨线程问题？为什么 204 个单测发现不了它？
10. `runner` 里的"空答案兜底"防的是什么体验问题？为什么不能返回空字符串？

---

## 6. 本阶段小结

| 文件 | 行数 | 作用 |
| --- | --- | --- |
| `agent/__init__.py` | 28 | 包出口 |
| `agent/state.py` | 109 | `AgentState` + 三个 reducer（覆盖 / 累加 / 去重取高分） |
| `agent/nodes.py` | 951 | 六个节点 + 工具循环 + `RecorderBox` + `select_contexts` |
| `agent/graph.py` | 227 | 四条条件边（纯函数）+ 装配 + checkpointer + `rag` 快路径 |
| `agent/tools.py` | 295 | 工具注册表 + OpenAI function-calling schema + 结果长度上限 |
| `agent/memory.py` | 170 | 双重截断 + `SELECT ... FOR UPDATE` + 行锁自增计数 |
| `agent/runner.py` | 285 | 图的门面：流式/非流式共用逻辑 + 帧序保证 + 空答案兜底 |
| **合计** | **2,065** | |

（行数按 `\n` 计数（本会话实测）。E10 记录的是 `agent 7 files 1,734 lines`——
E10 是某个时点的快照，**本教程的代码片段与行号以当前文件为准**。）

**四个可讲的设计点**（`docs/00` 第 1.2 节列的，面试官几乎必问，答案都在本阶段）：
1. **为什么 `grade` 用 LLM 判定**（而不是复用检索分数）：相似 ≠ 能回答；
   代价是多一次便宜的小模型调用 → 所以做成可开关。
2. **为什么 `rewrite` 有次数上限**：避免死循环烧钱；用尽就走"明确告知没有足够信息"。
3. **为什么 `reflect` 不重新检索**：职责分离——`grade` 管料，`reflect` 管话。
4. **为什么图要有 checkpointer**：`thread_id=trace_id` 做**可回溯**，
   而多轮记忆用"MySQL 消息 + 双重截断"，两者不能混。

```powershell
# 本阶段结束时的回归
cd D:\Projects\pythonProjects\knowflow\backend
$env:EMBEDDING_PROVIDER='hash'; $env:VECTOR_BACKEND='memory'
..\.venv\Scripts\python.exe -m pytest -q
```

⚠ 保护本阶段的 `test_agent_graph.py` / `test_memory.py` **不在当前工作区**（见 README §0），
所以上面的全量跑**覆盖不到**本阶段的核心承诺——这正是 §3 里我用 5 个手写脚本
逐个验证四条判据、reducer、双重截断、整图与 thread 隔离的原因。

**下一步**：阶段 8 会把这张图接上数据库与业务——
"写三处并保证一致"（向量库 / BM25 / MySQL）、两段事务、软删除语义。
本阶段刻意让图**完全不依赖数据库**，就是为了让阶段 8 的编排层能独立验证。

---

## 7. 延伸阅读

- `backend/src/knowflow/agent/state.py` — 三个 reducer 与"凭什么这么选"
- `backend/src/knowflow/agent/nodes.py` — `RecorderBox` 的 contextvars 事故、
  `_run_tool_loop` 的两个硬上限、`select_contexts` 的兜底
- `backend/src/knowflow/agent/graph.py` — 四条纯函数判据 + checkpointer 的 `thread_id` 决定
- `backend/src/knowflow/agent/memory.py` — 双重截断与 `with_for_update()`
- `backend/src/knowflow/agent/runner.py` — `stream_mode=["custom","updates"]` 与帧序保证
- `backend/src/knowflow/agent/tools.py` — "工具 description 是产品代码"
- `docs/00-架构与技术选型.md` 第 1 节 — Agent 状态图与四个可讲的设计点
- `docs/03-实测证据.md` E6（拒答短路、记忆、SSE 帧序、rag vs agent）、
  E6.3（51 帧与 `meta` 字段）、**E6.5（contextvars 跨线程，必读）**、E11（未实测清单）
- 官方文档：[LangGraph StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/)、
  [Persistence / Checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/)、
  [Starlette `iterate_in_threadpool`](https://www.starlette.io/concurrency/)
