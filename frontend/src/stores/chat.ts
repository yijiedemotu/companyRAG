/**
 * 对话 store：会话列表、消息、流式状态。
 *
 * 设计要点：
 * - 消息列表是「服务端消息 + 本地乐观消息」合流后的结果；流式回答落在同一条本地消息上，
 *   收到 `done` 帧后再用服务端返回的 message_id / usage / cost 回填；
 * - 流式状态（是否在生成、当前节点）放在 store 里，页面切换/组件卸载都不会丢；
 * - Agent 思考过程（trace / tool / reflect）按到达顺序存成时间线，供 AgentTracePanel 渲染。
 */

import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import * as conversationsApi from '@/api/conversations'
import type { ChatMode, CitationOut, ConversationOut, MessageOut, Source } from '@/types/models'
import type { GradingItem, ChatUsage } from '@/types/models'
import type { MetaEvent, UnsupportedClaim } from '@/types/sse'

/* ------------------------------------------------------------------ UI 消息模型 */

/** Agent 思考过程时间线的一项：trace 节点 / 工具调用 / 反思 */
export type AgentTraceItem =
  | {
      kind: 'node'
      node: string
      status: string
      duration_ms: number
      detail: unknown
      at: number
    }
  | {
      kind: 'tool'
      name: string
      /** 工具调用参数。后端是 `Any | None`（不保证是对象），原样保留由渲染层容错 */
      args: unknown
      result_count: number
      duration_ms: number
      at: number
    }
  | {
      kind: 'reflect'
      round: number
      passed: boolean
      /** 未被检索内容支撑的陈述；后端每项形如 `{sentence, reason}` */
      unsupported: UnsupportedClaim[]
      action: string
      at: number
    }

export type ChatMessageStatus = 'pending' | 'streaming' | 'done' | 'error' | 'stopped'

/** 页面渲染用的消息模型（服务端 MessageOut + 流式期本地状态） */
export interface ChatMessage {
  /** 服务端 id；本地乐观消息为 null */
  id: number | null
  /** 本地唯一 key，列表渲染用 */
  localId: string
  conversationId: number | null
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  citations: CitationOut[]
  /** 来源列表（来自 sources 事件，比 citations 多了页码/小节/分数） */
  sources: Source[]
  status: ChatMessageStatus
  createdAt: string
  mode: ChatMode | null
  model: string | null
  promptTokens: number
  completionTokens: number
  costUsd: number
  latencyMs: number | null
  refusal: boolean
  retrievalRounds: number
  traceId: string | null
  /** 助手消息的思考过程 */
  trace: AgentTraceItem[]
  grading: GradingItem[]
  rewrittenQueries: string[]
  reflectPassed: boolean | null
  usage: ChatUsage | null
  /** 出错时的可读文案 */
  errorText: string | null
  /** 服务端消息是「重建」的（来自 /messages），不参与再持久化 */
  hydrated: boolean
}

let localSeq = 0
function nextLocalId(prefix: string): string {
  localSeq += 1
  return `${prefix}-${Date.now().toString(36)}-${localSeq}`
}

/** 把服务端 MessageOut 转成页面模型 */
export function toChatMessage(raw: MessageOut): ChatMessage {
  return {
    id: raw.id,
    localId: `srv-${raw.id}`,
    conversationId: raw.conversation_id,
    role: raw.role,
    content: raw.content,
    citations: raw.citations ?? [],
    // 历史消息里没有 sources 事件，用 citations 兜底，保证「引用卡片」照样能画
    sources: (raw.citations ?? []).map((citation) => ({
      rank: citation.rank,
      chunk_id: citation.chunk_id ?? 0,
      doc_id: citation.doc_id ?? 0,
      doc_name: citation.doc_name ?? `片段 #${citation.chunk_id ?? '?'}`,
      page_no: citation.page_no ?? null,
      section_path: citation.section_path ?? null,
      score: citation.score,
      snippet: citation.snippet,
    })),
    status: 'done',
    createdAt: raw.created_at,
    mode: raw.mode,
    model: raw.model,
    promptTokens: raw.prompt_tokens,
    completionTokens: raw.completion_tokens,
    costUsd: raw.cost_usd,
    latencyMs: raw.latency_ms,
    refusal: raw.refusal,
    retrievalRounds: raw.retrieval_rounds,
    traceId: raw.trace_id,
    trace: [],
    grading: [],
    rewrittenQueries: [],
    reflectPassed: null,
    usage: {
      prompt_tokens: raw.prompt_tokens,
      completion_tokens: raw.completion_tokens,
      total_tokens: raw.prompt_tokens + raw.completion_tokens,
    },
    errorText: null,
    hydrated: true,
  }
}

/* ------------------------------------------------------------------ store */

export const useChatStore = defineStore('chat', () => {
  /* 会话列表 */
  const conversations = ref<ConversationOut[]>([])
  const conversationsTotal = ref(0)
  const conversationsPage = ref(1)
  const conversationsPages = ref(0)
  const conversationsLoading = ref(false)

  /* 当前会话与消息 */
  const currentConversationId = ref<number | null>(null)
  const currentConversation = ref<ConversationOut | null>(null)
  const messages = ref<ChatMessage[]>([])
  const messagesLoading = ref(false)

  /* 流式状态 */
  const streaming = ref(false)
  const streamMessageId = ref<string | null>(null)
  const abortController = ref<AbortController | null>(null)
  /** 正在执行的图节点（流式过程中 AgentTracePanel 自动展开它） */
  const activeNode = ref<string | null>(null)

  const lastAssistant = computed<ChatMessage | null>(() => {
    for (let index = messages.value.length - 1; index >= 0; index -= 1) {
      const item = messages.value[index]
      if (item && item.role === 'assistant') return item
    }
    return null
  })

  function findMessage(localId: string): ChatMessage | null {
    return messages.value.find((item) => item.localId === localId) ?? null
  }

  /* ---------------------------------------------------------------- 会话列表 */

  async function fetchConversations(options: { page?: number; size?: number; kbId?: number } = {}): Promise<void> {
    conversationsLoading.value = true
    try {
      const result = await conversationsApi.listConversations({
        page: options.page ?? 1,
        size: options.size ?? 30,
        kb_id: options.kbId,
      })
      conversations.value = result.items
      conversationsTotal.value = result.total
      conversationsPage.value = result.page
      conversationsPages.value = result.pages
    } finally {
      conversationsLoading.value = false
    }
  }

  /** 把新建的会话插到列表最前面 */
  function upsertConversation(conversation: ConversationOut): void {
    const index = conversations.value.findIndex((item) => item.id === conversation.id)
    if (index >= 0) conversations.value[index] = conversation
    else conversations.value = [conversation, ...conversations.value]
    if (currentConversationId.value === conversation.id) currentConversation.value = conversation
  }

  async function createConversation(kbId: number | null, title?: string): Promise<ConversationOut> {
    const conversation = await conversationsApi.createConversation({ kb_id: kbId, title })
    upsertConversation(conversation)
    return conversation
  }

  async function removeConversation(conversationId: number): Promise<void> {
    await conversationsApi.deleteConversation(conversationId)
    conversations.value = conversations.value.filter((item) => item.id !== conversationId)
    conversationsTotal.value = Math.max(0, conversationsTotal.value - 1)
    if (currentConversationId.value === conversationId) {
      currentConversationId.value = null
      currentConversation.value = null
      messages.value = []
    }
  }

  function startNewChat(): void {
    currentConversationId.value = null
    currentConversation.value = null
    messages.value = []
    activeNode.value = null
  }

  /* ---------------------------------------------------------------- 消息加载 */

  async function loadMessages(conversationId: number): Promise<void> {
    messagesLoading.value = true
    try {
      currentConversationId.value = conversationId
      // 拉全部消息：契约里 messages 是分页的，这里按 size=100 取最近一屏，
      // 更大的会话应改成分页上滑加载（见 README「已知取舍」）
      const result = await conversationsApi.listMessages(conversationId, { page: 1, size: 100 })
      messages.value = result.items.map(toChatMessage).sort((a, b) => {
        if (a.createdAt === b.createdAt) return (a.id ?? 0) - (b.id ?? 0)
        return a.createdAt < b.createdAt ? -1 : 1
      })
      const meta = conversations.value.find((item) => item.id === conversationId)
      if (meta) currentConversation.value = meta
      else currentConversation.value = await conversationsApi.getConversation(conversationId)
    } finally {
      messagesLoading.value = false
    }
  }

  async function submitFeedback(messageId: number, rating: number, comment?: string): Promise<void> {
    await conversationsApi.submitFeedback(messageId, { rating, comment })
  }

  /* ---------------------------------------------------------------- 流式写入 */

  /** 乐观插入用户提问（不等服务端） */
  function pushUserMessage(question: string, conversationId: number | null): ChatMessage {
    const message: ChatMessage = {
      id: null,
      localId: nextLocalId('user'),
      conversationId,
      role: 'user',
      content: question,
      citations: [],
      sources: [],
      status: 'done',
      createdAt: new Date().toISOString(),
      mode: null,
      model: null,
      promptTokens: 0,
      completionTokens: 0,
      costUsd: 0,
      latencyMs: null,
      refusal: false,
      retrievalRounds: 0,
      traceId: null,
      trace: [],
      grading: [],
      rewrittenQueries: [],
      reflectPassed: null,
      usage: null,
      errorText: null,
      hydrated: false,
    }
    messages.value.push(message)
    return message
  }

  /**
   * 建立助手消息气泡。
   * `meta` 帧到达前也可能先收到 trace/token（虽然契约里 meta 恒为第一帧），
   * 所以这里做幂等：已有 streaming 气泡就复用，避免出现两个空泡。
   */
  function ensureAssistantMessage(conversationId: number | null): ChatMessage {
    const existing = streamMessageId.value ? findMessage(streamMessageId.value) : null
    if (existing) return existing

    const message: ChatMessage = {
      id: null,
      localId: nextLocalId('assistant'),
      conversationId,
      role: 'assistant',
      content: '',
      citations: [],
      sources: [],
      status: 'streaming',
      createdAt: new Date().toISOString(),
      mode: null,
      model: null,
      promptTokens: 0,
      completionTokens: 0,
      costUsd: 0,
      latencyMs: null,
      refusal: false,
      retrievalRounds: 0,
      traceId: null,
      trace: [],
      grading: [],
      rewrittenQueries: [],
      reflectPassed: null,
      usage: null,
      errorText: null,
      hydrated: false,
    }
    messages.value.push(message)
    streamMessageId.value = message.localId
    return message
  }

  function applyMeta(meta: MetaEvent): void {
    if (meta.conversation_id) {
      currentConversationId.value = meta.conversation_id
    }
    const message = ensureAssistantMessage(meta.conversation_id || currentConversationId.value)
    message.conversationId = meta.conversation_id || currentConversationId.value
    message.mode = meta.mode === 'rag' ? 'rag' : meta.mode === 'agent' ? 'agent' : null
    message.model = meta.model
    message.traceId = meta.trace_id
  }

  function appendToken(text: string): void {
    const message = ensureAssistantMessage(currentConversationId.value)
    message.content += text
  }

  function setSources(sources: Source[]): void {
    const message = ensureAssistantMessage(currentConversationId.value)
    message.sources = sources
    message.citations = sources.map((source) => ({
      rank: source.rank,
      chunk_id: source.chunk_id,
      doc_id: source.doc_id,
      score: source.score,
      snippet: source.snippet,
      doc_name: source.doc_name,
      page_no: source.page_no,
      section_path: source.section_path,
    }))
  }

  function pushTraceNode(item: Extract<AgentTraceItem, { kind: 'node' }>): void {
    const message = ensureAssistantMessage(currentConversationId.value)
    // 后端每个节点只在结束时产出一帧（status 是 ok/error，没有 running），
    // 所以正常情况下一节点一帧。这里只做一种容错：若同名节点重复出现
    // （例如 rewrite 回边让 retrieve 再跑一次），且耗时不同才算新的一轮。
    // 用倒序手写循环（而不是 Array.findLastIndex）以兼容 ES2022 运行时。
    let lastSameIndex = -1
    for (let index = message.trace.length - 1; index >= 0; index -= 1) {
      const entry = message.trace[index]
      if (entry && entry.kind === 'node' && entry.node === item.node) {
        lastSameIndex = index
        break
      }
    }
    if (lastSameIndex >= 0) {
      const existing = message.trace[lastSameIndex]
      if (existing && existing.kind === 'node' && existing.duration_ms === item.duration_ms) {
        message.trace.splice(lastSameIndex, 1, item)
      } else {
        message.trace.push(item)
      }
    } else {
      message.trace.push(item)
    }
    // 「当前节点」= 最新产出的节点，用于高亮
    activeNode.value = item.node
  }

  function pushTool(item: Extract<AgentTraceItem, { kind: 'tool' }>): void {
    const message = ensureAssistantMessage(currentConversationId.value)
    message.trace.push(item)
  }

  function pushReflect(item: Extract<AgentTraceItem, { kind: 'reflect' }>): void {
    const message = ensureAssistantMessage(currentConversationId.value)
    message.trace.push(item)
    activeNode.value = 'reflect'
  }

  /** `done` 帧：回填服务端权威数据 */
  function applyDone(payload: {
    conversation_id: number
    message_id: number
    trace_id: string
    answer: string
    sources: Source[]
    refusal: boolean
    retrieval_rounds: number
    reflect_passed: boolean
    grading: GradingItem[]
    rewritten_queries: string[]
    usage: ChatUsage
    cost_usd: number
    latency_ms: number
    model: string
  }): void {
    const message = ensureAssistantMessage(payload.conversation_id || currentConversationId.value)
    if (payload.message_id) message.id = payload.message_id
    message.conversationId = payload.conversation_id || currentConversationId.value
    // 服务端可能对答案做了裁剪（例如去掉思考前缀），以服务端为准
    if (payload.answer) message.content = payload.answer
    if (payload.sources.length > 0) setSources(payload.sources)
    message.refusal = payload.refusal
    message.retrievalRounds = payload.retrieval_rounds
    message.reflectPassed = payload.reflect_passed
    message.grading = payload.grading
    message.rewrittenQueries = payload.rewritten_queries
    message.usage = payload.usage
    message.promptTokens = payload.usage.prompt_tokens
    message.completionTokens = payload.usage.completion_tokens
    message.costUsd = payload.cost_usd
    message.latencyMs = payload.latency_ms
    message.model = payload.model || message.model
    message.traceId = payload.trace_id || message.traceId
    message.status = 'done'
    streamMessageId.value = null
    activeNode.value = null

    // 标题/消息计数在服务端已更新，刷新一下当前会话的元信息
    if (message.conversationId) {
      const meta = conversations.value.find((item) => item.id === message.conversationId)
      if (meta) {
        upsertConversation({ ...meta, message_count: meta.message_count + 2 })
      }
    }
  }

  /** `error` 帧 */
  function applyStreamError(code: string, message: string, requestId: string): void {
    const target = ensureAssistantMessage(currentConversationId.value)
    const requestSuffix = requestId ? `（request_id: ${requestId}）` : ''
    target.errorText = `[${code}] ${message}${requestSuffix}`
    target.status = 'error'
    streamMessageId.value = null
    activeNode.value = null
  }

  /** 用户点「停止生成」 */
  function markStopped(): void {
    const target = streamMessageId.value ? findMessage(streamMessageId.value) : lastAssistant.value
    if (target && target.status === 'streaming') target.status = 'stopped'
    streamMessageId.value = null
    activeNode.value = null
  }

  function beginStream(controller: AbortController): void {
    abortController.value = controller
    streaming.value = true
    activeNode.value = null
  }

  function endStream(): void {
    streaming.value = false
    abortController.value = null
    activeNode.value = null
    streamMessageId.value = null
  }

  function abortStream(): void {
    abortController.value?.abort()
  }

  function reset(): void {
    conversations.value = []
    conversationsTotal.value = 0
    conversationsPage.value = 1
    conversationsPages.value = 0
    currentConversationId.value = null
    currentConversation.value = null
    messages.value = []
    streaming.value = false
    streamMessageId.value = null
    abortController.value = null
    activeNode.value = null
  }

  return {
    /* state */
    conversations,
    conversationsTotal,
    conversationsPage,
    conversationsPages,
    conversationsLoading,
    currentConversationId,
    currentConversation,
    messages,
    messagesLoading,
    streaming,
    streamMessageId,
    activeNode,
    /* getters */
    lastAssistant,
    /* actions */
    fetchConversations,
    upsertConversation,
    createConversation,
    removeConversation,
    startNewChat,
    loadMessages,
    submitFeedback,
    pushUserMessage,
    ensureAssistantMessage,
    applyMeta,
    appendToken,
    setSources,
    pushTraceNode,
    pushTool,
    pushReflect,
    applyDone,
    applyStreamError,
    markStopped,
    beginStream,
    endStream,
    abortStream,
    reset,
  }
})
