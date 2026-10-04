/**
 * SSE 事件流类型（契约 5.6 节）。
 *
 * 服务端帧格式：`event: <name>\ndata: <json>\n\n`
 * 顺序保证：meta → (trace|tool|reflect)* → sources → token* → done → end
 * 出错：发 error 帧，然后 end。
 *
 * 事件名共 9 个：meta / trace / tool / sources / reflect / token / done / error / end
 */

import type { ChatResponse, Source } from './models'

/** 9 种事件名（含 `end`，注意别漏） */
export type SSEEventName =
  | 'meta'
  | 'trace'
  | 'tool'
  | 'sources'
  | 'reflect'
  | 'token'
  | 'done'
  | 'error'
  | 'end'

/** 解析后的原始帧：data 尚未做类型断言，避免在解析层引入 any */
export interface SSERawEvent {
  event: SSEEventName
  data: unknown
}

/** 第一帧：前端据此建立助手气泡、锁定 conversation_id */
export interface MetaEvent {
  conversation_id: number
  mode: string
  trace_id: string
  model: string
  offline: boolean
  embedding_mode: string
}

/** 图节点开始/结束，前端画「思考过程」时间线 */
export interface TraceEvent {
  node: string
  /**
   * 节点状态。
   * 后端 `TraceEvent.status` 的取值是 `ok` / `error`（**没有 running**）：
   * 每个节点只在结束时产出一帧，附带耗时。
   */
  status: string
  duration_ms: number
  /** 节点附加信息（如候选条数），结构随节点而定 */
  detail?: unknown
}

/** 工具调用（检索） */
export interface ToolEvent {
  name: string
  /**
   * 调用参数（已截断），如 query/top_k。
   * 后端类型是 `Any | None`，所以不保证是对象，这里按 unknown 收下再由渲染层容错。
   */
  args: unknown
  result_count: number
  duration_ms: number
}

/** 引用来源，先于 token 到达，用户立刻有反馈 */
export interface SourcesEvent {
  sources: Source[]
}

/**
 * 反思里「未被支撑的陈述」的一项。
 * 后端 `ReflectEvent.unsupported` 是 `list[dict]`，每项形如 `{sentence, reason}`；
 * 若判官给出裸字符串，后端会包装成 `{sentence: ...}`。
 */
export interface UnsupportedClaim {
  sentence: string
  reason: string
  /** 后端 validator 之外的额外键（原样保留，便于排障） */
  [key: string]: unknown
}

/** 自我反思（Self-RAG）结果 */
export interface ReflectEvent {
  round: number
  passed: boolean
  unsupported: UnsupportedClaim[]
  /** 下一步动作：accept（接受）/ regenerate（重新生成） */
  action: string
}

/** 增量文本 */
export interface TokenEvent {
  text: string
}

/** 收尾统计：就是 ChatResponse 的完整字段 */
export type DoneEvent = ChatResponse

/** 错误帧 */
export interface ErrorEvent {
  code: string
  message: string
  request_id: string
}

/** 关闭流，data 恒为 {} */
export interface EndEvent {
  [key: string]: never
}

/** 事件名 → data 类型的映射（供 useChatStream 做类型收窄） */
export interface SSEEventPayloadMap {
  meta: MetaEvent
  trace: TraceEvent
  tool: ToolEvent
  sources: SourcesEvent
  reflect: ReflectEvent
  token: TokenEvent
  done: DoneEvent
  error: ErrorEvent
  end: EndEvent
}

/** 事件名 → 回调类型 */
export type SSEHandler<K extends SSEEventName> = (data: SSEEventPayloadMap[K]) => void

/** 全部回调可选；未提供的事件直接忽略 */
export type SSEHandlers = {
  [K in SSEEventName]?: SSEHandler<K>
}

/** streamChat 的参数 */
export interface StreamChatOptions {
  /** 请求体（ChatRequest 结构，见 types/dto.ts） */
  body: Record<string, unknown>
  /** 事件回调集合 */
  handlers: SSEHandlers
  /** 外部取消（用户点「停止生成」） */
  signal?: AbortSignal
}
