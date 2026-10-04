/**
 * SSE 客户端：`fetch` + `ReadableStream` 手工解析（**绝不能用 EventSource**）。
 *
 * 为什么不能用 EventSource：
 * 1. 原生 EventSource 只支持 GET，而 `/chat/stream` 是 POST；
 * 2. 原生 EventSource 不能自定义请求头，带不了 `Authorization: Bearer`。
 *
 * 本模块负责：
 * - 增量解码（TextDecoder `{stream:true}`）；
 * - 分帧解析（委托给 sse-parser.ts 的纯函数，可在 Node 里单测）；
 * - 事件 dispatch（9 种事件全部支持）；
 * - AbortController 取消（用户点「停止生成」）；
 * - 两种异常路径：(a) 连接中断；(b) 非 2xx 且响应体是错误信封 JSON 而不是 event-stream。
 */

import { API_BASE_URL, ApiError, getToken, toApiError } from './client'
import { parseSSEChunk, type ParsedSSEEvent } from './sse-parser'
import {
  coerceDone,
  coerceEnd,
  coerceErrorEvent,
  coerceMeta,
  coerceReflect,
  coerceSourcesEvent,
  coerceToken,
  coerceTool,
  coerceTrace,
  type CoerceResult,
} from '@/types/sse-guards'
import type { SSEEventName, SSEHandlers, StreamChatOptions } from '@/types/sse'

/** 解析层用 unknown 承载 data，dispatch 时按事件名挑对应的守卫做收窄 */
type Coercer = (raw: unknown) => CoerceResult<unknown>

const COERCERS: Record<SSEEventName, Coercer> = {
  meta: coerceMeta,
  trace: coerceTrace,
  tool: coerceTool,
  sources: coerceSourcesEvent,
  reflect: coerceReflect,
  token: coerceToken,
  done: coerceDone,
  error: coerceErrorEvent,
  end: coerceEnd,
}

/** 连接中断（流被掐断、没有收到 end 帧）时抛出的错误 */
export class StreamInterruptedError extends ApiError {
  constructor(message: string) {
    super({ code: 'STREAM_INTERRUPTED', message, status: 0 })
    this.name = 'StreamInterruptedError'
  }
}

/**
 * 是「用户主动取消」而不是真错误？
 * AbortController.abort() 会抛 AbortError，我们要把它和真异常区分开，
 * 前者不应该弹红色错误提示。
 */
export function isAbortLike(error: unknown): boolean {
  if (error instanceof DOMException && error.name === 'AbortError') return true
  if (error instanceof Error && error.name === 'AbortError') return true
  if (typeof error === 'object' && error !== null) {
    const name = (error as { name?: unknown }).name
    if (name === 'AbortError') return true
    const code = (error as { code?: unknown }).code
    if (code === 'ABORT_ERR' || code === 'CANCELED') return true
  }
  return false
}

/** data 为空（end 帧 `data: {}` 或空 data 行）时给个空对象，避免守卫直接失败 */
function normalizeRawData(event: ParsedSSEEvent): unknown {
  return event.data === undefined || event.data === null || event.data === '' ? {} : event.data
}

/**
 * 非 2xx 响应：后端返回的是统一错误信封 JSON，而不是 event-stream。
 * 这里手工读文本并解包，给出可读错误（并把 request_id 带出来）。
 */
async function throwHttpError(response: Response): Promise<never> {
  let text = ''
  try {
    text = await response.text()
  } catch {
    text = ''
  }

  let body: unknown = null
  if (text.trim() !== '') {
    try {
      body = JSON.parse(text) as unknown
    } catch {
      body = text
    }
  }

  const envelope = body as { error?: { code?: unknown; message?: unknown; request_id?: unknown; detail?: unknown } } | null
  if (envelope && typeof envelope === 'object' && envelope.error && typeof envelope.error === 'object') {
    const err = envelope.error
    throw new ApiError({
      code: typeof err.code === 'string' ? err.code : `HTTP_${response.status}`,
      message: typeof err.message === 'string' ? err.message : `请求失败（HTTP ${response.status}）`,
      status: response.status,
      requestId: typeof err.request_id === 'string' ? err.request_id : null,
      detail: err.detail ?? null,
    })
  }

  throw new ApiError({
    code: `HTTP_${response.status}`,
    message:
      typeof body === 'string' && body.trim() !== ''
        ? `请求失败（HTTP ${response.status}）：${body.slice(0, 200)}`
        : `请求失败（HTTP ${response.status}）`,
    status: response.status,
    detail: body,
  })
}

/**
 * 发起一次流式对话，把 9 种事件回调出去。
 *
 * @returns Promise，流正常结束（收到 end 或 done 后流关闭）时 resolve；
 *          出错时 reject（用户取消时 reject 一个 AbortError，调用方用 isAbortLike 判断）。
 */
export async function streamChat(options: StreamChatOptions): Promise<void> {
  const { body, handlers, signal } = options

  const token = getToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  if (token) headers.Authorization = `Bearer ${token}`

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
      // 说明：SSE 不使用浏览器 HTTP 缓存
      cache: 'no-store',
    })
  } catch (error) {
    if (isAbortLike(error)) throw error
    throw toApiError(error)
  }

  // 情况 (b)：非 2xx —— 响应体是错误信封 JSON，不是 event-stream
  if (!response.ok) {
    await throwHttpError(response)
  }

  const stream = response.body
  if (!stream) {
    throw new StreamInterruptedError('浏览器不支持流式响应（response.body 为空），无法接收增量内容')
  }

  const reader = stream.getReader()
  // 关键：{stream:true} 让一个被切断的多字节汉字能在下一批数据里补齐
  const decoder = new TextDecoder('utf-8')

  let buffer = ''
  let sawEnd = false
  let sawDone = false

  const dispatch = (event: ParsedSSEEvent): void => {
    const name = event.event as SSEEventName
    const coercer = COERCERS[name]
    if (!coercer) {
      // 未知事件名：忽略而不报错，保证后端新增事件时前端不会崩
      console.warn(`[knowflow][sse] 收到未知事件名：${event.event}`)
      return
    }

    const result = coercer(normalizeRawData(event))
    if (!result.ok) {
      console.warn(`[knowflow][sse] 事件 ${name} 的 data 缺字段：${result.missing}`, event.data)
      return
    }

    if (name === 'end') sawEnd = true
    if (name === 'done') sawDone = true

    // 类型收窄在 COERCERS 里按事件名成对绑定，此处是唯一的断言点
    const handler = handlers[name] as ((data: unknown) => void) | undefined
    if (handler) handler(result.value)
  }

  const consume = (text: string): void => {
    buffer += text
    const { events, rest } = parseSSEChunk(buffer)
    buffer = rest
    for (const event of events) dispatch(event)
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      if (value) consume(decoder.decode(value, { stream: true }))
    }
    // 冲刷解码器里可能残存的多字节片段
    consume(decoder.decode())
    if (buffer.trim() !== '') consume('\n\n')
  } catch (error) {
    if (isAbortLike(error)) throw error
    throw new StreamInterruptedError(`流式连接中断：${error instanceof Error ? error.message : String(error)}`)
  } finally {
    reader.releaseLock()
  }

  // 情况 (a)：连接中断 —— 没有 error 帧、也没有收齐 done/end，属于异常结束
  if (!sawEnd && !sawDone) {
    throw new StreamInterruptedError('流式连接异常中断：未收到 end 帧，回答可能不完整')
  }
}

/** 供上层构造取消控制器 */
export function createStreamAbortController(): AbortController {
  return new AbortController()
}

/** 重新导出一遍，方便页面从 `@/api/sse` 一个入口拿到全部 SSE 能力 */
export { parseSSEChunk, KNOWN_SSE_EVENTS } from './sse-parser'
export type { ParsedSSEEvent, ParseSSEChunkResult } from './sse-parser'