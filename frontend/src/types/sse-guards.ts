/**
 * SSE 事件的运行期类型守卫。
 *
 * 解析出来的 `data` 是 `unknown`（网络数据不可信），
 * 这里把它收窄成契约里声明的类型；收窄失败时返回 null 并指出缺了什么字段，
 * 而不是打 `as` 断言把类型错误藏起来。
 */

import type {
  DoneEvent,
  ErrorEvent,
  MetaEvent,
  ReflectEvent,
  SourcesEvent,
  TokenEvent,
  ToolEvent,
  TraceEvent,
  UnsupportedClaim,
} from './sse'
import type { ChatResponse, ChatUsage, GradingItem, Source } from './models'

export type CoerceResult<T> = { ok: true; value: T } | { ok: false; missing: string }

const ok = <T>(value: T): CoerceResult<T> => ({ ok: true, value })
const missing = <T>(field: string): CoerceResult<T> => ({ ok: false, missing: field })

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

/* ------------------------------------------------------------------ 基础取值 */

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function asNumberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null
  const parsed = asNumber(value, Number.NaN)
  return Number.isNaN(parsed) ? null : parsed
}

function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return fallback
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return fallback
}

function asStringOrNull(value: unknown): string | null {
  if (value === null || value === undefined) return null
  const text = asString(value, '')
  return text === '' ? null : text
}

function asBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === 'boolean') return value
  if (value === 1 || value === '1' || value === 'true') return true
  if (value === 0 || value === '0' || value === 'false') return false
  return fallback
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
}

/* ------------------------------------------------------------------ 结构化字段 */

/** Source（引用来源），字段名严格按契约 MessageOut.sources / ChatResponse.sources */
export function coerceSource(raw: unknown): Source | null {
  if (!isRecord(raw)) return null
  const docName = asStringOrNull(raw.doc_name)
  const chunkId = asNumberOrNull(raw.chunk_id)
  // doc_name / chunk_id 缺失说明这不是一个合法的引用对象
  if (docName === null && chunkId === null) return null
  return {
    rank: asNumber(raw.rank, 0),
    chunk_id: chunkId ?? 0,
    doc_id: asNumber(raw.doc_id, 0),
    doc_name: docName ?? '未知文档',
    page_no: asNumberOrNull(raw.page_no),
    section_path: asStringOrNull(raw.section_path),
    score: asNumber(raw.score, 0),
    snippet: asString(raw.snippet, ''),
  }
}

export function coerceSources(raw: unknown): Source[] {
  if (!Array.isArray(raw)) return []
  return raw.map(coerceSource).filter((item): item is Source => item !== null)
}

function coerceGrading(raw: unknown): GradingItem[] {
  if (!Array.isArray(raw)) return []
  const result: GradingItem[] = []
  for (const item of raw) {
    if (!isRecord(item)) continue
    result.push({
      chunk_id: asNumber(item.chunk_id, 0),
      relevant: asBoolean(item.relevant, false),
      reason: asString(item.reason, ''),
    })
  }
  return result
}

function coerceUsage(raw: unknown): ChatUsage {
  if (!isRecord(raw)) return { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  const prompt = asNumber(raw.prompt_tokens, 0)
  const completion = asNumber(raw.completion_tokens, 0)
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: asNumber(raw.total_tokens, prompt + completion),
  }
}

/* ------------------------------------------------------------------ 各事件守卫 */

export function coerceMeta(raw: unknown): CoerceResult<MetaEvent> {
  if (!isRecord(raw)) return missing<MetaEvent>('data 不是对象')
  if (raw.conversation_id === undefined || raw.conversation_id === null) {
    return missing<MetaEvent>('conversation_id')
  }
  return ok<MetaEvent>({
    conversation_id: asNumber(raw.conversation_id),
    mode: asString(raw.mode, ''),
    trace_id: asString(raw.trace_id, ''),
    model: asString(raw.model, ''),
    offline: asBoolean(raw.offline, false),
    embedding_mode: asString(raw.embedding_mode, ''),
  })
}

export function coerceTrace(raw: unknown): CoerceResult<TraceEvent> {
  if (!isRecord(raw)) return missing<TraceEvent>('data 不是对象')
  if (raw.node === undefined) return missing<TraceEvent>('node')
  return ok<TraceEvent>({
    node: asString(raw.node),
    // 契约里 status 是「节点开始/结束」，取值可能是 running/ok/error，统一转小写比较
    status: asString(raw.status, 'running').toLowerCase(),
    duration_ms: asNumber(raw.duration_ms, 0),
    detail: raw.detail ?? null,
  })
}

export function coerceTool(raw: unknown): CoerceResult<ToolEvent> {
  if (!isRecord(raw)) return missing<ToolEvent>('data 不是对象')
  if (raw.name === undefined) return missing<ToolEvent>('name')
  return ok<ToolEvent>({
    name: asString(raw.name),
    // 后端是 `Any | None`，原样透传，渲染层用 detailJson 容错
    args: raw.args ?? null,
    result_count: asNumber(raw.result_count, 0),
    duration_ms: asNumber(raw.duration_ms, 0),
  })
}

export function coerceSourcesEvent(raw: unknown): CoerceResult<SourcesEvent> {
  if (!isRecord(raw)) return missing<SourcesEvent>('data 不是对象')
  return ok<SourcesEvent>({ sources: coerceSources(raw.sources) })
}

/**
 * 后端 `ReflectEvent.unsupported` 是 `list[dict]`，每项形如 `{sentence, reason}`；
 * 裸字符串会被后端包装成 `{sentence: ...}`。这里再兜一层，防止字段名漂移导致整帧被丢。
 */
function coerceUnsupported(value: unknown): UnsupportedClaim[] {
  if (!Array.isArray(value)) return []
  const result: UnsupportedClaim[] = []
  for (const item of value) {
    if (typeof item === 'string') {
      result.push({ sentence: item, reason: '' })
      continue
    }
    if (!isRecord(item)) continue
    result.push({
      ...item,
      sentence: asString(item.sentence ?? item.text ?? item.claim, ''),
      reason: asString(item.reason ?? item.why, ''),
    })
  }
  return result
}

export function coerceReflect(raw: unknown): CoerceResult<ReflectEvent> {
  if (!isRecord(raw)) return missing<ReflectEvent>('data 不是对象')
  return ok<ReflectEvent>({
    round: asNumber(raw.round, 1),
    passed: asBoolean(raw.passed, false),
    unsupported: coerceUnsupported(raw.unsupported),
    action: asString(raw.action, ''),
  })
}

export function coerceToken(raw: unknown): CoerceResult<TokenEvent> {
  if (!isRecord(raw)) return missing<TokenEvent>('data 不是对象')
  if (raw.text === undefined) return missing<TokenEvent>('text')
  return ok<TokenEvent>({ text: asString(raw.text) })
}

/** done 帧 = ChatResponse 的完整字段 */
export function coerceDone(raw: unknown): CoerceResult<DoneEvent> {
  if (!isRecord(raw)) return missing<DoneEvent>('data 不是对象')
  const value: ChatResponse = {
    conversation_id: asNumber(raw.conversation_id, 0),
    message_id: asNumber(raw.message_id, 0),
    trace_id: asString(raw.trace_id, ''),
    answer: asString(raw.answer, ''),
    sources: coerceSources(raw.sources),
    refusal: asBoolean(raw.refusal, false),
    retrieval_rounds: asNumber(raw.retrieval_rounds, 0),
    reflect_passed: asBoolean(raw.reflect_passed, true),
    grading: coerceGrading(raw.grading),
    rewritten_queries: asStringArray(raw.rewritten_queries),
    usage: coerceUsage(raw.usage),
    cost_usd: asNumber(raw.cost_usd, 0),
    latency_ms: asNumber(raw.latency_ms, 0),
    model: asString(raw.model, ''),
    offline: asBoolean(raw.offline, false),
  }
  return ok(value)
}

export function coerceErrorEvent(raw: unknown): CoerceResult<ErrorEvent> {
  if (!isRecord(raw)) return missing<ErrorEvent>('data 不是对象')
  return ok<ErrorEvent>({
    code: asString(raw.code, 'INTERNAL_ERROR'),
    message: asString(raw.message, '服务端发生未知错误'),
    request_id: asString(raw.request_id, ''),
  })
}

/** end 帧 data 恒为 {}，无论内容是什么都算成功 */
export function coerceEnd(): CoerceResult<Record<string, never>> {
  return ok<Record<string, never>>({})
}
