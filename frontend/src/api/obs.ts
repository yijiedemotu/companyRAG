/**
 * 可观测接口（契约 5.9 / 5.10）。
 *
 * 字段名已与后端 `knowflow/schemas/observability.py` 的 `ObsStatsOut` / `QualityOut` 对齐：
 * `requests` / `tokens.{prompt,completion,total}` / `cost_usd` / `cost_cny` /
 * `latency.{p50,p95,p99,max,avg}` / `by_mode[]` / `by_model[]` / `timeline[]`。
 *
 * 唯一保留的容错：`timeline` / `by_mode` / `by_model` 若因后端版本差异缺失，
 * 归一化成空数组，保证页面图表为空而不是直接崩。
 */

import { get } from './client'
import type { ObsQualityQuery, ObsStatsQuery, ObsTraceListQuery } from '@/types/dto'
import type {
  ByModeOut,
  ByModelOut,
  MessageOut,
  ObsQualityOut,
  ObsStatsOut,
  Page,
  TimelinePointOut,
  TraceDetailOut,
  TraceOut,
  TraceSpanOut,
} from '@/types/models'

/* ------------------------------------------------------------------ 归一化工具 */

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function num(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function str(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function normalizeByMode(value: unknown): ByModeOut[] {
  return asArray(value)
    .filter(isRecord)
    .map((raw) => ({
      mode: str(raw.mode),
      requests: num(raw.requests),
      tokens: num(raw.tokens),
      cost_usd: num(raw.cost_usd),
      avg_latency_ms: num(raw.avg_latency_ms),
      refusal_rate: num(raw.refusal_rate),
    }))
}

function normalizeByModel(value: unknown): ByModelOut[] {
  return asArray(value)
    .filter(isRecord)
    .map((raw) => ({
      model: str(raw.model),
      requests: num(raw.requests),
      prompt_tokens: num(raw.prompt_tokens),
      completion_tokens: num(raw.completion_tokens),
      cost_usd: num(raw.cost_usd),
      avg_latency_ms: num(raw.avg_latency_ms),
    }))
}

function normalizeTimeline(value: unknown): TimelinePointOut[] {
  return asArray(value)
    .filter(isRecord)
    .map((raw) => ({
      bucket: typeof raw.bucket === 'string' ? raw.bucket : null,
      requests: num(raw.requests),
      tokens: num(raw.tokens),
      cost_usd: num(raw.cost_usd),
      avg_latency_ms: num(raw.avg_latency_ms),
      refusal_rate: num(raw.refusal_rate),
    }))
}

/** 把后端的 ObsStatsOut 归一化成前端类型（字段名一致，只做缺失兜底） */
export function normalizeStats(raw: unknown): ObsStatsOut {
  const source = isRecord(raw) ? raw : {}
  const tokens = isRecord(source.tokens) ? source.tokens : {}
  const latency = isRecord(source.latency) ? source.latency : {}

  const prompt = num(tokens.prompt)
  const completion = num(tokens.completion)

  return {
    hours: num(source.hours, 24),
    requests: num(source.requests),
    tokens: {
      prompt,
      completion,
      total: num(tokens.total, prompt + completion),
    },
    cost_usd: num(source.cost_usd),
    cost_cny: num(source.cost_cny),
    latency: {
      p50: num(latency.p50),
      p95: num(latency.p95),
      p99: num(latency.p99),
      max: num(latency.max),
      avg: num(latency.avg),
    },
    by_mode: normalizeByMode(source.by_mode),
    by_model: normalizeByModel(source.by_model),
    timeline: normalizeTimeline(source.timeline),
  }
}

/* ------------------------------------------------------------------ 接口 */

/** GET /obs/stats?hours=24 */
export async function getStats(query: ObsStatsQuery = {}): Promise<ObsStatsOut> {
  const { hours = 24 } = query
  const raw = await get<unknown>('/obs/stats', { hours })
  return normalizeStats(raw)
}

/** GET /obs/quality?hours=24 */
export function getQuality(query: ObsQualityQuery = {}): Promise<ObsQualityOut> {
  const { hours = 24 } = query
  return get<ObsQualityOut>('/obs/quality', { hours })
}

/** GET /obs/traces?page&size&status&mode&hours */
export function listTraces(query: ObsTraceListQuery = {}): Promise<Page<TraceOut>> {
  const { page = 1, size = 20, status, mode, hours } = query
  return get<Page<TraceOut>>('/obs/traces', {
    page,
    size,
    status: status || undefined,
    mode: mode || undefined,
    hours,
  })
}

/**
 * GET /obs/traces/{trace_id} —— `{trace, spans, messages}`。
 * `spans` 后端已按 `seq` 升序返回，这里再排一次只为了保证瀑布图行序稳定。
 */
export async function getTraceDetail(traceId: string): Promise<TraceDetailOut> {
  const raw = await get<{ trace?: TraceOut; spans?: TraceSpanOut[]; messages?: MessageOut[] }>(
    `/obs/traces/${encodeURIComponent(traceId)}`,
  )

  const spans = Array.isArray(raw.spans) ? [...raw.spans].sort((a, b) => a.seq - b.seq) : []

  return {
    trace: raw.trace ?? ({} as TraceOut),
    spans,
    // 后端把 messages 定义为 `list[dict]`（含角色与内容摘要），按 MessageOut 消费
    messages: Array.isArray(raw.messages) ? (raw.messages as MessageOut[]) : [],
  }
}
