/**
 * 格式化工具（时间统一用 dayjs）。
 * 后端时间一律是 UTF 带上 Z 的 ISO8601，展示层转本地时区。
 */

import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import utc from 'dayjs/plugin/utc'
import 'dayjs/locale/zh-cn'

dayjs.extend(utc)
dayjs.extend(relativeTime)
dayjs.locale('zh-cn')

/** 后端可能返回秒级时间戳或字符串，统一转成 dayjs 对象 */
function toDayjs(value: string | number | Date | null | undefined): dayjs.Dayjs | null {
  if (value === null || value === undefined || value === '') return null
  const parsed = dayjs(value)
  return parsed.isValid() ? parsed : null
}

/** `2026-01-01 08:00:00` */
export function formatDateTime(value: string | number | Date | null | undefined): string {
  const time = toDayjs(value)
  return time ? time.format('YYYY-MM-DD HH:mm:ss') : '-'
}

/** `01-01 08:00`，列表里省地方 */
export function formatShortTime(value: string | number | Date | null | undefined): string {
  const time = toDayjs(value)
  return time ? time.format('MM-DD HH:mm') : '-'
}

/** 仅时分秒，例如 span 时间轴 */
export function formatClock(value: string | number | Date | null | undefined): string {
  const time = toDayjs(value)
  return time ? time.format('HH:mm:ss.SSS') : '-'
}

/** 相对时间：`3 分钟前` */
export function formatFromNow(value: string | number | Date | null | undefined): string {
  const time = toDayjs(value)
  return time ? time.fromNow() : '-'
}

/** 毫秒 → 人类可读：`412 ms` / `4.21 s` / `1m 12s` */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '-'
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`
  const minutes = Math.floor(ms / 60_000)
  const seconds = Math.round((ms % 60_000) / 1000)
  return `${minutes}m ${seconds}s`
}

/** 字节 → `1.2 MB` */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

/** 千分位整数 */
export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  return value.toLocaleString('zh-CN')
}

/** 指定小数位 */
export function formatFixed(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  return value.toFixed(digits)
}

/** 0~1 的比例 → `87.0%` */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  return `${(value * 100).toFixed(digits)}%`
}

/** 金额：USD 小数位多，CNY 两位 */
export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  if (value === 0) return '$0'
  if (value < 0.01) return `$${value.toFixed(6)}`
  return `$${value.toFixed(4)}`
}

export function formatCny(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  return `¥${value.toFixed(4)}`
}

/** USD → CNY（后端 USD_TO_CNY 默认 7.2；这里只用于展示兜底，不参与落库计算） */
export const FALLBACK_USD_TO_CNY = 7.2

export function usdToCny(usd: number | null | undefined, rate = FALLBACK_USD_TO_CNY): number | null {
  if (usd === null || usd === undefined || !Number.isFinite(usd)) return null
  return usd * rate
}

/** 文档状态 → Element Plus tag 类型 */
export function documentStatusTag(
  status: string,
): 'success' | 'warning' | 'danger' | 'info' | 'primary' {
  switch (status) {
    case 'READY':
      return 'success'
    case 'FAILED':
      return 'danger'
    case 'UPLOADED':
      return 'info'
    default:
      // PARSING / CHUNKING / EMBEDDING 都是进行中
      return 'warning'
  }
}

/** 文档状态 → 中文 */
export function documentStatusText(status: string): string {
  const map: Record<string, string> = {
    UPLOADED: '已上传',
    PARSING: '解析中',
    CHUNKING: '切分中',
    EMBEDDING: '向量化中',
    READY: '就绪',
    FAILED: '失败',
  }
  return map[status] ?? status
}

/** 是否处于处理中（需要轮询刷新） */
export function isDocumentProcessing(status: string): boolean {
  return ['UPLOADED', 'PARSING', 'CHUNKING', 'EMBEDDING'].includes(status)
}

/** 检索模式 → 中文 */
export function searchModeText(mode: string): string {
  const map: Record<string, string> = {
    vector: '纯向量',
    bm25: '纯关键词 BM25',
    hybrid: '混合检索',
    hybrid_rerank: '混合 + 重排',
    agent: 'Agent',
  }
  return map[mode] ?? mode
}

/** 数字安全取值 */
export function safeNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}
