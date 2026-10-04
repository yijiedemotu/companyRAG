/** 知识库接口 `/api/v1/kbs`（契约 5.2） */

import { del, get, patch, post } from './client'
import type { KBCreateRequest, KBListQuery, KBUpdateRequest } from '@/types/dto'
import type { KBOut, KBStatsOut, Page } from '@/types/models'

/** GET /kbs?page&size&keyword */
export function listKBs(query: KBListQuery = {}): Promise<Page<KBOut>> {
  const { page = 1, size = 20, keyword } = query
  return get<Page<KBOut>>('/kbs', { page, size, keyword: keyword || undefined })
}

/** POST /kbs —— 201 KBOut */
export function createKB(payload: KBCreateRequest): Promise<KBOut> {
  return post<KBOut>('/kbs', payload)
}

/** GET /kbs/{kb_id} */
export function getKB(kbId: number): Promise<KBOut> {
  return get<KBOut>(`/kbs/${kbId}`)
}

/** PATCH /kbs/{kb_id} */
export function updateKB(kbId: number, payload: KBUpdateRequest): Promise<KBOut> {
  return patch<KBOut>(`/kbs/${kbId}`, payload)
}

/** DELETE /kbs/{kb_id} —— 204 软删 + 清向量集合 + 清 BM25 */
export function deleteKB(kbId: number): Promise<void> {
  return del(`/kbs/${kbId}`)
}

/** GET /kbs/{kb_id}/stats —— 含 consistent，不一致时前端要显红点 */
export function getKBStats(kbId: number): Promise<KBStatsOut> {
  return get<KBStatsOut>(`/kbs/${kbId}/stats`)
}
