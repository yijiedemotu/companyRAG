/** 检索调试接口（契约 5.4）—— **这个接口不调用大模型（不花钱）**，是调参主力 */

import { get, post } from './client'
import type { SearchRequest } from '@/types/dto'
import type { SearchResponse } from '@/types/models'
import type { HealthResponse } from '@/types/models'

/** POST /kbs/{kb_id}/search */
export function search(kbId: number, payload: SearchRequest): Promise<SearchResponse> {
  return post<SearchResponse>(`/kbs/${kbId}/search`, payload)
}

/**
 * GET `/health` —— **根路径**（不在 `/api/v1` 下）、无需鉴权。
 * 契约 5.10：用于如实展示后端的离线/降级模式（`offline` / `embedding_mode` / `bm25_doc_count`）。
 *
 * 注意：这里传绝对路径 `/health`。
 * axios 的 `baseURL` 只对相对路径生效，以 `/` 开头的 URL 会覆盖 baseURL，
 * 这样才会打到 `http://127.0.0.1:8000/health` 而不是 `http://127.0.0.1:8000/api/v1/health`。
 */
export function health(): Promise<HealthResponse> {
  return get<HealthResponse>('/health')
}
