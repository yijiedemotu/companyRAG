/** 文档接口（契约 5.3） */

import { del, get, upload } from './client'
import type { DocumentListQuery } from '@/types/dto'
import type { ChunkOut, DocumentOut, Page } from '@/types/models'

/**
 * POST /kbs/{kb_id}/documents —— multipart `file`，201 DocumentOut。
 * 允许扩展名：md / markdown / txt / pdf / csv / json。
 */
export function uploadDocument(kbId: number, file: File): Promise<DocumentOut> {
  return upload<DocumentOut>(`/kbs/${kbId}/documents`, file, 'file')
}

/** GET /kbs/{kb_id}/documents?page&size&status */
export function listDocuments(kbId: number, query: DocumentListQuery = {}): Promise<Page<DocumentOut>> {
  const { page = 1, size = 20, status } = query
  return get<Page<DocumentOut>>(`/kbs/${kbId}/documents`, { page, size, status: status || undefined })
}

/** GET /documents/{doc_id} */
export function getDocument(docId: number): Promise<DocumentOut> {
  return get<DocumentOut>(`/documents/${docId}`)
}

/** DELETE /documents/{doc_id} —— 204 软删 + 清向量 + 清 chunks + 重建 BM25 */
export function deleteDocument(docId: number): Promise<void> {
  return del(`/documents/${docId}`)
}

/** GET /documents/{doc_id}/chunks?page&size —— chunk 预览抽屉 */
export function listChunks(docId: number, page = 1, size = 10): Promise<Page<ChunkOut>> {
  return get<Page<ChunkOut>>(`/documents/${docId}/chunks`, { page, size })
}
