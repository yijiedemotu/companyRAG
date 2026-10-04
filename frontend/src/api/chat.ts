/** 对话接口（契约 5.5）—— 非流式在 chat.ts，流式在 sse.ts */

import { post } from './client'
import type { ChatRequest } from '@/types/dto'
import type { ChatResponse } from '@/types/models'

/** POST /chat —— 200 ChatResponse（一次性返回，非流式） */
export function chat(payload: ChatRequest): Promise<ChatResponse> {
  return post<ChatResponse>('/chat', payload)
}
