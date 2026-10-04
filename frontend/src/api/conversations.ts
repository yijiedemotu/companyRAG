/** 会话与消息接口（契约 5.7） */

import { del, get, post } from './client'
import type { ConversationCreateRequest, ConversationListQuery, FeedbackRequest, MessageListQuery } from '@/types/dto'
import type { ConversationOut, FeedbackOut, MessageOut, Page } from '@/types/models'

/** GET /conversations?page&size&kb_id */
export function listConversations(query: ConversationListQuery = {}): Promise<Page<ConversationOut>> {
  const { page = 1, size = 20, kb_id } = query
  return get<Page<ConversationOut>>('/conversations', { page, size, kb_id })
}

/** POST /conversations —— 201 ConversationOut */
export function createConversation(payload: ConversationCreateRequest = {}): Promise<ConversationOut> {
  return post<ConversationOut>('/conversations', payload)
}

/** GET /conversations/{id} */
export function getConversation(conversationId: number): Promise<ConversationOut> {
  return get<ConversationOut>(`/conversations/${conversationId}`)
}

/** GET /conversations/{id}/messages?page&size —— 含 citations */
export function listMessages(conversationId: number, query: MessageListQuery = {}): Promise<Page<MessageOut>> {
  const { page = 1, size = 50 } = query
  return get<Page<MessageOut>>(`/conversations/${conversationId}/messages`, { page, size })
}

/** DELETE /conversations/{id} —— 204 软删 */
export function deleteConversation(conversationId: number): Promise<void> {
  return del(`/conversations/${conversationId}`)
}

/** POST /messages/{message_id}/feedback —— 1 有用 / -1 没用 / 0 中立 */
export function submitFeedback(messageId: number, payload: FeedbackRequest): Promise<FeedbackOut> {
  return post<FeedbackOut>(`/messages/${messageId}/feedback`, payload)
}
