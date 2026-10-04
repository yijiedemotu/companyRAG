/**
 * 与后端契约一一对应的请求 DTO（出参方向：前端 → 后端）。
 * 依据：docs/01-数据库与接口契约.md 第五节「REST 接口清单」。
 * 字段名必须逐字一致，不得改名。
 */

import type { ChatMode, EvalMode, FusionMode, SearchMode } from './models'

/* ------------------------------------------------------------------ 5.1 认证 */

export interface RegisterRequest {
  username: string
  password: string
  display_name?: string | null
}

export interface LoginRequest {
  username: string
  password: string
}

/* ------------------------------------------------------------------ 5.2 知识库 */

export interface KBCreateRequest {
  name: string
  description?: string | null
  chunk_size?: number
  chunk_overlap?: number
}

export interface KBUpdateRequest {
  name?: string
  description?: string | null
  chunk_size?: number
  chunk_overlap?: number
  is_active?: boolean
}

export interface KBListQuery {
  page?: number
  size?: number
  keyword?: string
}

/* ------------------------------------------------------------------ 5.3 文档 */

export interface DocumentListQuery {
  page?: number
  size?: number
  /** 按状态过滤：UPLOADED/PARSING/CHUNKING/EMBEDDING/READY/FAILED */
  status?: string
}

/* ------------------------------------------------------------------ 5.4 检索调试 */

export interface SearchRequest {
  query: string
  /** vector | bm25 | hybrid | hybrid_rerank */
  mode?: SearchMode
  top_k?: number
  fetch_k?: number
  /** rrf | weighted，仅 hybrid/hybrid_rerank 生效 */
  fusion?: FusionMode
  /** 仅 weighted 生效，向量权重 */
  alpha?: number
  rerank?: boolean
  use_autocut?: boolean
  /** null = 用后端配置默认值 */
  vector_threshold?: number | null
  /** null = 用后端配置默认值 */
  keyword_threshold?: number | null
  include_debug?: boolean
}

/* ------------------------------------------------------------------ 5.5 对话 */

export interface ChatRequest {
  /** 1..4000 字符 */
  question: string
  /** 可空 = 全局检索 */
  kb_id?: number | null
  /** 可空 = 新建会话 */
  conversation_id?: number | null
  /** rag | agent */
  mode?: ChatMode
  top_k?: number
  use_rerank?: boolean
  use_memory?: boolean
}

/* ------------------------------------------------------------------ 5.7 会话 */

export interface ConversationCreateRequest {
  kb_id?: number | null
  title?: string | null
}

export interface ConversationListQuery {
  page?: number
  size?: number
  kb_id?: number
}

export interface MessageListQuery {
  page?: number
  size?: number
}

export interface FeedbackRequest {
  /** 1 有用 / -1 没用 / 0 中立 */
  rating: number
  comment?: string | null
}

/* ------------------------------------------------------------------ 5.8 评测 */

export interface EvalCaseInput {
  question: string
  ground_truth?: string | null
  expected_doc?: string | null
  expected_sections?: string[] | null
  tags?: string[] | null
}

export interface DatasetCreateRequest {
  name: string
  description?: string | null
  cases?: EvalCaseInput[]
}

export interface EvalRunCreateRequest {
  dataset_id: number
  name?: string
  /** vector | bm25 | hybrid | hybrid_rerank | agent */
  mode: EvalMode
  top_k?: number
  use_rerank?: boolean
  use_llm_judge?: boolean
}

export interface EvalDatasetListQuery {
  page?: number
  size?: number
}

export interface EvalRunListQuery {
  page?: number
  size?: number
  dataset_id?: number
}

export interface EvalResultListQuery {
  page?: number
  size?: number
}

export interface EvalAblationQuery {
  dataset_id: number
  top_k?: number
}

/* ------------------------------------------------------------------ 5.9 可观测 */

export interface ObsStatsQuery {
  /** 统计窗口小时数，默认 24 */
  hours?: number
}

export interface ObsTraceListQuery {
  page?: number
  size?: number
  /** ok | error */
  status?: string
  mode?: string
  hours?: number
}

export interface ObsQualityQuery {
  hours?: number
}
