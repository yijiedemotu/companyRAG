/**
 * 与后端契约一一对应的响应模型（入参方向：后端 → 前端）。
 * 依据：docs/01-数据库与接口契约.md。
 *
 * 约定：
 * - 时间字段是 ISO8601 带 Z 的 UTC 字符串，展示层用 dayjs 转本地时区；
 * - 金额单位 USD（后端 USD_TO_CNY 默认 7.2，前端仅做展示换算，不参与计算落库）。
 */

/* ------------------------------------------------------------------ 通用信封 */

/** 分页响应统一结构 */
export interface Page<T> {
  items: T[]
  total: number
  page: number
  size: number
  pages: number
}

/** 统一错误响应体的 error 子对象 */
export interface ApiErrorBody {
  code: string
  message: string
  detail: unknown
  request_id: string | null
  timestamp: string
}

/** 统一错误响应体 */
export interface ApiErrorEnvelope {
  error: ApiErrorBody
}

/* ------------------------------------------------------------------ 5.1 认证 */

export type UserRole = 'admin' | 'user'

export interface UserOut {
  id: number
  username: string
  display_name: string | null
  role: UserRole
  is_active: boolean
  created_at: string
}

export interface TokenOut {
  access_token: string
  token_type: 'bearer'
  expires_in: number
  user: UserOut
}

/* ------------------------------------------------------------------ 5.2 知识库 */

export type EmbeddingProvider = 'local' | 'hash' | 'api'

export interface KBOut {
  id: number
  name: string
  description: string | null
  owner_id: number
  embedding_provider: EmbeddingProvider
  embedding_model: string
  embedding_dim: number
  chunk_size: number
  chunk_overlap: number
  is_active: boolean
  doc_count: number
  chunk_count: number
  created_at: string
}

export interface KBStatsOut {
  kb_id: number
  doc_count: number
  ready_doc_count: number
  chunk_count: number
  vector_count: number
  bm25_doc_count: number
  total_chars: number
  total_tokens: number
  /** chunk_count == vector_count；false 表示向量库与关系库不一致，前端要显红点 */
  consistent: boolean
}

/* ------------------------------------------------------------------ 5.3 文档 */

export type DocumentStatus =
  | 'UPLOADED'
  | 'PARSING'
  | 'CHUNKING'
  | 'EMBEDDING'
  | 'READY'
  | 'FAILED'

export interface DocumentOut {
  id: number
  kb_id: number
  filename: string
  ext: string
  size_bytes: number
  status: DocumentStatus
  parser: string | null
  page_count: number | null
  char_count: number
  chunk_count: number
  token_count: number
  ingest_ms: number | null
  error_code: string | null
  error_message: string | null
  created_at: string
}

export interface ChunkOut {
  id: number
  doc_id: number
  chunk_index: number
  content: string
  char_count: number
  token_count: number
  page_no: number | null
  section_path: string | null
  vector_id: string
}

/* ------------------------------------------------------------------ 5.4 检索调试 */

export type SearchMode = 'vector' | 'bm25' | 'hybrid' | 'hybrid_rerank'
export type FusionMode = 'rrf' | 'weighted'

export interface SearchHit {
  rank: number
  chunk_id: number
  doc_id: number
  doc_name: string
  page_no: number | null
  section_path: string | null
  content: string
  snippet: string
  /** 融合后的最终分数 */
  score: number
  vector_score: number | null
  bm25_score: number | null
  vector_rank: number | null
  bm25_rank: number | null
  rerank_score: number | null
}

/** 相关性闸门判定结果：防幻觉的最后一道 */
export interface SearchGate {
  passed: boolean
  reason: string
  vector_threshold: number
  keyword_threshold: number
  best_vector_score: number | null
  best_keyword_coverage: number | null
}

export interface SearchDebugInfo {
  vector_candidates: number
  bm25_candidates: number
  fused: number
  after_autocut: number
  dropped_by_gate: number
}

export interface SearchResponse {
  query: string
  mode: SearchMode
  latency_ms: number
  embedding_ms: number
  vector_ms: number
  bm25_ms: number
  rerank_ms: number
  hits: SearchHit[]
  gate: SearchGate
  debug: SearchDebugInfo | null
}

/* ------------------------------------------------------------------ 5.5 对话 */

export type ChatMode = 'rag' | 'agent'

/** 引用来源（sources 事件与 ChatResponse.sources 同构） */
export interface Source {
  rank: number
  chunk_id: number
  doc_id: number
  doc_name: string
  page_no: number | null
  section_path: string | null
  score: number
  snippet: string
}

/** 候选片段相关性判定（grade 节点产物） */
export interface GradingItem {
  chunk_id: number
  relevant: boolean
  reason: string
}

export interface ChatUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ChatResponse {
  conversation_id: number
  message_id: number
  trace_id: string
  answer: string
  sources: Source[]
  refusal: boolean
  retrieval_rounds: number
  reflect_passed: boolean
  grading: GradingItem[]
  rewritten_queries: string[]
  usage: ChatUsage
  cost_usd: number
  latency_ms: number
  model: string
  offline: boolean
}

/* ------------------------------------------------------------------ 5.7 会话 */

export interface ConversationOut {
  id: number
  kb_id: number | null
  user_id: number
  title: string
  mode: ChatMode
  message_count: number
  created_at: string
  updated_at: string
}

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool'

/** 消息内嵌引用（对应 message_citations 表） */
export interface CitationOut {
  rank: number
  chunk_id: number | null
  doc_id: number | null
  score: number
  snippet: string
  doc_name?: string | null
  page_no?: number | null
  section_path?: string | null
}

export interface MessageOut {
  id: number
  conversation_id: number
  role: MessageRole
  content: string
  mode: ChatMode | null
  model: string | null
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
  latency_ms: number | null
  refusal: boolean
  retrieval_rounds: number
  trace_id: string | null
  citations: CitationOut[]
  created_at: string
}

export interface FeedbackOut {
  id: number
  message_id: number
  user_id: number
  rating: number
  comment: string | null
  created_at: string
}

/* ------------------------------------------------------------------ 5.8 评测 */

export type EvalMode = 'vector' | 'bm25' | 'hybrid' | 'hybrid_rerank' | 'agent'

export interface DatasetOut {
  id: number
  name: string
  description: string | null
  case_count: number
}

export interface EvalCaseOut {
  id: number
  dataset_id: number
  question: string
  ground_truth: string | null
  expected_doc: string | null
  expected_sections: string[]
  tags: string[]
  created_at: string
}

export type EvalRunStatus = 'pending' | 'running' | 'done' | 'failed'

/**
 * 聚合指标（metrics_json / EvalMetrics）。
 *
 * 除 `case_count` 外**全部可为 null**，语义是「没算」而不是「算出来是 0」：
 * LLM 判官关闭时 `faithfulness` / `answer_relevance` 无从计算，给 0 会被误读成 0 分。
 */
export interface EvalMetrics {
  case_count: number
  hit_rate: number | null
  recall_at_k: number | null
  mrr: number | null
  ndcg: number | null
  faithfulness: number | null
  answer_relevance: number | null
  avg_latency_ms: number | null
  p95_latency_ms: number | null
  refusal_rate: number | null
}

export interface EvalRunOut {
  id: number
  dataset_id: number
  name: string
  mode: EvalMode
  top_k: number
  config_json: Record<string, unknown> | null
  status: EvalRunStatus
  case_count: number
  passed_count: number
  metrics_json: EvalMetrics | null
  error: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
}

export interface EvalCaseResultOut {
  id: number
  run_id: number
  case_id: number
  question: string
  /** 命中列表快照（含各阶段分数与排名）；可能是 null */
  retrieved_json: Array<Record<string, unknown>> | null
  answer: string | null
  hit: boolean
  recall_at_k: number | null
  mrr: number | null
  ndcg: number | null
  faithfulness: number | null
  answer_relevance: number | null
  latency_ms: number | null
  error: string | null
  created_at: string
}

/**
 * 消融实验的一个分组。
 *
 * 字段是**扁平**的（指标直接在分组上），不是 `group.metrics.recall_at_k`。
 * `delta_vs_baseline` 是带符号差值（分组指标 − 基线指标），正数表示更好；
 * 延迟类指标方向相反，展示时要注意。
 */
export interface AblationGroup {
  mode: EvalMode | string
  /** 展示名称，如「hybrid + rerank」 */
  label: string
  top_k: number
  /** 对应运行 ID；未跑过为 null */
  run_id: number | null
  case_count: number
  hit_rate: number | null
  recall_at_k: number | null
  mrr: number | null
  ndcg: number | null
  avg_latency_ms: number | null
  delta_vs_baseline: Record<string, number>
}

export interface AblationResponse {
  dataset_id: number
  /** 本次按哪个 top_k 计算（回显） */
  top_k: number
  /** 作为基线的模式；其余分组都与它比较 */
  baseline_mode: EvalMode | null
  groups: AblationGroup[]
}

export interface EvalCompareResponse {
  a: EvalRunOut
  b: EvalRunOut
  /** B − A 的各指标差值，键是指标名（如 recall_at_k） */
  delta: Record<string, number>
  /** 差值 95% 置信区间：{low, high}；区间不跨 0 才算显著 */
  ci95: { low: number; high: number }
  significant: boolean
}

/* ------------------------------------------------------------------ 5.9 可观测 */

export interface TraceOut {
  id: number
  trace_id: string
  request_id: string | null
  user_id: number | null
  conversation_id: number | null
  kb_id: number | null
  /** chat / search / ingest / eval */
  name: string
  mode: string | null
  status: 'ok' | 'error'
  latency_ms: number
  retrieval_ms: number
  rerank_ms: number
  generate_ms: number
  llm_calls: number
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
  retrieval_rounds: number
  source_count: number
  refusal: boolean
  error: string | null
  created_at: string
}

/** span 类型：决定瀑布图配色 */
export type SpanType = 'node' | 'llm' | 'retrieval' | 'db' | 'tool'

export interface TraceSpanOut {
  id: number
  trace_id: string
  seq: number
  name: string
  span_type: SpanType
  /** 相对 trace 起点的偏移（毫秒）—— 瀑布图横轴起点 */
  start_offset_ms: number
  /** 持续时长（毫秒）—— 瀑布图横轴长度 */
  duration_ms: number
  status: 'ok' | 'error'
  input_json: unknown
  output_json: unknown
  error: string | null
  created_at: string
}

export interface TraceDetailOut {
  trace: TraceOut
  spans: TraceSpanOut[]
  messages: MessageOut[]
}

/** 按 mode 分组的统计（对应后端 `ByModeOut`） */
export interface ByModeOut {
  mode: string
  requests: number
  tokens: number
  cost_usd: number
  avg_latency_ms: number
  refusal_rate: number
}

/** 按 model 分组的统计（对应后端 `ByModelOut`）。用来回答「钱花在哪个模型上」 */
export interface ByModelOut {
  model: string
  requests: number
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
  avg_latency_ms: number
}

/** 时间序列上的一个点（对应后端 `TimelinePointOut`）。`bucket` 是时间桶起点（UTC ISO8601 带 Z） */
export interface TimelinePointOut {
  bucket: string | null
  requests: number
  tokens: number
  cost_usd: number
  avg_latency_ms: number
  refusal_rate: number
}

/** token 用量汇总（对应后端 `TokenStatsOut`） */
export interface TokenStatsOut {
  prompt: number
  completion: number
  total: number
}

/** 延迟分位（对应后端 `LatencyOut`，单位毫秒） */
export interface LatencyOut {
  p50: number
  p95: number
  p99: number
  max: number
  avg: number
}

/**
 * 观测统计（对应后端 `ObsStatsOut`）。
 *
 * 注意这是**嵌套**结构而不是扁平结构：
 * `requests` / `tokens.{prompt,completion,total}` / `cost_usd` / `cost_cny` /
 * `latency.{p50,p95,p99,max,avg}` / `by_mode[]` / `by_model[]` / `timeline[]`。
 */
export interface ObsStatsOut {
  hours: number
  requests: number
  tokens: TokenStatsOut
  cost_usd: number
  cost_cny: number
  latency: LatencyOut
  by_mode: ByModeOut[]
  by_model: ByModelOut[]
  timeline: TimelinePointOut[]
}

export interface ObsQualityOut {
  hours: number
  total: number
  refusal_rate: number
  avg_source_count: number
  zero_source_rate: number
  avg_retrieval_rounds: number
  reflect_pass_rate: number
}

/* ------------------------------------------------------------------ 5.10 运维 */

export interface HealthDb {
  ok: boolean
  dialect: string
  version: string
}

export interface HealthResponse {
  status: string
  version: string
  uptime_s: number
  offline: boolean
  llm_mode: string
  embedding_mode: string
  vector_backend: string
  vector_count: number
  bm25_doc_count: number
  db: HealthDb
  pool: Record<string, unknown>
  warnings: string[]
}
