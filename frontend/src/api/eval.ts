/** 评测接口（契约 5.8） */

import { get, post } from './client'
import type {
  DatasetCreateRequest,
  EvalAblationQuery,
  EvalDatasetListQuery,
  EvalResultListQuery,
  EvalRunCreateRequest,
  EvalRunListQuery,
} from '@/types/dto'
import type {
  AblationResponse,
  DatasetOut,
  EvalCaseOut,
  EvalCaseResultOut,
  EvalCompareResponse,
  EvalRunOut,
  Page,
} from '@/types/models'

/** GET /eval/datasets —— 注意：这个接口在契约里是**数组**而不是 Page */
export function listDatasets(query: EvalDatasetListQuery = {}): Promise<DatasetOut[]> {
  const { page = 1, size = 100 } = query
  return get<DatasetOut[]>('/eval/datasets', { page, size })
}

/** POST /eval/datasets —— `{name, description?, cases:[...]}` */
export function createDataset(payload: DatasetCreateRequest): Promise<DatasetOut> {
  return post<DatasetOut>('/eval/datasets', payload)
}

/** GET /eval/datasets/{id}/cases?page&size */
export function listDatasetCases(datasetId: number, page = 1, size = 20): Promise<Page<EvalCaseOut>> {
  return get<Page<EvalCaseOut>>(`/eval/datasets/${datasetId}/cases`, { page, size })
}

/** POST /eval/runs —— 201，后台执行 */
export function createRun(payload: EvalRunCreateRequest): Promise<EvalRunOut> {
  return post<EvalRunOut>('/eval/runs', payload)
}

/** GET /eval/runs?page&size&dataset_id */
export function listRuns(query: EvalRunListQuery = {}): Promise<Page<EvalRunOut>> {
  const { page = 1, size = 20, dataset_id } = query
  return get<Page<EvalRunOut>>('/eval/runs', { page, size, dataset_id })
}

/** GET /eval/runs/{id} —— 含 metrics_json */
export function getRun(runId: number): Promise<EvalRunOut> {
  return get<EvalRunOut>(`/eval/runs/${runId}`)
}

/** GET /eval/runs/{id}/results?page&size —— 逐条结果 */
export function listRunResults(runId: number, query: EvalResultListQuery = {}): Promise<Page<EvalCaseResultOut>> {
  const { page = 1, size = 20 } = query
  return get<Page<EvalCaseResultOut>>(`/eval/runs/${runId}/results`, { page, size })
}

/** GET /eval/ablation?dataset_id&top_k —— 消融对比（纯向量/纯BM25/混合/混合+重排） */
export function ablation(query: EvalAblationQuery): Promise<AblationResponse> {
  const { dataset_id, top_k = 5 } = query
  return get<AblationResponse>('/eval/ablation', { dataset_id, top_k })
}

/** GET /eval/compare?run_a&run_b —— 两个 run 对比 + 95% 置信区间与显著性 */
export function compare(runA: number, runB: number): Promise<EvalCompareResponse> {
  return get<EvalCompareResponse>('/eval/compare', { run_a: runA, run_b: runB })
}
