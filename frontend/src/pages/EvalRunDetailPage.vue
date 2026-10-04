<script setup lang="ts">
/**
 * 评测运行详情页 `/eval/runs/:id`：指标卡 + 逐条结果表 + 与另一个 run 的对比。
 *
 * 契约 5.8：`GET /eval/runs/{id}`（含 metrics_json）、`GET /eval/runs/{id}/results`、
 * `GET /eval/compare?run_a&run_b` → `{a, b, delta, ci95:{low,high}, significant}`。
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

import { describeError } from '@/api/client'
import * as evalApi from '@/api/eval'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LoadingBlock from '@/components/LoadingBlock.vue'
import PageBar from '@/components/PageBar.vue'
import StatCard from '@/components/StatCard.vue'
import type { EvalCaseResultOut, EvalCompareResponse, EvalMetrics, EvalRunOut } from '@/types/models'
import { formatDateTime, formatDuration, formatFixed, formatInt, formatPercent, searchModeText } from '@/utils/format'

const props = defineProps<{ id: string }>()

const route = useRoute()
const router = useRouter()

const runId = computed<number>(() => Number(props.id))
const run = ref<EvalRunOut | null>(null)
const runError = ref('')
const loadingRun = ref(false)

const results = ref<EvalCaseResultOut[]>([])
const resultsTotal = ref(0)
const resultsPage = ref(1)
const resultsSize = ref(20)
const resultsPages = ref(0)
const loadingResults = ref(false)
const resultsError = ref('')

/** 对比 */
const compareRunId = ref<number | null>(null)
const compareCandidates = ref<EvalRunOut[]>([])
const comparing = ref(false)
const comparison = ref<EvalCompareResponse | null>(null)

const metrics = computed<Partial<EvalMetrics>>(() => run.value?.metrics_json ?? {})

const isActive = computed<boolean>(() => run.value?.status === 'pending' || run.value?.status === 'running')

let pollTimer: number | null = null

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function syncPolling(): void {
  if (isActive.value && pollTimer === null) {
    pollTimer = window.setInterval(() => {
      void loadRun(true)
      void loadResults(true)
    }, 4000)
  } else if (!isActive.value) {
    stopPolling()
  }
}

async function loadRun(silent = false): Promise<void> {
  if (!silent) loadingRun.value = true
  runError.value = ''
  try {
    run.value = await evalApi.getRun(runId.value)
    syncPolling()
  } catch (error) {
    if (!silent) runError.value = describeError(error)
  } finally {
    loadingRun.value = false
  }
}

async function loadResults(silent = false): Promise<void> {
  if (!silent) loadingResults.value = true
  resultsError.value = ''
  try {
    const result = await evalApi.listRunResults(runId.value, { page: resultsPage.value, size: resultsSize.value })
    results.value = result.items
    resultsTotal.value = result.total
    resultsPages.value = result.pages
  } catch (error) {
    if (!silent) resultsError.value = describeError(error)
  } finally {
    loadingResults.value = false
  }
}

function handlePageChange(payload: { page: number; size: number }): void {
  resultsPage.value = payload.page
  resultsSize.value = payload.size
  void loadResults()
}

/** 可选对比对象：同一数据集下的其它 run */
async function loadCompareCandidates(): Promise<void> {
  if (!run.value) return
  try {
    const result = await evalApi.listRuns({ page: 1, size: 50, dataset_id: run.value.dataset_id })
    compareCandidates.value = result.items.filter((item) => item.id !== runId.value && item.status === 'done')
  } catch {
    compareCandidates.value = []
  }
}

async function handleCompare(): Promise<void> {
  if (compareRunId.value === null || !run.value) return
  comparing.value = true
  try {
    comparison.value = await evalApi.compare(compareRunId.value, runId.value)
  } catch (error) {
    ElMessage.error(describeError(error))
    comparison.value = null
  } finally {
    comparing.value = false
  }
}

/** 对比结果里的 delta 键（按契约 EvalMetrics 的字段名） */
const COMPARE_ROWS: Array<{ key: keyof EvalMetrics; label: string; percent?: boolean }> = [
  { key: 'hit_rate', label: 'hit_rate', percent: true },
  { key: 'recall_at_k', label: 'recall@k', percent: true },
  { key: 'mrr', label: 'MRR', percent: false },
  { key: 'ndcg', label: 'nDCG', percent: false },
  { key: 'faithfulness', label: 'faithfulness', percent: false },
  { key: 'answer_relevance', label: 'answer_relevance', percent: false },
  { key: 'avg_latency_ms', label: '平均延迟(ms)', percent: false },
  { key: 'p95_latency_ms', label: 'P95 延迟(ms)', percent: false },
  { key: 'refusal_rate', label: '拒答率', percent: true },
]

/** 指标值展示：null/undefined 都显示 ——（后端用 null 表示「没算」，不能显示成 0） */
function metricText(value: number | null | undefined, percent = false): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—'
  return percent ? formatPercent(value) : formatFixed(value, 3)
}

function deltaText(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—'
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(4)}`
}

function deltaClass(value: number | null | undefined, higherIsBetter: boolean): string {
  if (value === undefined || value === null || !Number.isFinite(value) || value === 0) return ''
  const good = higherIsBetter ? value > 0 : value < 0
  return good ? 'delta--good' : 'delta--bad'
}

/** 命中列表快照：契约里是 JSON，字段按检索命中结构容错读取 */
interface RetrievedRow {
  rank: number
  doc: string
  section: string
  score: string
}

function retrievedRows(raw: unknown): RetrievedRow[] {
  if (!Array.isArray(raw)) return []
  return raw.slice(0, 10).map((item, index) => {
    if (typeof item !== 'object' || item === null) {
      return { rank: index + 1, doc: String(item), section: '', score: '' }
    }
    const record = item as Record<string, unknown>
    const doc = record.doc_name ?? record.doc ?? record.filename ?? record.chunk_id ?? `#${index + 1}`
    const section = record.section_path ?? ''
    const score = record.score ?? record.rerank_score ?? record.vector_score ?? ''
    return {
      rank: typeof record.rank === 'number' ? record.rank : index + 1,
      doc: String(doc),
      section: typeof section === 'string' ? section : '',
      score: typeof score === 'number' ? score.toFixed(3) : String(score),
    }
  })
}

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'done') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'running') return 'warning'
  return 'info'
}

function statusText(status: string): string {
  const map: Record<string, string> = { pending: '排队中', running: '执行中', done: '已完成', failed: '失败' }
  return map[status] ?? status
}

watch(runId, () => {
  void loadRun()
  void loadResults()
})

onMounted(async () => {
  await loadRun()
  await loadResults()
  await loadCompareCandidates()
  // 从列表页点「对比」过来时，直接聚焦对比区
  if (route.query.compare === '1') ElMessage.info('选择另一个已完成的运行进行对比')
})

onBeforeUnmount(stopPolling)
</script>

<template>
  <div class="kf-page">
    <div class="kf-page-header">
      <div class="kf-row">
        <el-button link @click="router.push('/eval')">← 评测列表</el-button>
        <h2 class="kf-page-title">
          运行 #{{ runId }}
          <span v-if="run" class="kf-muted run-name">{{ run.name }}</span>
        </h2>
        <el-tag v-if="run" size="small" :type="statusTagType(run.status)" effect="plain">
          {{ statusText(run.status) }}
        </el-tag>
        <el-tag v-if="run" size="small" effect="plain">{{ searchModeText(run.mode) }}</el-tag>
      </div>
      <div class="kf-row">
        <el-button @click="loadRun(); loadResults()">刷新</el-button>
        <el-select v-model="compareRunId" placeholder="选择要对比的运行" clearable size="small" style="width: 220px">
          <el-option
            v-for="candidate in compareCandidates"
            :key="candidate.id"
            :label="`#${candidate.id} ${candidate.name}（${searchModeText(candidate.mode)}）`"
            :value="candidate.id"
          />
        </el-select>
        <el-button type="primary" :disabled="compareRunId === null" :loading="comparing" @click="handleCompare">
          对比
        </el-button>
      </div>
    </div>

    <ErrorBlock v-if="runError" :message="runError" @retry="loadRun()" />

    <LoadingBlock v-else-if="loadingRun && !run" height="200px" />

    <template v-else-if="run">
      <el-alert
        v-if="run.error"
        type="error"
        :closable="false"
        show-icon
        title="评测运行失败"
        :description="run.error"
      />

      <el-alert
        v-if="isActive"
        class="active-alert"
        type="warning"
        :closable="false"
        show-icon
        title="评测执行中"
        :description="`已处理 ${formatInt(resultsTotal)} / ${formatInt(run.case_count)} 条，页面每 4 秒自动刷新。`"
      />

      <!-- 指标卡：契约 EvalMetrics 的字段 -->
      <div class="metric-grid">
        <StatCard label="用例数" :value="formatInt(metrics.case_count ?? run.case_count)" icon="🗂️" />
        <StatCard label="命中率 hit_rate" :value="formatPercent(metrics.hit_rate)" icon="🎯" tone="primary" />
        <StatCard label="recall@k" :value="formatFixed(metrics.recall_at_k, 4)" icon="📈" />
        <StatCard label="MRR" :value="formatFixed(metrics.mrr, 4)" icon="🏅" />
        <StatCard label="nDCG" :value="formatFixed(metrics.ndcg, 4)" icon="📐" />
        <StatCard label="faithfulness" :value="formatFixed(metrics.faithfulness, 4)" icon="🧷" />
        <StatCard label="answer_relevance" :value="formatFixed(metrics.answer_relevance, 4)" icon="💬" />
        <StatCard label="拒答率 refusal_rate" :value="formatPercent(metrics.refusal_rate)" icon="🙅" tone="warning" />
        <StatCard label="平均延迟" :value="formatDuration(metrics.avg_latency_ms)" icon="⏱️" />
        <StatCard label="P95 延迟" :value="formatDuration(metrics.p95_latency_ms)" icon="🐢" tone="danger" />
      </div>

      <!-- 对比结果 -->
      <div v-if="comparison" class="kf-card comparison-card">
        <div class="kf-card-title">
          <h3>对比：A #{{ comparison.a?.id }} ↔ B #{{ comparison.b?.id }}（B 为本页运行）</h3>
          <el-tag :type="comparison.significant ? 'success' : 'info'" effect="dark">
            {{ comparison.significant ? '差异显著' : '差异不显著' }}
          </el-tag>
        </div>
        <div class="kf-card-body">
          <div class="ci-line">
            <span class="kf-muted">95% 置信区间（bootstrap）：</span>
            <span class="kf-mono">
              [{{ formatFixed(comparison.ci95?.low, 4) }}, {{ formatFixed(comparison.ci95?.high, 4) }}]
            </span>
            <span class="kf-muted ci-hint">区间跨 0 说明提升可能只是噪声——这一步就是为了避免自欺欺人。</span>
          </div>
          <el-table :data="COMPARE_ROWS" size="small">
            <el-table-column label="指标" width="180">
              <template #default="{ row }: { row: { key: keyof EvalMetrics; label: string; percent?: boolean } }">
                {{ row.label }}
              </template>
            </el-table-column>
            <el-table-column label="A" width="120" align="right">
              <template #default="{ row }: { row: { key: keyof EvalMetrics; label: string; percent?: boolean } }">
                {{ metricText(comparison.a?.metrics_json?.[row.key], row.percent) }}
              </template>
            </el-table-column>
            <el-table-column label="B（本页）" width="120" align="right">
              <template #default="{ row }: { row: { key: keyof EvalMetrics; label: string; percent?: boolean } }">
                {{ metricText(comparison.b?.metrics_json?.[row.key], row.percent) }}
              </template>
            </el-table-column>
            <el-table-column label="delta (B - A)" min-width="140" align="right">
              <template #default="{ row }: { row: { key: keyof EvalMetrics; label: string; percent?: boolean } }">
                <span
                  class="kf-mono"
                  :class="deltaClass(comparison.delta?.[row.key], !row.label.includes('延迟') && row.label !== '拒答率')"
                >
                  {{ deltaText(comparison.delta?.[row.key]) }}
                </span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>

      <!-- 逐条结果 -->
      <div class="kf-card">
        <div class="kf-card-title">
          <h3>逐条结果</h3>
          <span class="kf-muted">共 {{ formatInt(resultsTotal) }} 条</span>
        </div>

        <ErrorBlock v-if="resultsError" class="kf-card-body" :message="resultsError" @retry="loadResults()" />

        <LoadingBlock v-else-if="loadingResults && results.length === 0" height="180px" />

        <EmptyState v-else-if="results.length === 0" icon="📄" title="还没有逐条结果" description="运行完成后这里会列出每条用例的命中情况与各项指标。" />

        <template v-else>
          <el-table :data="results" size="small">
            <el-table-column type="expand">
              <template #default="{ row }: { row: EvalCaseResultOut }">
                <div class="result-expand">
                  <div class="kf-muted">问题</div>
                  <div class="result-expand__question">{{ row.question }}</div>

                  <div class="kf-muted">模型回答</div>
                  <pre class="kf-pre">{{ row.answer || '（无回答）' }}</pre>

                  <div class="kf-muted">命中列表快照（retrieved_json）</div>
                  <el-table v-if="retrievedRows(row.retrieved_json).length > 0" :data="retrievedRows(row.retrieved_json)" size="small">
                    <el-table-column prop="rank" label="#" width="50" />
                    <el-table-column prop="doc" label="文档" min-width="160" show-overflow-tooltip />
                    <el-table-column prop="section" label="小节" min-width="140" show-overflow-tooltip />
                    <el-table-column prop="score" label="分数" width="90" align="right" />
                  </el-table>
                  <pre v-else class="kf-pre">{{ JSON.stringify(row.retrieved_json, null, 2) }}</pre>

                  <div v-if="row.error" class="result-expand__error">错误：{{ row.error }}</div>
                </div>
              </template>
            </el-table-column>

            <el-table-column prop="case_id" label="case" width="70" />
            <el-table-column prop="question" label="问题" min-width="220" show-overflow-tooltip />
            <el-table-column label="命中" width="80">
              <template #default="{ row }: { row: EvalCaseResultOut }">
                <el-tag size="small" :type="row.hit ? 'success' : 'danger'" effect="plain">
                  {{ row.hit ? '命中' : '未命中' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="recall@k" width="100" align="right">
              <template #default="{ row }: { row: EvalCaseResultOut }">{{ formatFixed(row.recall_at_k, 3) }}</template>
            </el-table-column>
            <el-table-column label="MRR" width="80" align="right">
              <template #default="{ row }: { row: EvalCaseResultOut }">{{ formatFixed(row.mrr, 3) }}</template>
            </el-table-column>
            <el-table-column label="nDCG" width="80" align="right">
              <template #default="{ row }: { row: EvalCaseResultOut }">{{ formatFixed(row.ndcg, 3) }}</template>
            </el-table-column>
            <el-table-column label="faithfulness" width="110" align="right">
              <template #default="{ row }: { row: EvalCaseResultOut }">{{ formatFixed(row.faithfulness, 3) }}</template>
            </el-table-column>
            <el-table-column label="answer_relevance" width="130" align="right">
              <template #default="{ row }: { row: EvalCaseResultOut }">
                {{ formatFixed(row.answer_relevance, 3) }}
              </template>
            </el-table-column>
            <el-table-column label="延迟" width="100" align="right">
              <template #default="{ row }: { row: EvalCaseResultOut }">{{ formatDuration(row.latency_ms) }}</template>
            </el-table-column>
            <el-table-column label="错误" width="120">
              <template #default="{ row }: { row: EvalCaseResultOut }">
                <span v-if="row.error" class="result-error kf-truncate" :title="row.error">{{ row.error }}</span>
                <span v-else class="kf-muted">—</span>
              </template>
            </el-table-column>
          </el-table>

          <PageBar
            :page="resultsPage"
            :size="resultsSize"
            :total="resultsTotal"
            :pages="resultsPages"
            @change="handlePageChange"
          />
        </template>
      </div>

      <div class="kf-card raw-card">
        <div class="kf-card-title"><h3>元信息</h3></div>
        <div class="kf-card-body raw-body">
          <div><span class="kf-muted">数据集：</span>#{{ run.dataset_id }}</div>
          <div><span class="kf-muted">top_k：</span>{{ run.top_k }}</div>
          <div><span class="kf-muted">通过/总数：</span>{{ formatInt(run.passed_count) }} / {{ formatInt(run.case_count) }}</div>
          <div><span class="kf-muted">开始：</span>{{ formatDateTime(run.started_at) }}</div>
          <div><span class="kf-muted">结束：</span>{{ formatDateTime(run.finished_at) }}</div>
        </div>
        <div class="kf-card-body">
          <div class="kf-muted">config_json（本次运行的完整参数快照）</div>
          <pre class="kf-pre">{{ JSON.stringify(run.config_json, null, 2) }}</pre>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.run-name {
  font-size: 14px;
  font-weight: 400;
  margin-left: 8px;
}

.active-alert {
  margin-bottom: 16px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.comparison-card {
  margin-bottom: 16px;
}

.ci-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 13px;
}

.ci-hint {
  font-size: 12px;
}

.delta--good {
  color: var(--kf-success);
  font-weight: 600;
}

.delta--bad {
  color: var(--kf-danger);
  font-weight: 600;
}

.result-expand {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 14px;
  font-size: 12px;
}

.result-expand__question {
  font-size: 13px;
  color: var(--kf-text-primary);
  margin-bottom: 4px;
}

.result-expand__error {
  color: var(--kf-danger);
}

.result-error {
  color: var(--kf-danger);
  font-size: 12px;
  display: inline-block;
  max-width: 110px;
}

.raw-card {
  margin-top: 16px;
}

.raw-body {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  font-size: 13px;
  padding-bottom: 0;
}
</style>
