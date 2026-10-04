<script setup lang="ts">
/**
 * 评测页 `/eval`：数据集 + 运行列表 + 消融对比。
 *
 * 契约 5.8：
 * - `GET /eval/datasets`（注意返回的是**数组**，不是 Page）
 * - `POST /eval/datasets`（`{name, description?, cases:[...]}`）
 * - `GET/POST /eval/runs`、`GET /eval/runs/{id}`、`GET /eval/runs/{id}/results`
 * - `GET /eval/ablation?dataset_id&top_k` → 纯向量 / 纯 BM25 / 混合 / 混合+重排 四组对比
 */
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'

import { describeError } from '@/api/client'
import * as evalApi from '@/api/eval'
import BarChart from '@/components/BarChart.vue'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LoadingBlock from '@/components/LoadingBlock.vue'
import PageBar from '@/components/PageBar.vue'
import StatCard from '@/components/StatCard.vue'
import type { AblationGroup, DatasetOut, EvalMode, EvalRunOut } from '@/types/models'
import { formatDateTime, formatDuration, formatFixed, formatInt, searchModeText } from '@/utils/format'

const router = useRouter()

/* ---------------------------------------------------------------- 数据集 */

const datasets = ref<DatasetOut[]>([])
const datasetsLoading = ref(false)
const datasetsError = ref('')
const selectedDatasetId = ref<number | null>(null)

/* ---------------------------------------------------------------- 运行列表 */

const runs = ref<EvalRunOut[]>([])
const runsTotal = ref(0)
const runsPage = ref(1)
const runsSize = ref(10)
const runsPages = ref(0)
const runsLoading = ref(false)
const runsError = ref('')

/* ---------------------------------------------------------------- 消融 */

const ablationGroups = ref<AblationGroup[]>([])
const ablationLoading = ref(false)
const ablationError = ref('')
const ablationTopK = ref(5)

/* ---------------------------------------------------------------- 新建对话框 */

const datasetDialogVisible = ref(false)
const runDialogVisible = ref(false)
const submitting = ref(false)

const datasetForm = reactive({
  name: '',
  description: '',
  /** JSONL 风格文本：每行一个问题；可选 `问题<TAB>标准答案` */
  casesText: '',
})

const runForm = reactive({
  name: '',
  mode: 'hybrid_rerank' as EvalMode,
  top_k: 5,
  use_rerank: true,
  use_llm_judge: false,
})

const MODE_OPTIONS: Array<{ label: string; value: EvalMode }> = [
  { label: '纯向量 vector', value: 'vector' },
  { label: '纯关键词 bm25', value: 'bm25' },
  { label: '混合 hybrid', value: 'hybrid' },
  { label: '混合 + 重排 hybrid_rerank', value: 'hybrid_rerank' },
  { label: 'Agent agent', value: 'agent' },
]

const selectedDataset = computed<DatasetOut | null>(
  () => datasets.value.find((item) => item.id === selectedDatasetId.value) ?? null,
)

/** 只要有 run 还在跑就轮询 */
const hasActiveRun = computed<boolean>(() =>
  runs.value.some((run) => run.status === 'pending' || run.status === 'running'),
)
let pollTimer: number | null = null

function syncPolling(): void {
  if (hasActiveRun.value && pollTimer === null) {
    pollTimer = window.setInterval(() => {
      void loadRuns(true)
    }, 4000)
  } else if (!hasActiveRun.value && pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

/* ---------------------------------------------------------------- 加载 */

async function loadDatasets(): Promise<void> {
  datasetsLoading.value = true
  datasetsError.value = ''
  try {
    datasets.value = await evalApi.listDatasets()
    if (selectedDatasetId.value === null && datasets.value.length > 0) {
      selectedDatasetId.value = datasets.value[0]?.id ?? null
    }
  } catch (error) {
    datasetsError.value = describeError(error)
  } finally {
    datasetsLoading.value = false
  }
}

async function loadRuns(silent = false): Promise<void> {
  if (!silent) runsLoading.value = true
  runsError.value = ''
  try {
    const result = await evalApi.listRuns({
      page: runsPage.value,
      size: runsSize.value,
      dataset_id: selectedDatasetId.value ?? undefined,
    })
    runs.value = result.items
    runsTotal.value = result.total
    runsPages.value = result.pages
    syncPolling()
  } catch (error) {
    if (!silent) runsError.value = describeError(error)
  } finally {
    runsLoading.value = false
  }
}

async function loadAblation(): Promise<void> {
  if (selectedDatasetId.value === null) return
  ablationLoading.value = true
  ablationError.value = ''
  try {
    const result = await evalApi.ablation({ dataset_id: selectedDatasetId.value, top_k: ablationTopK.value })
    ablationGroups.value = result.groups
  } catch (error) {
    ablationError.value = describeError(error)
    ablationGroups.value = []
  } finally {
    ablationLoading.value = false
  }
}

function handleDatasetChange(): void {
  runsPage.value = 1
  ablationGroups.value = []
  void loadRuns()
}

function selectDataset(datasetId: number): void {
  selectedDatasetId.value = datasetId
  handleDatasetChange()
}

function handleRunPageChange(payload: { page: number; size: number }): void {
  runsPage.value = payload.page
  runsSize.value = payload.size
  void loadRuns()
}

/* ---------------------------------------------------------------- 动作 */

/** 把用户粘贴的文本转成 EvalCaseInput[]（每行一个问题，可用 Tab 或 `|` 分隔标准答案） */
function parseCases(text: string): Array<{ question: string; ground_truth: string | null }> {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '')
    .map((line) => {
      const parts = line.split(/\t|\s\|\s/)
      return {
        question: (parts[0] ?? '').trim(),
        ground_truth: parts.length > 1 ? (parts.slice(1).join(' ').trim() || null) : null,
      }
    })
}

async function submitDataset(): Promise<void> {
  if (datasetForm.name.trim() === '') {
    ElMessage.warning('请填写数据集名称')
    return
  }
  const cases = parseCases(datasetForm.casesText)
  // 后端 `DatasetCreateRequest.cases` 要求至少 1 条：
  // 空数据集能建成但跑起来必然 case_count=0、指标全 null，看起来像评测功能坏了。
  if (cases.length === 0) {
    ElMessage.warning('至少需要 1 条用例（每行一条：问题，可选 Tab/| 分隔标准答案）')
    return
  }
  submitting.value = true
  try {
    const created = await evalApi.createDataset({
      name: datasetForm.name.trim(),
      description: datasetForm.description.trim() || null,
      cases,
    })
    ElMessage.success(`数据集「${created.name}」已创建（${cases.length} 条用例）`)
    datasetDialogVisible.value = false
    datasetForm.name = ''
    datasetForm.description = ''
    datasetForm.casesText = ''
    await loadDatasets()
    selectedDatasetId.value = created.id
    await loadRuns()
  } catch (error) {
    ElMessage.error(describeError(error))
  } finally {
    submitting.value = false
  }
}

async function submitRun(): Promise<void> {
  if (selectedDatasetId.value === null) {
    ElMessage.warning('请先选择数据集')
    return
  }
  submitting.value = true
  try {
    const created = await evalApi.createRun({
      dataset_id: selectedDatasetId.value,
      name: runForm.name.trim() || undefined,
      mode: runForm.mode,
      top_k: runForm.top_k,
      use_rerank: runForm.use_rerank,
      use_llm_judge: runForm.use_llm_judge,
    })
    ElMessage.success(`评测任务 #${created.id} 已创建，后台执行中`)
    runDialogVisible.value = false
    runForm.name = ''
    runsPage.value = 1
    await loadRuns()
  } catch (error) {
    ElMessage.error(describeError(error))
  } finally {
    submitting.value = false
  }
}

function openRunDetail(run: EvalRunOut): void {
  void router.push(`/eval/runs/${run.id}`)
}

function openCompare(run: EvalRunOut): void {
  // 对比功能在运行详情页里选另一个 run，这里带上 run_a
  void router.push({ path: `/eval/runs/${run.id}`, query: { compare: '1' } })
  ElMessage.info('在详情页选择「对比运行」即可查看 delta 与 95% 置信区间')
}

async function confirmDeleteRun(run: EvalRunOut): Promise<void> {
  // 契约里没有 DELETE /eval/runs/{id}，所以这里只提示，不发请求
  try {
    await ElMessageBox.alert(
      `契约 5.8 未定义删除评测运行的接口，因此前端不提供删除操作。运行 #${run.id}（${run.name}）如需删除请在数据库中处理。`,
      '暂不支持删除',
      { confirmButtonText: '知道了' },
    )
  } catch {
    /* 用户关闭弹窗 */
  }
}

/** 消融对比图数据：AblationGroup 的指标是**扁平**字段，不是嵌套的 metrics */
const ablationModes = computed<string[]>(() =>
  ablationGroups.value.map((group) => group.label || searchModeText(String(group.mode))),
)
const ablationMetrics = computed(() => ({
  recall: ablationGroups.value.map((group) => group.recall_at_k),
  hit: ablationGroups.value.map((group) => group.hit_rate),
  mrr: ablationGroups.value.map((group) => group.mrr),
  ndcg: ablationGroups.value.map((group) => group.ndcg),
}))

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'done') return 'success'
  if (status === 'failed') return 'danger'
  if (status === 'running') return 'warning'
  return 'info'
}

/** 相对基线差值的展示：正数带 + 号，正绿负红 */
function deltaText(value: number | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(4)}`
}

function deltaClass(value: number | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value) || value === 0) return ''
  return value > 0 ? 'delta--good' : 'delta--bad'
}

function statusText(status: string): string {
  const map: Record<string, string> = { pending: '排队中', running: '执行中', done: '已完成', failed: '失败' }
  return map[status] ?? status
}

onMounted(async () => {
  await loadDatasets()
  await loadRuns()
})

onBeforeUnmount(stopPolling)
</script>

<template>
  <div class="kf-page">
    <div class="kf-page-header">
      <div>
        <h2 class="kf-page-title">评测</h2>
        <p class="kf-page-subtitle">
          把「我感觉更好了」变成数字：标注集 + recall@k / MRR / nDCG + 消融实验。
        </p>
      </div>
      <div class="kf-row">
        <el-button @click="loadDatasets(); loadRuns()">刷新</el-button>
        <el-button @click="datasetDialogVisible = true">新建数据集</el-button>
        <el-button type="primary" :disabled="selectedDatasetId === null" @click="runDialogVisible = true">
          新建评测运行
        </el-button>
      </div>
    </div>

    <ErrorBlock v-if="datasetsError" :message="datasetsError" @retry="loadDatasets" />

    <div class="kf-grid kf-grid--2">
      <!-- 数据集 -->
      <div class="kf-card">
        <div class="kf-card-title">
          <h3>数据集</h3>
          <span class="kf-muted">{{ datasets.length }} 个</span>
        </div>
        <LoadingBlock v-if="datasetsLoading && datasets.length === 0" height="160px" />
        <EmptyState
          v-else-if="datasets.length === 0"
          icon="🗂️"
          title="还没有评测数据集"
          description="数据集包含问题 + 标准答案 + 应命中的文档/小节，覆盖同义改写、专有名词、多跳三类难点。"
        >
          <el-button type="primary" @click="datasetDialogVisible = true">新建数据集</el-button>
        </EmptyState>
        <div v-else class="dataset-list">
          <div
            v-for="dataset in datasets"
            :key="dataset.id"
            class="dataset-item"
            :class="{ 'dataset-item--active': dataset.id === selectedDatasetId }"
            @click="selectDataset(dataset.id)"
          >
            <div class="dataset-item__main">
              <div class="dataset-item__name">{{ dataset.name }}</div>
              <div class="dataset-item__desc kf-muted">{{ dataset.description || '（无描述）' }}</div>
              <div class="dataset-item__meta kf-muted">
                {{ formatInt(dataset.case_count) }} 条用例
              </div>
            </div>
            <el-tag v-if="dataset.id === selectedDatasetId" size="small" type="primary" effect="dark">已选</el-tag>
          </div>
        </div>
      </div>

      <!-- 消融对比 -->
      <div class="kf-card">
        <div class="kf-card-title">
          <h3>消融对比</h3>
          <div class="kf-row">
            <span class="kf-muted">top_k</span>
            <el-input-number v-model="ablationTopK" size="small" :min="1" :max="20" style="width: 90px" />
            <el-button size="small" type="primary" :disabled="selectedDatasetId === null" @click="loadAblation">
              运行对比
            </el-button>
          </div>
        </div>

        <ErrorBlock v-if="ablationError" class="kf-card-body" :message="ablationError" @retry="loadAblation" />

        <EmptyState
          v-else-if="ablationGroups.length === 0"
          icon="🧪"
          title="未运行消融实验"
          description="纯向量 / 纯 BM25 / 混合 / 混合+重排 四组跑同一数据集，输出对比表——简历里「recall@5 从 X 提到 Y」的数字就来自这里。"
        >
          <el-button type="primary" :disabled="selectedDatasetId === null" @click="loadAblation">运行对比</el-button>
        </EmptyState>

        <template v-else>
          <BarChart
            :labels="ablationModes"
            :series="[
              { name: 'recall@k', data: ablationMetrics.recall },
              { name: 'hit_rate', data: ablationMetrics.hit },
              { name: 'MRR', data: ablationMetrics.mrr },
              { name: 'nDCG', data: ablationMetrics.ndcg },
            ]"
            height="240px"
            :decimals="3"
            :loading="ablationLoading"
          />
          <el-table :data="ablationGroups" size="small">
            <el-table-column label="方案" min-width="150">
              <template #default="{ row }: { row: AblationGroup }">
                {{ row.label || searchModeText(String(row.mode)) }}
              </template>
            </el-table-column>
            <el-table-column label="mode" width="120">
              <template #default="{ row }: { row: AblationGroup }">
                <span class="kf-mono">{{ row.mode }}</span>
              </template>
            </el-table-column>
            <el-table-column label="run" width="70">
              <template #default="{ row }: { row: AblationGroup }">{{ row.run_id ?? '—' }}</template>
            </el-table-column>
            <el-table-column label="用例数" width="80" align="right">
              <template #default="{ row }: { row: AblationGroup }">{{ formatInt(row.case_count) }}</template>
            </el-table-column>
            <el-table-column label="recall@k" width="100" align="right">
              <template #default="{ row }: { row: AblationGroup }">{{ formatFixed(row.recall_at_k, 3) }}</template>
            </el-table-column>
            <el-table-column label="hit_rate" width="100" align="right">
              <template #default="{ row }: { row: AblationGroup }">{{ formatFixed(row.hit_rate, 3) }}</template>
            </el-table-column>
            <el-table-column label="MRR" width="90" align="right">
              <template #default="{ row }: { row: AblationGroup }">{{ formatFixed(row.mrr, 3) }}</template>
            </el-table-column>
            <el-table-column label="nDCG" width="90" align="right">
              <template #default="{ row }: { row: AblationGroup }">{{ formatFixed(row.ndcg, 3) }}</template>
            </el-table-column>
            <el-table-column label="平均延迟" width="110" align="right">
              <template #default="{ row }: { row: AblationGroup }">
                {{ formatDuration(row.avg_latency_ms) }}
              </template>
            </el-table-column>
            <el-table-column label="Δ recall@k" width="110" align="right">
              <template #default="{ row }: { row: AblationGroup }">
                <span
                  class="kf-mono"
                  :class="deltaClass(row.delta_vs_baseline.recall_at_k)"
                >
                  {{ deltaText(row.delta_vs_baseline.recall_at_k) }}
                </span>
              </template>
            </el-table-column>
          </el-table>
        </template>
      </div>
    </div>

    <!-- 运行列表 -->
    <div class="kf-card">
      <div class="kf-card-title">
        <h3>
          评测运行
          <span class="kf-muted" v-if="selectedDataset">（数据集：{{ selectedDataset.name }}）</span>
        </h3>
        <div class="kf-row">
          <el-tag v-if="hasActiveRun" size="small" type="warning" effect="plain">有任务执行中，自动刷新</el-tag>
          <el-button size="small" @click="loadRuns()">刷新</el-button>
        </div>
      </div>

      <ErrorBlock v-if="runsError" class="kf-card-body" :message="runsError" @retry="loadRuns()" />

      <LoadingBlock v-else-if="runsLoading && runs.length === 0" height="160px" />

      <EmptyState
        v-else-if="runs.length === 0"
        icon="📊"
        title="还没有评测运行"
        description="选一个数据集，指定检索模式与 top_k，后端会后台逐条跑完并算出聚合指标。"
      />

      <template v-else>
        <el-table :data="runs" size="small">
          <el-table-column prop="id" label="#" width="60" />
          <el-table-column prop="name" label="名称" min-width="150" show-overflow-tooltip />
          <el-table-column label="模式" width="130">
            <template #default="{ row }: { row: EvalRunOut }">{{ searchModeText(row.mode) }}</template>
          </el-table-column>
          <el-table-column prop="top_k" label="top_k" width="70" />
          <el-table-column label="状态" width="100">
            <template #default="{ row }: { row: EvalRunOut }">
              <el-tag size="small" :type="statusTagType(row.status)" effect="plain">{{ statusText(row.status) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="通过/总数" width="110">
            <template #default="{ row }: { row: EvalRunOut }">
              {{ formatInt(row.passed_count) }} / {{ formatInt(row.case_count) }}
            </template>
          </el-table-column>
          <el-table-column label="recall@k" width="100" align="right">
            <template #default="{ row }: { row: EvalRunOut }">{{ formatFixed(row.metrics_json?.recall_at_k, 3) }}</template>
          </el-table-column>
          <el-table-column label="MRR" width="90" align="right">
            <template #default="{ row }: { row: EvalRunOut }">{{ formatFixed(row.metrics_json?.mrr, 3) }}</template>
          </el-table-column>
          <el-table-column label="faithfulness" width="120" align="right">
            <template #default="{ row }: { row: EvalRunOut }">{{ formatFixed(row.metrics_json?.faithfulness, 3) }}</template>
          </el-table-column>
          <el-table-column label="P95 延迟" width="110" align="right">
            <template #default="{ row }: { row: EvalRunOut }">
              {{ formatDuration(row.metrics_json?.p95_latency_ms) }}
            </template>
          </el-table-column>
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }: { row: EvalRunOut }">{{ formatDateTime(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="170" fixed="right">
            <template #default="{ row }: { row: EvalRunOut }">
              <el-button link size="small" type="primary" @click="openRunDetail(row)">详情</el-button>
              <el-button link size="small" @click="openCompare(row)">对比</el-button>
              <el-button link size="small" type="info" @click="confirmDeleteRun(row)">删除</el-button>
            </template>
          </el-table-column>
          <el-table-column type="expand">
            <template #default="{ row }: { row: EvalRunOut }">
              <div class="run-expand">
                <div v-if="row.error" class="run-expand__error">失败原因：{{ row.error }}</div>
                <div class="kf-muted">参数快照（config_json）</div>
                <pre class="kf-pre">{{ JSON.stringify(row.config_json, null, 2) }}</pre>
                <div class="kf-muted">聚合指标（metrics_json）</div>
                <pre class="kf-pre">{{ JSON.stringify(row.metrics_json, null, 2) }}</pre>
                <div class="kf-muted">
                  开始 {{ formatDateTime(row.started_at) }} · 结束 {{ formatDateTime(row.finished_at) }}
                </div>
              </div>
            </template>
          </el-table-column>
        </el-table>

        <PageBar :page="runsPage" :size="runsSize" :total="runsTotal" :pages="runsPages" @change="handleRunPageChange" />
      </template>
    </div>

    <!-- 关键指标速览 -->
    <div v-if="selectedDataset" class="kf-grid kf-grid--4 quick-stats">
      <StatCard label="数据集用例数" :value="formatInt(selectedDataset.case_count)" icon="🗂️" />
      <StatCard label="运行总数" :value="formatInt(runsTotal)" icon="📊" />
      <StatCard
        label="最近一次 recall@k"
        :value="formatFixed(runs[0]?.metrics_json?.recall_at_k, 3)"
        icon="🎯"
        tone="primary"
      />
      <StatCard
        label="最近一次 P95 延迟"
        :value="formatDuration(runs[0]?.metrics_json?.p95_latency_ms)"
        icon="⏱️"
      />
    </div>

    <!-- 新建数据集 -->
    <el-dialog v-model="datasetDialogVisible" title="新建评测数据集" width="620px">
      <el-form label-width="90px">
        <el-form-item label="名称" required>
          <el-input v-model="datasetForm.name" placeholder="例如：员工制度问答-30" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="datasetForm.description" type="textarea" :rows="2" />
        </el-form-item>
        <el-form-item label="用例">
          <el-input
            v-model="datasetForm.casesText"
            type="textarea"
            :rows="8"
            placeholder="每行一条：问题&lt;TAB&gt;标准答案（标准答案可省略）&#10;例：一线城市住宿标准是多少&#9;每晚 600 元"
          />
          <div class="kf-muted dialog-hint">
            将解析出 {{ parseCases(datasetForm.casesText).length }} 条用例；标准答案可用 Tab 或 <code> | </code> 分隔。
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="datasetDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitDataset">创建</el-button>
      </template>
    </el-dialog>

    <!-- 新建运行 -->
    <el-dialog v-model="runDialogVisible" title="新建评测运行" width="520px">
      <el-form label-width="110px">
        <el-form-item label="数据集">
          <el-input :model-value="selectedDataset?.name ?? ''" disabled />
        </el-form-item>
        <el-form-item label="名称">
          <el-input v-model="runForm.name" placeholder="留空由后端生成" />
        </el-form-item>
        <el-form-item label="检索模式">
          <el-select v-model="runForm.mode" style="width: 100%">
            <el-option v-for="option in MODE_OPTIONS" :key="option.value" :label="option.label" :value="option.value" />
          </el-select>
        </el-form-item>
        <el-form-item label="top_k">
          <el-input-number v-model="runForm.top_k" :min="1" :max="20" />
        </el-form-item>
        <el-form-item label="启用重排">
          <el-switch v-model="runForm.use_rerank" />
        </el-form-item>
        <el-form-item label="LLM 判定">
          <el-switch v-model="runForm.use_llm_judge" />
          <span class="kf-muted dialog-hint">开启会用大模型做 faithfulness 判定（更准但花钱）</span>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="runDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitRun">开始评测</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.dataset-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 320px;
  overflow: auto;
}

.dataset-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid transparent;
  border-radius: var(--kf-radius-sm);
  cursor: pointer;
}

.dataset-item:hover {
  background: var(--kf-bg-hover);
}

.dataset-item--active {
  border-color: var(--kf-primary);
  background: color-mix(in srgb, var(--kf-primary) 10%, transparent);
}

.dataset-item__main {
  flex: 1 1 auto;
  min-width: 0;
}

.dataset-item__name {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--kf-text-primary);
}

.dataset-item__desc,
.dataset-item__meta {
  font-size: 12px;
}

.run-expand {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 12px;
  font-size: 12px;
}

.run-expand__error {
  color: var(--kf-danger);
}

.quick-stats {
  margin-top: 16px;
}

.dialog-hint {
  font-size: 12px;
  line-height: 1.5;
}

.delta--good {
  color: var(--kf-success);
  font-weight: 600;
}

.delta--bad {
  color: var(--kf-danger);
  font-weight: 600;
}
</style>
