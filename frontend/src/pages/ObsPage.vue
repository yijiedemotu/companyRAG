<script setup lang="ts">
/**
 * 可观测页 `/obs`。
 *
 * 契约 5.9：
 * - `GET /obs/stats?hours=24` → 请求量 / token / 成本(USD+CNY) / 延迟分位 + 按 mode 与按 model 分组 + 时间序列；
 * - `GET /obs/quality?hours=24` → 拒答率 / 平均引用数 / 零引用率 / 平均检索轮数 / 反思通过率；
 * - `GET /obs/traces?page&size&status&mode&hours` → trace 列表，点进 `/obs/traces/{trace_id}` 看瀑布图。
 *
 * 注意：契约只给了 ObsStatsOut 的**语义**，没有逐字的嵌套字段名，
 * 所以 `api/obs.ts` 里做了一层运行期归一化（详见该文件的注释与 README 的契约歧义清单）。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { describeError } from '@/api/client'
import { getQuality, getStats, listTraces } from '@/api/obs'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LineChart from '@/components/LineChart.vue'
import PageBar from '@/components/PageBar.vue'
import StatCard from '@/components/StatCard.vue'
import type {
  ByModeOut,
  ByModelOut,
  ObsQualityOut,
  ObsStatsOut,
  TimelinePointOut,
  TraceOut,
} from '@/types/models'
import {
  formatCny,
  formatDateTime,
  formatDuration,
  formatFixed,
  formatInt,
  formatPercent,
  formatShortTime,
  formatUsd,
  safeNumber,
  searchModeText,
  usdToCny,
} from '@/utils/format'

const router = useRouter()

/* ---------------------------------------------------------------- 窗口与筛选 */

const hours = ref(24)
const traceStatus = ref('')
const traceMode = ref('')

/* ---------------------------------------------------------------- 数据 */

const stats = ref<ObsStatsOut | null>(null)
const statsLoading = ref(false)
const statsError = ref('')

const quality = ref<ObsQualityOut | null>(null)
const qualityError = ref('')

const traces = ref<TraceOut[]>([])
const tracesTotal = ref(0)
const tracesPage = ref(1)
const tracesSize = ref(10)
const tracesPages = ref(0)
const tracesLoading = ref(false)
const tracesError = ref('')

let pollTimer: number | null = null

async function loadStats(): Promise<void> {
  statsLoading.value = true
  statsError.value = ''
  try {
    stats.value = await getStats({ hours: hours.value })
  } catch (error) {
    statsError.value = describeError(error)
  } finally {
    statsLoading.value = false
  }
}

async function loadQuality(): Promise<void> {
  qualityError.value = ''
  try {
    quality.value = await getQuality({ hours: hours.value })
  } catch (error) {
    qualityError.value = describeError(error)
    quality.value = null
  }
}

async function loadTraces(silent = false): Promise<void> {
  if (!silent) tracesLoading.value = true
  tracesError.value = ''
  try {
    const result = await listTraces({
      page: tracesPage.value,
      size: tracesSize.value,
      status: traceStatus.value || undefined,
      mode: traceMode.value || undefined,
      hours: hours.value,
    })
    traces.value = result.items
    tracesTotal.value = result.total
    tracesPages.value = result.pages
  } catch (error) {
    if (!silent) tracesError.value = describeError(error)
  } finally {
    tracesLoading.value = false
  }
}

async function refreshAll(): Promise<void> {
  await Promise.all([loadStats(), loadQuality(), loadTraces()])
}

function handleHoursChange(): void {
  tracesPage.value = 1
  void refreshAll()
}

function handleTraceFilterChange(): void {
  tracesPage.value = 1
  void loadTraces()
}

function handleTracePageChange(payload: { page: number; size: number }): void {
  tracesPage.value = payload.page
  tracesSize.value = payload.size
  void loadTraces()
}

/* ---------------------------------------------------------------- 派生数据 */

/** 成本：后端直接给了 cost_cny（由 USD_TO_CNY 换算），缺失时前端兜底换算 */
const costCny = computed<number | null>(() => {
  const data = stats.value
  if (!data) return null
  if (data.cost_cny > 0) return data.cost_cny
  return usdToCny(data.cost_usd)
})

/** 时间序列：后端 `timeline[].bucket` 是时间桶起点（UTC ISO8601 带 Z） */
const seriesLabels = computed<string[]>(() =>
  (stats.value?.timeline ?? []).map((point: TimelinePointOut, index) => {
    if (!point.bucket) return `#${index + 1}`
    return formatShortTime(point.bucket)
  }),
)

const seriesRequests = computed<Array<number | null>>(() =>
  (stats.value?.timeline ?? []).map((point) => point.requests),
)

const seriesTokens = computed<Array<number | null>>(() =>
  (stats.value?.timeline ?? []).map((point) => point.tokens),
)

const seriesCost = computed<Array<number | null>>(() =>
  (stats.value?.timeline ?? []).map((point) => point.cost_usd),
)

/**
 * 时间序列里没有 P95（只有平均延迟），所以这张图画的是**平均延迟趋势**。
 * 全窗口的 P50/P95/P99 在上面那张延迟卡里。
 */
const seriesLatency = computed<Array<number | null>>(() =>
  (stats.value?.timeline ?? []).map((point) => point.avg_latency_ms),
)

const hasSeries = computed<boolean>(() => seriesLabels.value.length > 0)

function modeLabel(mode: string): string {
  return searchModeText(mode)
}

function refusalText(value: number): string {
  return formatPercent(safeNumber(value, 0))
}

function statusTagType(status: string): 'success' | 'danger' {
  return status === 'ok' ? 'success' : 'danger'
}

function openTrace(trace: TraceOut): void {
  void router.push(`/obs/traces/${trace.trace_id}`)
}

onMounted(async () => {
  await refreshAll()
  // 可观测页面数据变化快，30 秒自动刷新一次
  pollTimer = window.setInterval(() => {
    void loadStats()
    void loadTraces(true)
  }, 30_000)
})

onBeforeUnmount(() => {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
})
</script>

<template>
  <div class="kf-page">
    <div class="kf-page-header">
      <div>
        <h2 class="kf-page-title">可观测</h2>
        <p class="kf-page-subtitle">
          回答三个问题：这次请求为什么慢？这个功能一天花多少钱？线上质量有没有退化？
        </p>
      </div>
      <div class="kf-row">
        <el-radio-group v-model="hours" size="small" @change="handleHoursChange">
          <el-radio-button :value="1">1 小时</el-radio-button>
          <el-radio-button :value="24">24 小时</el-radio-button>
          <el-radio-button :value="168">7 天</el-radio-button>
          <el-radio-button :value="720">30 天</el-radio-button>
        </el-radio-group>
        <el-button @click="refreshAll">刷新</el-button>
      </div>
    </div>

    <ErrorBlock v-if="statsError" :message="statsError" @retry="loadStats" />

    <!-- 四类核心卡片 -->
    <div class="stat-grid">
      <StatCard label="请求量" :value="formatInt(stats?.requests)" icon="📨" tone="primary" :loading="statsLoading">
        <template #hint>窗口 {{ hours }} 小时</template>
      </StatCard>

      <StatCard label="token 总量" :value="formatInt(stats?.tokens.total)" icon="🔤" :loading="statsLoading">
        <template #hint>
          输入 {{ formatInt(stats?.tokens.prompt) }} · 输出 {{ formatInt(stats?.tokens.completion) }}
        </template>
      </StatCard>

      <StatCard label="成本" :value="formatUsd(stats?.cost_usd)" icon="💰" tone="warning" :loading="statsLoading">
        <template #hint>
          后端换算 {{ formatCny(costCny) }}（USD_TO_CNY）
        </template>
      </StatCard>

      <StatCard
        label="延迟 P50 / P95 / P99"
        :value="formatDuration(stats?.latency.p95)"
        icon="⏱️"
        tone="danger"
        :loading="statsLoading"
      >
        <template #hint>
          P50 {{ formatDuration(stats?.latency.p50) }} · P95 {{ formatDuration(stats?.latency.p95) }} · P99
          {{ formatDuration(stats?.latency.p99) }} · 最大 {{ formatDuration(stats?.latency.max) }}
        </template>
      </StatCard>
    </div>

    <!-- 质量卡 -->
    <div class="kf-card">
      <div class="kf-card-title">
        <h3>质量指标（/obs/quality）</h3>
        <span class="kf-muted">窗口 {{ hours }} 小时 · 样本 {{ formatInt(quality?.total) }} 次请求</span>
      </div>
      <ErrorBlock v-if="qualityError" class="kf-card-body" :message="qualityError" @retry="loadQuality" />
      <div v-else class="quality-body">
        <div class="quality-item">
          <div class="quality-item__label kf-muted">拒答率</div>
          <div class="quality-item__value">{{ formatPercent(quality?.refusal_rate) }}</div>
          <div class="quality-item__hint kf-muted">闸门拦住的比例；突然升高通常意味着召回变差</div>
        </div>
        <div class="quality-item">
          <div class="quality-item__label kf-muted">平均引用数</div>
          <div class="quality-item__value">{{ formatFixed(quality?.avg_source_count, 2) }}</div>
          <div class="quality-item__hint kf-muted">每条回答平均引用几个片段</div>
        </div>
        <div class="quality-item">
          <div class="quality-item__label kf-muted">零引用率</div>
          <div class="quality-item__value">{{ formatPercent(quality?.zero_source_rate) }}</div>
          <div class="quality-item__hint kf-muted">完全没有引用的回答占比（无依据断言的风险面）</div>
        </div>
        <div class="quality-item">
          <div class="quality-item__label kf-muted">平均检索轮数</div>
          <div class="quality-item__value">{{ formatFixed(quality?.avg_retrieval_rounds, 2) }}</div>
          <div class="quality-item__hint kf-muted">&gt; 1 说明 rewrite 在起作用</div>
        </div>
        <div class="quality-item">
          <div class="quality-item__label kf-muted">反思通过率</div>
          <div class="quality-item__value">{{ formatPercent(quality?.reflect_pass_rate) }}</div>
          <div class="quality-item__hint kf-muted">Self-RAG 检查「每句都有引用支撑」的通过比例</div>
        </div>
      </div>
    </div>

    <!-- 时间序列 -->
    <div v-if="hasSeries" class="kf-grid kf-grid--2 charts">
      <div class="kf-card">
        <div class="kf-card-title"><h3>请求量趋势</h3></div>
        <div class="kf-card-body">
          <LineChart :labels="seriesLabels" :series="[{ name: '请求数', data: seriesRequests }]" area height="240px" />
        </div>
      </div>
      <div class="kf-card">
        <div class="kf-card-title"><h3>token 与成本趋势</h3></div>
        <div class="kf-card-body">
          <LineChart
            :labels="seriesLabels"
            :series="[
              { name: 'total_tokens', data: seriesTokens },
              { name: 'cost_usd', data: seriesCost, yAxisIndex: 1 },
            ]"
            y-name="tokens"
            y-name-right="USD"
            height="240px"
          />
        </div>
      </div>
      <div class="kf-card">
        <div class="kf-card-title"><h3>P95 延迟趋势</h3></div>
        <div class="kf-card-body">
          <LineChart
            :labels="seriesLabels"
            :series="[{ name: '平均延迟(ms)', data: seriesLatency }]"
            y-name="ms"
            area
            height="240px"
          />
        </div>
      </div>
    </div>

    <!-- 分组表 -->
    <div class="kf-grid kf-grid--2">
      <div class="kf-card">
        <div class="kf-card-title"><h3>按 mode 分组</h3></div>
        <el-table :data="stats?.by_mode ?? []" size="small">
          <el-table-column label="mode" min-width="130">
            <template #default="{ row }: { row: ByModeOut }">{{ modeLabel(row.mode) }}</template>
          </el-table-column>
          <el-table-column label="请求数" width="90" align="right">
            <template #default="{ row }: { row: ByModeOut }">{{ formatInt(row.requests) }}</template>
          </el-table-column>
          <el-table-column label="token" width="110" align="right">
            <template #default="{ row }: { row: ByModeOut }">{{ formatInt(row.tokens) }}</template>
          </el-table-column>
          <el-table-column label="成本" width="110" align="right">
            <template #default="{ row }: { row: ByModeOut }">{{ formatUsd(row.cost_usd) }}</template>
          </el-table-column>
          <el-table-column label="平均延迟" width="110" align="right">
            <template #default="{ row }: { row: ByModeOut }">{{ formatDuration(row.avg_latency_ms) }}</template>
          </el-table-column>
          <el-table-column label="拒答率" width="100" align="right">
            <template #default="{ row }: { row: ByModeOut }">{{ refusalText(row.refusal_rate) }}</template>
          </el-table-column>
        </el-table>
      </div>

      <div class="kf-card">
        <div class="kf-card-title"><h3>按 model 分组</h3></div>
        <el-table :data="stats?.by_model ?? []" size="small">
          <el-table-column label="model" min-width="150">
            <template #default="{ row }: { row: ByModelOut }">
              <span class="kf-mono">{{ row.model || '未知' }}</span>
            </template>
          </el-table-column>
          <el-table-column label="请求数" width="90" align="right">
            <template #default="{ row }: { row: ByModelOut }">{{ formatInt(row.requests) }}</template>
          </el-table-column>
          <el-table-column label="输入 token" width="110" align="right">
            <template #default="{ row }: { row: ByModelOut }">{{ formatInt(row.prompt_tokens) }}</template>
          </el-table-column>
          <el-table-column label="输出 token" width="110" align="right">
            <template #default="{ row }: { row: ByModelOut }">{{ formatInt(row.completion_tokens) }}</template>
          </el-table-column>
          <el-table-column label="成本" width="110" align="right">
            <template #default="{ row }: { row: ByModelOut }">{{ formatUsd(row.cost_usd) }}</template>
          </el-table-column>
          <el-table-column label="平均延迟" width="110" align="right">
            <template #default="{ row }: { row: ByModelOut }">{{ formatDuration(row.avg_latency_ms) }}</template>
          </el-table-column>
        </el-table>
      </div>
    </div>

    <!-- trace 列表 -->
    <div class="kf-card traces-card">
      <div class="kf-card-title">
        <h3>链路列表</h3>
        <div class="kf-row">
          <el-select v-model="traceStatus" placeholder="全部状态" clearable size="small" style="width: 130px" @change="handleTraceFilterChange">
            <el-option label="成功" value="ok" />
            <el-option label="失败" value="error" />
          </el-select>
          <el-select v-model="traceMode" placeholder="全部模式" clearable size="small" style="width: 130px" @change="handleTraceFilterChange">
            <el-option label="rag" value="rag" />
            <el-option label="agent" value="agent" />
          </el-select>
          <el-button size="small" @click="loadTraces()">刷新</el-button>
        </div>
      </div>

      <ErrorBlock v-if="tracesError" class="kf-card-body" :message="tracesError" @retry="loadTraces()" />

      <el-table v-loading="tracesLoading" :data="traces" size="small">
        <el-table-column prop="trace_id" label="trace_id" min-width="200" show-overflow-tooltip>
          <template #default="{ row }: { row: TraceOut }">
            <span class="kf-mono">{{ row.trace_id }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="名称" width="90" />
        <el-table-column label="模式" width="80">
          <template #default="{ row }: { row: TraceOut }">{{ row.mode ?? '—' }}</template>
        </el-table-column>
        <el-table-column label="状态" width="90">
          <template #default="{ row }: { row: TraceOut }">
            <el-tag size="small" :type="statusTagType(row.status)" effect="plain">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="端到端" width="100" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ formatDuration(row.latency_ms) }}</template>
        </el-table-column>
        <el-table-column label="检索" width="90" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ formatDuration(row.retrieval_ms) }}</template>
        </el-table-column>
        <el-table-column label="重排" width="90" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ formatDuration(row.rerank_ms) }}</template>
        </el-table-column>
        <el-table-column label="生成" width="90" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ formatDuration(row.generate_ms) }}</template>
        </el-table-column>
        <el-table-column label="LLM 调用" width="95" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ formatInt(row.llm_calls) }}</template>
        </el-table-column>
        <el-table-column label="token" width="110" align="right">
          <template #default="{ row }: { row: TraceOut }">
            {{ formatInt(row.prompt_tokens) }} / {{ formatInt(row.completion_tokens) }}
          </template>
        </el-table-column>
        <el-table-column label="成本" width="100" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ formatUsd(row.cost_usd) }}</template>
        </el-table-column>
        <el-table-column label="检索轮数" width="90" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ row.retrieval_rounds }}</template>
        </el-table-column>
        <el-table-column label="引用" width="70" align="right">
          <template #default="{ row }: { row: TraceOut }">{{ row.source_count }}</template>
        </el-table-column>
        <el-table-column label="拒答" width="70">
          <template #default="{ row }: { row: TraceOut }">
            <el-tag v-if="row.refusal" size="small" type="warning" effect="plain">是</el-tag>
            <span v-else class="kf-muted">否</span>
          </template>
        </el-table-column>
        <el-table-column label="时间" width="170">
          <template #default="{ row }: { row: TraceOut }">{{ formatDateTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="90" fixed="right">
          <template #default="{ row }: { row: TraceOut }">
            <el-button link size="small" type="primary" @click="openTrace(row)">瀑布图</el-button>
          </template>
        </el-table-column>
        <el-table-column type="expand">
          <template #default="{ row }: { row: TraceOut }">
            <div class="trace-expand">
              <div><span class="kf-muted">request_id：</span><span class="kf-mono">{{ row.request_id ?? '—' }}</span></div>
              <div><span class="kf-muted">conversation_id：</span>{{ row.conversation_id ?? '—' }}</div>
              <div><span class="kf-muted">kb_id：</span>{{ row.kb_id ?? '—' }}</div>
              <div v-if="row.error" class="trace-expand__error">错误：{{ row.error }}</div>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <PageBar
        :page="tracesPage"
        :size="tracesSize"
        :total="tracesTotal"
        :pages="tracesPages"
        @change="handleTracePageChange"
      />
    </div>
  </div>
</template>

<style scoped>
.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.quality-body {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  padding: 16px;
}

.quality-item__label {
  font-size: 12px;
}

.quality-item__value {
  font-size: 20px;
  font-weight: 600;
  color: var(--kf-text-primary);
  font-variant-numeric: tabular-nums;
}

.quality-item__hint {
  font-size: 11.5px;
  margin-top: 2px;
}

.charts {
  margin-top: 16px;
}

.traces-card {
  margin-top: 16px;
}

.trace-expand {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 14px;
  font-size: 12px;
}

.trace-expand__error {
  color: var(--kf-danger);
}
</style>
