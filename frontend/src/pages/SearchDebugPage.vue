<script setup lang="ts">
/**
 * 检索调试页 `/search` —— 参数面板最丰富的页面，也是调参主力。
 *
 * 契约 5.4：`POST /api/v1/kbs/{kb_id}/search`
 * 全部参数（mode/fusion/alpha/top_k/fetch_k/rerank/use_autocut/vector_threshold/keyword_threshold/include_debug）
 * 都做成可调控件；结果表同时展示
 * `vector_score` / `bm25_score` / `rerank_score` / `vector_rank` / `bm25_rank`
 * 以及 `gate` 判定原因和 `debug` 计数。
 *
 * **这个接口不调用大模型（不花钱）**，所以可以随手狂点。
 */
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'

import { describeError } from '@/api/client'
import { search as searchApi } from '@/api/search'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import { useKbStore } from '@/stores/kb'
import type { FusionMode, SearchHit, SearchMode, SearchResponse } from '@/types/models'
import { formatDuration, formatInt, searchModeText } from '@/utils/format'

const route = useRoute()
const kbStore = useKbStore()

/* ---------------------------------------------------------------- 参数面板 */

const params = reactive({
  query: '',
  mode: 'hybrid_rerank' as SearchMode,
  fusion: 'rrf' as FusionMode,
  alpha: 0.6,
  top_k: 5,
  fetch_k: 20,
  rerank: true,
  use_autocut: true,
  /** null = 用后端配置默认值（对应契约里的 `null`） */
  vector_threshold: null as number | null,
  keyword_threshold: null as number | null,
  include_debug: true,
})

/** 阈值是否覆盖后端默认值（null 时不覆盖） */
const overrideVectorThreshold = ref(false)
const overrideKeywordThreshold = ref(false)

const selectedKbId = ref<number | null>(kbStore.currentKbId)

const running = ref(false)
const response = ref<SearchResponse | null>(null)
const errorText = ref('')
const elapsedMs = ref<number | null>(null)

/** 只有 hybrid / hybrid_rerank 才有融合与 alpha 的概念 */
const showFusion = computed<boolean>(() => params.mode === 'hybrid' || params.mode === 'hybrid_rerank')
const showAlpha = computed<boolean>(() => showFusion.value && params.fusion === 'weighted')

/**
 * 后端 `SearchRequest` 有一条校验：`fetch_k >= top_k`
 * （先召回才有得筛，候选数少于目标条数时结果必然不足）。
 * 这里在 UI 上就把它挡住，避免用户调到非法组合后拿到 422。
 */
watch(
  () => params.top_k,
  (topK) => {
    if (params.fetch_k < topK) params.fetch_k = topK
  },
)

function onTopKChange(): void {
  if (params.fetch_k < params.top_k) params.fetch_k = params.top_k
}

function onFetchKChange(): void {
  if (params.fetch_k < params.top_k) params.top_k = params.fetch_k
}

const hits = computed<SearchHit[]>(() => response.value?.hits ?? [])
const gate = computed(() => response.value?.gate ?? null)
const debugInfo = computed(() => response.value?.debug ?? null)

/** 表格里「最终分数」列的分组说明 */
const timingRows = computed(() => {
  const data = response.value
  if (!data) return []
  return [
    { label: '总耗时', value: data.latency_ms },
    { label: 'embedding', value: data.embedding_ms },
    { label: 'vector', value: data.vector_ms },
    { label: 'bm25', value: data.bm25_ms },
    { label: 'rerank', value: data.rerank_ms },
  ]
})

function scoreText(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) return '—'
  return value.toFixed(digits)
}

/** 单元格配色：越好的分数越绿（只看存在的分数） */
function scoreClass(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'score--none'
  if (value >= 0.7) return 'score--high'
  if (value >= 0.35) return 'score--mid'
  return 'score--low'
}

async function run(): Promise<void> {
  if (!selectedKbId.value) {
    ElMessage.warning('请先选择一个知识库')
    return
  }
  if (params.query.trim() === '') {
    ElMessage.warning('请输入查询语句')
    return
  }

  running.value = true
  errorText.value = ''
  const startedAt = performance.now()
  try {
    response.value = await searchApi(selectedKbId.value, {
      query: params.query.trim(),
      mode: params.mode,
      top_k: params.top_k,
      fetch_k: params.fetch_k,
      fusion: params.fusion,
      alpha: params.alpha,
      rerank: params.rerank,
      use_autocut: params.use_autocut,
      vector_threshold: overrideVectorThreshold.value ? params.vector_threshold : null,
      keyword_threshold: overrideKeywordThreshold.value ? params.keyword_threshold : null,
      include_debug: params.include_debug,
    })
    elapsedMs.value = Math.round(performance.now() - startedAt)
  } catch (error) {
    errorText.value = describeError(error)
    response.value = null
    elapsedMs.value = null
  } finally {
    running.value = false
  }
}

function resetParams(): void {
  params.mode = 'hybrid_rerank'
  params.fusion = 'rrf'
  params.alpha = 0.6
  params.top_k = 5
  params.fetch_k = 20
  params.rerank = true
  params.use_autocut = true
  params.vector_threshold = null
  params.keyword_threshold = null
  params.include_debug = true
  overrideVectorThreshold.value = false
  overrideKeywordThreshold.value = false
  ElMessage.info('已恢复默认参数')
}

/** 只切换检索模式重跑，用于肉眼对比四路信号 */
async function runWithMode(mode: SearchMode): Promise<void> {
  params.mode = mode
  if (mode === 'bm25' || mode === 'vector') params.rerank = false
  await run()
}

onMounted(async () => {
  // 允许从别的页面带 ?kb_id= 跳过来
  const raw = route.query.kb_id
  if (typeof raw === 'string' && raw !== '' && Number.isFinite(Number(raw))) {
    selectedKbId.value = Number(raw)
  }
  if (kbStore.items.length === 0) {
    await kbStore.fetchList({ size: 100 }).catch(() => undefined)
  }
  if (selectedKbId.value === null && kbStore.items.length > 0) {
    selectedKbId.value = kbStore.items[0]?.id ?? null
  }
})
</script>

<template>
  <div class="search-page">
    <!-- 左：参数面板 -->
    <aside class="param-panel kf-card">
      <div class="kf-card-title">
        <h3>检索参数</h3>
        <el-button link size="small" @click="resetParams">恢复默认</el-button>
      </div>

      <div class="kf-card-body param-body">
        <div class="param-field">
          <label>知识库</label>
          <el-select v-model="selectedKbId" placeholder="请选择知识库" size="small" style="width: 100%">
            <el-option v-for="kb in kbStore.items" :key="kb.id" :label="kb.name" :value="kb.id" />
          </el-select>
        </div>

        <div class="param-field">
          <label>查询语句</label>
          <el-input
            v-model="params.query"
            type="textarea"
            :rows="2"
            resize="none"
            placeholder="例如：一线城市住宿标准是多少"
            @keydown.enter.exact.prevent="run"
          />
        </div>

        <el-divider content-position="left">召回</el-divider>

        <div class="param-field">
          <label>检索模式 mode</label>
          <el-select v-model="params.mode" size="small" style="width: 100%">
            <el-option label="纯向量 vector" value="vector" />
            <el-option label="纯关键词 bm25" value="bm25" />
            <el-option label="混合 hybrid" value="hybrid" />
            <el-option label="混合 + 重排 hybrid_rerank" value="hybrid_rerank" />
          </el-select>
        </div>

        <div v-if="showFusion" class="param-field">
          <label>融合方式 fusion</label>
          <el-radio-group v-model="params.fusion" size="small">
            <el-radio-button value="rrf">RRF</el-radio-button>
            <el-radio-button value="weighted">加权归一化</el-radio-button>
          </el-radio-group>
        </div>

        <div v-if="showAlpha" class="param-field">
          <label>向量权重 alpha = {{ params.alpha.toFixed(2) }}</label>
          <el-slider v-model="params.alpha" :min="0" :max="1" :step="0.05" size="small" />
          <div class="param-hint kf-muted">仅 weighted 生效；1 - alpha 是 BM25 权重</div>
        </div>

        <div class="param-field param-field--inline">
          <label>top_k</label>
          <el-input-number
            v-model="params.top_k"
            size="small"
            :min="1"
            :max="50"
            controls-position="right"
            @change="onTopKChange"
          />
        </div>

        <div class="param-field param-field--inline">
          <label>fetch_k</label>
          <el-input-number
            v-model="params.fetch_k"
            size="small"
            :min="params.top_k"
            :max="500"
            controls-position="right"
            @change="onFetchKChange"
          />
        </div>
        <div class="param-hint kf-muted">fetch_k 必须 ≥ top_k（先多召回再筛选），UI 会自动纠偏</div>

        <el-divider content-position="left">后处理</el-divider>

        <div class="param-field param-field--inline">
          <label>重排 rerank</label>
          <el-switch v-model="params.rerank" size="small" />
        </div>

        <div class="param-field param-field--inline">
          <label>分数断崖截断 autocut</label>
          <el-switch v-model="params.use_autocut" size="small" />
        </div>

        <el-divider content-position="left">相关性闸门</el-divider>

        <div class="param-field param-field--inline">
          <label>覆盖向量阈值</label>
          <el-switch v-model="overrideVectorThreshold" size="small" />
        </div>
        <div v-if="overrideVectorThreshold" class="param-field">
          <label>vector_threshold</label>
          <el-input-number
            v-model="params.vector_threshold"
            size="small"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            controls-position="right"
            style="width: 100%"
          />
        </div>
        <div v-else class="param-hint kf-muted">null → 用后端配置默认值（VECTOR_MIN_SCORE，默认 0.35）</div>

        <div class="param-field param-field--inline">
          <label>覆盖关键词阈值</label>
          <el-switch v-model="overrideKeywordThreshold" size="small" />
        </div>
        <div v-if="overrideKeywordThreshold" class="param-field">
          <label>keyword_threshold</label>
          <el-input-number
            v-model="params.keyword_threshold"
            size="small"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            controls-position="right"
            style="width: 100%"
          />
        </div>
        <div v-else class="param-hint kf-muted">null → 用后端配置默认值（KEYWORD_MIN_COVERAGE，默认 0.25）</div>

        <div class="param-field param-field--inline">
          <label>include_debug</label>
          <el-switch v-model="params.include_debug" size="small" />
        </div>

        <el-button type="primary" class="param-run" :loading="running" @click="run">
          执行检索（不花钱）
        </el-button>
        <div class="param-compare">
          <span class="kf-muted">快速对比：</span>
          <el-button link size="small" @click="runWithMode('vector')">vector</el-button>
          <el-button link size="small" @click="runWithMode('bm25')">bm25</el-button>
          <el-button link size="small" @click="runWithMode('hybrid')">hybrid</el-button>
          <el-button link size="small" @click="runWithMode('hybrid_rerank')">hybrid_rerank</el-button>
        </div>
      </div>
    </aside>

    <!-- 右：结果 -->
    <section class="result-panel">
      <ErrorBlock v-if="errorText" :message="errorText" @retry="run" />

      <EmptyState
        v-if="!response && !errorText"
        icon="🔍"
        title="还没有检索结果"
        description="左侧选知识库、填查询语句，点「执行检索」。这个接口不调用大模型，可以放心反复调参。"
      />

      <template v-else-if="response">
        <!-- 概览：耗时拆分 + 闸门 + debug 计数 -->
        <div class="overview">
          <div class="overview__card kf-card">
            <div class="kf-card-title"><h3>耗时拆分</h3></div>
            <div class="kf-card-body timing-body">
              <div v-for="row in timingRows" :key="row.label" class="timing-row">
                <span class="kf-muted">{{ row.label }}</span>
                <span class="kf-mono">{{ formatDuration(row.value) }}</span>
              </div>
              <div v-if="elapsedMs !== null" class="timing-row timing-row--total">
                <span class="kf-muted">浏览器观测</span>
                <span class="kf-mono">{{ formatDuration(elapsedMs) }}</span>
              </div>
            </div>
          </div>

          <div class="overview__card kf-card" :class="gate?.passed ? 'gate--pass' : 'gate--fail'">
            <div class="kf-card-title">
              <h3>相关性闸门</h3>
              <el-tag size="small" :type="gate?.passed ? 'success' : 'danger'" effect="dark">
                {{ gate?.passed ? '通过' : '拦截' }}
              </el-tag>
            </div>
            <div class="kf-card-body gate-body">
              <div class="gate-reason">
                <span class="kf-muted">判定原因</span>
                <code>{{ gate?.reason ?? '—' }}</code>
              </div>
              <div class="gate-grid">
                <div>
                  <span class="kf-muted">vector_threshold</span>
                  <span class="kf-mono">{{ scoreText(gate?.vector_threshold) }}</span>
                </div>
                <div>
                  <span class="kf-muted">keyword_threshold</span>
                  <span class="kf-mono">{{ scoreText(gate?.keyword_threshold) }}</span>
                </div>
                <div>
                  <span class="kf-muted">best_vector_score</span>
                  <span class="kf-mono">{{ scoreText(gate?.best_vector_score) }}</span>
                </div>
                <div>
                  <span class="kf-muted">best_keyword_coverage</span>
                  <span class="kf-mono">{{ scoreText(gate?.best_keyword_coverage) }}</span>
                </div>
              </div>
              <div v-if="gate && !gate.passed" class="gate-warn">
                两条阈值都不过 → 后端会短路、不调大模型，直接返回「知识库中没有找到相关内容」。
              </div>
            </div>
          </div>

          <div v-if="debugInfo" class="overview__card kf-card">
            <div class="kf-card-title"><h3>召回漏斗（debug）</h3></div>
            <div class="kf-card-body debug-body">
              <div class="debug-row">
                <span class="kf-muted">vector_candidates</span>
                <span class="kf-mono">{{ formatInt(debugInfo.vector_candidates) }}</span>
              </div>
              <div class="debug-row">
                <span class="kf-muted">bm25_candidates</span>
                <span class="kf-mono">{{ formatInt(debugInfo.bm25_candidates) }}</span>
              </div>
              <div class="debug-row">
                <span class="kf-muted">fused</span>
                <span class="kf-mono">{{ formatInt(debugInfo.fused) }}</span>
              </div>
              <div class="debug-row">
                <span class="kf-muted">after_autocut</span>
                <span class="kf-mono">{{ formatInt(debugInfo.after_autocut) }}</span>
              </div>
              <div class="debug-row">
                <span class="kf-muted">dropped_by_gate</span>
                <span class="kf-mono">{{ formatInt(debugInfo.dropped_by_gate) }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="kf-card">
          <div class="kf-card-title">
            <h3>
              命中结果
              <span class="kf-muted">
                （{{ hits.length }} 条 · {{ searchModeText(response.mode) }} · 返回耗时
                {{ formatDuration(response.latency_ms) }}）
              </span>
            </h3>
            <el-tag size="small" effect="plain" class="kf-mono">query: {{ response.query }}</el-tag>
          </div>

          <EmptyState v-if="hits.length === 0" icon="🫙" title="没有命中任何片段" description="可尝试放宽阈值、换成 hybrid 模式，或确认文档已完成向量化（状态 READY）。" />

          <el-table v-else :data="hits" size="small" class="hit-table">
            <el-table-column type="expand">
              <template #default="{ row }: { row: SearchHit }">
                <div class="hit-detail">
                  <div class="kf-muted hit-detail__label">content（送进上下文前会替换为 parent_content 父块）</div>
                  <pre class="kf-pre">{{ row.content }}</pre>
                  <div class="kf-muted hit-detail__label">snippet</div>
                  <pre class="kf-pre">{{ row.snippet }}</pre>
                </div>
              </template>
            </el-table-column>

            <el-table-column label="#" width="52">
              <template #default="{ row }: { row: SearchHit }">
                <el-tag size="small" effect="dark" type="primary">{{ row.rank }}</el-tag>
              </template>
            </el-table-column>

            <el-table-column label="文档 / 小节" min-width="200">
              <template #default="{ row }: { row: SearchHit }">
                <div class="hit-doc">
                  <span class="hit-doc__name kf-truncate" :title="row.doc_name">{{ row.doc_name }}</span>
                  <div class="hit-doc__meta kf-muted">
                    <el-tag v-if="row.page_no !== null" size="small" type="info" effect="plain">P{{ row.page_no }}</el-tag>
                    <span v-if="row.section_path" class="kf-truncate" :title="row.section_path">📑 {{ row.section_path }}</span>
                    <span class="kf-mono">chunk {{ row.chunk_id }}</span>
                  </div>
                </div>
              </template>
            </el-table-column>

            <el-table-column label="最终 score" width="110" align="right">
              <template #default="{ row }: { row: SearchHit }">
                <span class="score kf-mono" :class="scoreClass(row.score)">{{ scoreText(row.score) }}</span>
              </template>
            </el-table-column>

            <el-table-column label="vector_score" width="120" align="right">
              <template #default="{ row }: { row: SearchHit }">
                <span class="score kf-mono" :class="scoreClass(row.vector_score)">{{ scoreText(row.vector_score) }}</span>
              </template>
            </el-table-column>

            <el-table-column label="bm25_score" width="110" align="right">
              <template #default="{ row }: { row: SearchHit }">
                <span class="kf-mono">{{ row.bm25_score === null ? '—' : row.bm25_score.toFixed(2) }}</span>
              </template>
            </el-table-column>

            <el-table-column label="rerank_score" width="120" align="right">
              <template #default="{ row }: { row: SearchHit }">
                <span class="score kf-mono" :class="scoreClass(row.rerank_score)">{{ scoreText(row.rerank_score) }}</span>
              </template>
            </el-table-column>

            <el-table-column label="vector_rank" width="110" align="right">
              <template #default="{ row }: { row: SearchHit }">
                <el-tag v-if="row.vector_rank !== null" size="small" effect="plain" type="primary">{{ row.vector_rank }}</el-tag>
                <span v-else class="kf-muted">未召回</span>
              </template>
            </el-table-column>

            <el-table-column label="bm25_rank" width="100" align="right">
              <template #default="{ row }: { row: SearchHit }">
                <el-tag v-if="row.bm25_rank !== null" size="small" effect="plain" type="success">{{ row.bm25_rank }}</el-tag>
                <span v-else class="kf-muted">未召回</span>
              </template>
            </el-table-column>

            <el-table-column label="片段" min-width="200" show-overflow-tooltip>
              <template #default="{ row }: { row: SearchHit }">
                <span class="hit-snippet">{{ row.snippet }}</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </template>
    </section>
  </div>
</template>

<style scoped>
.search-page {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 16px;
  padding: 16px 20px 32px;
  align-items: start;
}

/* ---------------------------------------------------------------- 参数面板 */

.param-panel {
  position: sticky;
  top: 12px;
  max-height: calc(100vh - 90px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.param-body {
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.param-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.param-field--inline {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
}

.param-field label {
  font-size: 12.5px;
  color: var(--kf-text-regular);
}

.param-hint {
  font-size: 11.5px;
}

.param-run {
  width: 100%;
  margin-top: 6px;
}

.param-compare {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2px;
  font-size: 12px;
}

/* ---------------------------------------------------------------- 结果区 */

.result-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}

.overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}

.overview__card {
  min-width: 0;
}

.gate--pass {
  border-left: 3px solid var(--kf-success);
}

.gate--fail {
  border-left: 3px solid var(--kf-danger);
}

.timing-body,
.debug-body,
.gate-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 13px;
}

.timing-row,
.debug-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.timing-row--total {
  padding-top: 4px;
  border-top: 1px dashed var(--kf-border);
}

.gate-reason {
  display: flex;
  align-items: center;
  gap: 8px;
}

.gate-reason code {
  font-family: var(--kf-font-mono);
  font-size: 12px;
  color: var(--kf-primary);
  word-break: break-all;
}

.gate-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 12px;
  font-size: 12px;
}

.gate-grid > div {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.gate-warn {
  margin-top: 4px;
  padding: 6px 8px;
  border-radius: var(--kf-radius-sm);
  background: color-mix(in srgb, var(--kf-danger) 10%, transparent);
  color: var(--kf-danger);
  font-size: 12px;
}

.hit-table {
  width: 100%;
}

.hit-doc {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.hit-doc__name {
  font-size: 13px;
  font-weight: 600;
  color: var(--kf-text-primary);
}

.hit-doc__meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}

.hit-detail {
  padding: 8px 12px;
}

.hit-detail__label {
  font-size: 11.5px;
  margin: 6px 0 4px;
}

.hit-snippet {
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.score {
  font-weight: 600;
}

.score--high {
  color: var(--kf-success);
}

.score--mid {
  color: var(--kf-warning);
}

.score--low {
  color: var(--kf-danger);
}

.score--none {
  color: var(--kf-text-placeholder);
}
</style>
