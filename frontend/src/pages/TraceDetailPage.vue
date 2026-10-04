<script setup lang="ts">
/**
 * 链路详情页 `/obs/traces/:traceId`：span 瀑布图。
 *
 * 契约 5.9：`GET /obs/traces/{trace_id}` → `{trace, spans, messages}`。
 *
 * 瀑布图用**绝对定位 div** 实现（比 ECharts custom series 简单可靠得多）：
 * - 横轴 = 时间，靠 `start_offset_ms`（相对 trace 起点的偏移）定位 left；
 * - 每行宽度 = `duration_ms`；
 * - 颜色按 `span_type`（node/llm/retrieval/db/tool）区分；
 * - hover 显示 input_json / output_json。
 */
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import { describeError } from '@/api/client'
import { getTraceDetail } from '@/api/obs'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LoadingBlock from '@/components/LoadingBlock.vue'
import type { SpanType, TraceDetailOut, TraceSpanOut } from '@/types/models'
import {
  formatDateTime,
  formatDuration,
  formatInt,
  formatUsd,
  usdToCny,
  formatCny,
} from '@/utils/format'

const props = defineProps<{ traceId: string }>()

const router = useRouter()

const detail = ref<TraceDetailOut | null>(null)
const loading = ref(false)
const errorText = ref('')
const expandedSpanId = ref<number | null>(null)

const trace = computed(() => detail.value?.trace ?? null)
const spans = computed<TraceSpanOut[]>(() => detail.value?.spans ?? [])
const messages = computed(() => detail.value?.messages ?? [])

/** trace 总时长：优先用 trace.latency_ms，否则取 span 结束时间的最大值 */
const totalMs = computed<number>(() => {
  if (trace.value && trace.value.latency_ms > 0) return trace.value.latency_ms
  const ends = spans.value.map((span) => span.start_offset_ms + span.duration_ms)
  return ends.length > 0 ? Math.max(...ends, 1) : 1
})

/** 时间轴刻度（5 段，含末尾） */
const ticks = computed<number[]>(() => {
  const total = totalMs.value
  return Array.from({ length: 6 }, (_item, index) => Math.round((total / 5) * index))
})

/** 最慢的 span（一眼看出瓶颈） */
const slowestSpan = computed<TraceSpanOut | null>(() => {
  if (spans.value.length === 0) return null
  return spans.value.reduce((slowest, span) => (span.duration_ms > slowest.duration_ms ? span : slowest))
})

const SPAN_TYPE_TEXT: Record<SpanType, string> = {
  node: '图节点',
  llm: '大模型',
  retrieval: '检索',
  db: '数据库',
  tool: '工具',
}

function spanTypeText(type: string): string {
  return SPAN_TYPE_TEXT[type as SpanType] ?? type
}

/** 条形图的 left / width 百分比 */
function barStyle(span: TraceSpanOut): Record<string, string> {
  const total = totalMs.value || 1
  const left = Math.min(100, (span.start_offset_ms / total) * 100)
  // 极短 span 也要看得见，最窄给 0.6%
  const width = Math.max(0.6, Math.min(100 - left, (span.duration_ms / total) * 100))
  return {
    left: `${left}%`,
    width: `${width}%`,
  }
}

function toggleDetail(span: TraceSpanOut): void {
  expandedSpanId.value = expandedSpanId.value === span.id ? null : span.id
}

function jsonText(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

const spanTypeLegend = computed<SpanType[]>(() => {
  const types = new Set<SpanType>()
  spans.value.forEach((span) => types.add(span.span_type))
  return [...types]
})

async function load(): Promise<void> {
  loading.value = true
  errorText.value = ''
  try {
    detail.value = await getTraceDetail(props.traceId)
  } catch (error) {
    errorText.value = describeError(error)
    detail.value = null
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="kf-page">
    <div class="kf-page-header">
      <div class="kf-row">
        <el-button link @click="router.push('/obs')">← 可观测</el-button>
        <h2 class="kf-page-title">链路详情</h2>
        <span class="kf-mono trace-id">{{ traceId }}</span>
        <el-tag v-if="trace" size="small" :type="trace.status === 'ok' ? 'success' : 'danger'" effect="plain">
          {{ trace.status }}
        </el-tag>
      </div>
      <div class="kf-row">
        <el-button @click="load">刷新</el-button>
        <el-button v-if="trace?.conversation_id" @click="router.push(`/chat/${trace.conversation_id}`)">
          回到会话
        </el-button>
      </div>
    </div>

    <ErrorBlock v-if="errorText" :message="errorText" @retry="load" />

    <LoadingBlock v-else-if="loading && !detail" height="220px" />

    <template v-else-if="trace">
      <!-- trace 概览 -->
      <div class="kf-card">
        <div class="kf-card-title">
          <h3>请求概览</h3>
          <span class="kf-muted">{{ formatDateTime(trace.created_at) }}</span>
        </div>
        <div class="kf-card-body overview">
          <div class="overview__item">
            <span class="kf-muted">名称 / 模式</span>
            <span>{{ trace.name }} · {{ trace.mode ?? '—' }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">端到端延迟</span>
            <span class="kf-mono">{{ formatDuration(trace.latency_ms) }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">检索 / 重排 / 生成</span>
            <span class="kf-mono">
              {{ formatDuration(trace.retrieval_ms) }} / {{ formatDuration(trace.rerank_ms) }} /
              {{ formatDuration(trace.generate_ms) }}
            </span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">LLM 调用次数</span>
            <span>{{ formatInt(trace.llm_calls) }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">token（输入 / 输出）</span>
            <span>{{ formatInt(trace.prompt_tokens) }} / {{ formatInt(trace.completion_tokens) }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">成本</span>
            <span>{{ formatUsd(trace.cost_usd) }} · ≈ {{ formatCny(usdToCny(trace.cost_usd)) }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">检索轮数 / 引用数</span>
            <span>{{ trace.retrieval_rounds }} / {{ trace.source_count }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">是否拒答</span>
            <el-tag v-if="trace.refusal" size="small" type="warning" effect="plain">拒答</el-tag>
            <span v-else>否</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">request_id</span>
            <span class="kf-mono">{{ trace.request_id ?? '—' }}</span>
          </div>
          <div class="overview__item">
            <span class="kf-muted">用户 / 知识库 / 会话</span>
            <span class="kf-mono">
              {{ trace.user_id ?? '—' }} / {{ trace.kb_id ?? '—' }} / {{ trace.conversation_id ?? '—' }}
            </span>
          </div>
        </div>
        <el-alert
          v-if="trace.error"
          class="trace-error"
          type="error"
          :closable="false"
          show-icon
          title="该链路发生错误"
          :description="trace.error"
        />
      </div>

      <!-- 瀑布图 -->
      <div class="kf-card waterfall-card">
        <div class="kf-card-title">
          <h3>Span 瀑布图</h3>
          <div class="kf-row">
            <template v-for="type in spanTypeLegend" :key="type">
              <span class="legend">
                <span class="legend__dot" :class="`span-bar--${type}`" />
                {{ spanTypeText(type) }}
              </span>
            </template>
            <span v-if="slowestSpan" class="kf-muted slowest">
              最慢：<span class="kf-mono">{{ slowestSpan.name }}</span>
              （{{ formatDuration(slowestSpan.duration_ms) }}，占
              {{ ((slowestSpan.duration_ms / totalMs) * 100).toFixed(1) }}%）
            </span>
          </div>
        </div>

        <EmptyState v-if="spans.length === 0" icon="🧭" title="没有 span 数据" description="该 trace 可能被采样丢弃，或埋点未写入 trace_spans 表。" />

        <div v-else class="waterfall">
          <!-- 时间轴 -->
          <div class="waterfall__axis">
            <div class="waterfall__axis-label kf-muted">span</div>
            <div class="waterfall__axis-track">
              <span
                v-for="(tick, index) in ticks"
                :key="index"
                class="waterfall__tick"
                :style="{ left: `${(index / (ticks.length - 1)) * 100}%` }"
              >
                {{ tick }}ms
              </span>
            </div>
          </div>

          <div
            v-for="span in spans"
            :key="span.id"
            class="waterfall__row"
            :class="{ 'waterfall__row--expanded': expandedSpanId === span.id }"
          >
            <div class="waterfall__name" :title="span.name" @click="toggleDetail(span)">
              <span class="waterfall__seq kf-mono">{{ span.seq }}</span>
              <span class="kf-truncate">{{ span.name }}</span>
              <el-tag v-if="span.status === 'error'" size="small" type="danger" effect="plain">err</el-tag>
            </div>

            <div class="waterfall__track" @click="toggleDetail(span)">
              <el-tooltip placement="top" :show-after="120">
                <template #content>
                  <div class="tip">
                    <div><strong>{{ span.name }}</strong> · {{ spanTypeText(span.span_type) }}</div>
                    <div>start_offset_ms: {{ span.start_offset_ms }} ms</div>
                    <div>duration_ms: {{ span.duration_ms }} ms</div>
                    <div v-if="span.error" class="tip__error">error: {{ span.error }}</div>
                    <div class="tip__label">input_json</div>
                    <pre class="tip__json">{{ jsonText(span.input_json) }}</pre>
                    <div class="tip__label">output_json</div>
                    <pre class="tip__json">{{ jsonText(span.output_json) }}</pre>
                  </div>
                </template>
                <div
                  class="span-bar"
                  :class="[`span-bar--${span.span_type}`, { 'span-bar--error': span.status === 'error' }]"
                  :style="barStyle(span)"
                >
                  <span class="span-bar__label">{{ formatDuration(span.duration_ms) }}</span>
                </div>
              </el-tooltip>
            </div>

            <!-- 展开：input/output JSON -->
            <div v-if="expandedSpanId === span.id" class="waterfall__detail">
              <div class="waterfall__detail-grid">
                <div>
                  <div class="kf-muted detail-label">input_json</div>
                  <pre class="kf-pre">{{ jsonText(span.input_json) }}</pre>
                </div>
                <div>
                  <div class="kf-muted detail-label">output_json</div>
                  <pre class="kf-pre">{{ jsonText(span.output_json) }}</pre>
                </div>
              </div>
              <div v-if="span.error" class="detail-error">error：{{ span.error }}</div>
              <div class="kf-muted detail-label">
                span_type: {{ span.span_type }} · offset {{ span.start_offset_ms }}ms · 时长
                {{ span.duration_ms }}ms · 序号 {{ span.seq }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 关联消息 -->
      <div v-if="messages.length > 0" class="kf-card">
        <div class="kf-card-title">
          <h3>关联消息</h3>
          <span class="kf-muted">{{ messages.length }} 条</span>
        </div>
        <div class="messages">
          <div v-for="message in messages" :key="message.id" class="message-item">
            <div class="message-item__head">
              <el-tag size="small" :type="message.role === 'user' ? 'info' : 'primary'" effect="plain">
                {{ message.role }}
              </el-tag>
              <span class="kf-muted kf-mono">
                {{ formatDuration(message.latency_ms) }} ·
                {{ formatInt(message.prompt_tokens) }}/{{ formatInt(message.completion_tokens) }} token ·
                {{ formatUsd(message.cost_usd) }}
              </span>
              <span class="kf-spacer" />
              <span class="kf-muted">{{ formatDateTime(message.created_at) }}</span>
            </div>
            <p class="message-item__content">{{ message.content }}</p>
            <div v-if="message.citations.length > 0" class="message-item__citations">
              <el-tag v-for="citation in message.citations" :key="citation.rank" size="small" effect="plain">
                [{{ citation.rank }}] {{ citation.doc_name ?? `chunk ${citation.chunk_id}` }}
                <template v-if="citation.page_no !== null"> · P{{ citation.page_no }}</template>
              </el-tag>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.trace-id {
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px 20px;
}

.overview__item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 13px;
}

.trace-error {
  margin: 0 16px 16px;
}

.waterfall-card {
  margin-top: 16px;
}

.legend {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.legend__dot {
  width: 10px;
  height: 10px;
  border-radius: 2px;
}

.slowest {
  font-size: 12px;
}

.waterfall {
  padding: 12px 16px 16px;
  overflow-x: auto;
}

.waterfall__axis,
.waterfall__row {
  display: grid;
  grid-template-columns: 180px minmax(400px, 1fr);
  align-items: center;
  gap: 8px;
}

.waterfall__axis {
  padding-bottom: 4px;
  border-bottom: 1px solid var(--kf-border);
  margin-bottom: 4px;
}

.waterfall__axis-track {
  position: relative;
  height: 16px;
}

.waterfall__tick {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  font-size: 11px;
  color: var(--kf-text-placeholder);
  white-space: nowrap;
}

.waterfall__row {
  padding: 3px 0;
  border-radius: var(--kf-radius-sm);
}

.waterfall__row:hover {
  background: var(--kf-bg-hover);
}

.waterfall__row--expanded {
  background: var(--kf-bg-subtle);
}

.waterfall__name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12.5px;
  min-width: 0;
  cursor: pointer;
}

.waterfall__seq {
  flex: none;
  width: 20px;
  color: var(--kf-text-placeholder);
  font-size: 11px;
}

.waterfall__track {
  position: relative;
  height: 22px;
  border-left: 1px solid var(--kf-border);
  cursor: pointer;
}

/* 每行一条水平网格线，帮助对齐时间 */
.waterfall__track::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: linear-gradient(to right, var(--kf-border) 1px, transparent 1px);
  background-size: 20% 100%;
  opacity: 0.5;
  pointer-events: none;
}

.span-bar {
  position: absolute;
  top: 3px;
  height: 16px;
  border-radius: 3px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 4px;
  min-width: 3px;
  opacity: 0.92;
  transition: opacity 0.15s ease;
}

.span-bar:hover {
  opacity: 1;
}

.span-bar__label {
  font-size: 10px;
  color: #fff;
  font-family: var(--kf-font-mono);
  white-space: nowrap;
  overflow: hidden;
  text-shadow: 0 1px 1px rgba(0, 0, 0, 0.35);
}

/* 颜色按 span_type 区分 */
.span-bar--node {
  background: var(--kf-span-node);
}

.span-bar--llm {
  background: var(--kf-span-llm);
}

.span-bar--retrieval {
  background: var(--kf-span-retrieval);
}

.span-bar--db {
  background: var(--kf-span-db);
}

.span-bar--tool {
  background: var(--kf-span-tool);
}

.span-bar--error {
  background: var(--kf-danger);
  background-image: repeating-linear-gradient(
    45deg,
    rgba(255, 255, 255, 0.35) 0 4px,
    transparent 4px 8px
  );
}

.waterfall__detail {
  grid-column: 1 / -1;
  padding: 8px 4px 10px 28px;
}

.waterfall__detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}

.detail-label {
  font-size: 11.5px;
  margin-bottom: 4px;
}

.detail-error {
  margin-top: 6px;
  color: var(--kf-danger);
  font-size: 12px;
}

/* tooltip 里的 JSON 展示 */
.tip {
  max-width: 460px;
  font-size: 12px;
  line-height: 1.5;
}

.tip__label {
  margin-top: 4px;
  opacity: 0.7;
}

.tip__json {
  margin: 2px 0 0;
  max-height: 160px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: var(--kf-font-mono);
  font-size: 11px;
}

.tip__error {
  color: #fca5a5;
}

.messages {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 16px 16px;
}

.message-item {
  padding: 10px 12px;
  border: 1px solid var(--kf-border);
  border-radius: var(--kf-radius-sm);
}

.message-item__head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.message-item__content {
  margin: 6px 0 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--kf-text-regular);
  white-space: pre-wrap;
  word-break: break-word;
}

.message-item__citations {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}
</style>
