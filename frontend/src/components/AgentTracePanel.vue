<script setup lang="ts">
/**
 * 「思考过程」折叠面板 —— 本项目的差异化亮点。
 *
 * 时间线展示三类事件（契约 5.6）：
 * - `trace`：LangGraph 图节点（analyze / retrieve / grade / rewrite / generate / reflect…）+ 耗时；
 * - `tool`：工具调用（检索）+ 命中数；
 * - `reflect`：第几轮反思 + 是否通过 + 未支撑的句子。
 *
 * 交互要求：**默认折叠**；流式过程中自动展开「当前节点」那一项。
 */
import { computed, ref, watch } from 'vue'

import type { AgentTraceItem } from '@/stores/chat'
import type { UnsupportedClaim } from '@/types/sse'
import { formatDuration } from '@/utils/format'

const props = withDefaults(
  defineProps<{
    /** 时间线（按到达顺序） */
    items: AgentTraceItem[]
    /** 当前正在执行的节点名；变化时自动展开对应项 */
    activeNode?: string | null
    /** 是否正在流式生成（决定默认是否展开整个面板） */
    streaming?: boolean
    /** 是否默认展开面板 */
    defaultExpanded?: boolean
    /** 标题 */
    title?: string
  }>(),
  {
    activeNode: null,
    streaming: false,
    defaultExpanded: false,
    title: '思考过程',
  },
)

/** 面板整体是否展开 */
const panelExpanded = ref<boolean>(props.defaultExpanded)
/** 单个条目的展开态（key → 是否展开） */
const expandedKeys = ref<Set<string>>(new Set())

/** LangGraph 节点中文说明（与 docs/00 的图节点一一对应） */
const NODE_LABELS: Record<string, string> = {
  analyze: '规划：判断问题类型与是否需要检索',
  retrieve: '工具：混合检索（向量 + BM25）',
  grade: '反思①：逐条判断召回是否相关（CRAG）',
  rewrite: '改写：去口语化 / 换同义词 / 拆子问题',
  generate: '生成：只依据 [n] 编号的上下文作答',
  reflect: '反思②：检查每句是否有引用支撑（Self-RAG）',
}

/** 节点耗时占最慢节点的比例，用于画横条 */
const maxNodeDuration = computed<number>(() => {
  const durations = props.items
    .filter((item): item is Extract<AgentTraceItem, { kind: 'node' }> => item.kind === 'node')
    .map((item) => item.duration_ms)
  return durations.length > 0 ? Math.max(...durations, 1) : 1
})

function keyOf(item: AgentTraceItem, index: number): string {
  switch (item.kind) {
    case 'node':
      return `node-${item.node}-${index}`
    case 'tool':
      return `tool-${item.name}-${index}`
    default:
      return `reflect-${item.round}-${index}`
  }
}

function toggleKey(key: string): void {
  const next = new Set(expandedKeys.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  expandedKeys.value = next
}

/** 事件是否有可展开的细节（detail / 工具参数 / 未支撑句子） */
function hasDetail(item: AgentTraceItem): boolean {
  if (item.kind === 'node') return item.detail !== null && item.detail !== undefined
  if (item.kind === 'tool') {
    if (item.args === null || item.args === undefined) return false
    if (typeof item.args === 'object') return Object.keys(item.args).length > 0
    return String(item.args) !== ''
  }
  return item.unsupported.length > 0
}

/** 把 detail / args 序列化成可读 JSON */
function detailText(item: AgentTraceItem): string {
  if (item.kind === 'node') {
    if (typeof item.detail === 'string') return item.detail
    try {
      return JSON.stringify(item.detail, null, 2)
    } catch {
      return String(item.detail)
    }
  }
  if (item.kind === 'tool') {
    if (typeof item.args === 'string') return item.args
    try {
      return JSON.stringify(item.args, null, 2)
    } catch {
      return String(item.args)
    }
  }
  // 未支撑的句子：后端是 `[{sentence, reason}]`，渲染成「句子 —— 原因」
  return item.unsupported
    .map((claim: UnsupportedClaim) => {
      const sentence = claim.sentence || JSON.stringify(claim)
      return claim.reason ? `${sentence}  ——  ${claim.reason}` : sentence
    })
    .join('\n')
}

function nodeLabel(name: string): string {
  return NODE_LABELS[name] ?? ''
}

/**
 * 流式过程中自动展开「当前节点」：
 * activeNode 变化时把对应条目展开（用户手动收起过的不再强行展开，故只在变化时处理一次）。
 */
watch(
  () => props.activeNode,
  (node) => {
    if (!node) return
    panelExpanded.value = true
    const next = new Set(expandedKeys.value)
    next.add(`node-${node}-0`)
    expandedKeys.value = next
  },
)

/** 有内容时流式开始也要展开面板，让用户看到「它在思考」 */
watch(
  () => props.streaming,
  (value) => {
    if (value && props.items.length > 0) panelExpanded.value = true
  },
)

const nodeCount = computed<number>(() => props.items.filter((item) => item.kind === 'node').length)
const toolCount = computed<number>(() => props.items.filter((item) => item.kind === 'tool').length)
const reflectCount = computed<number>(() => props.items.filter((item) => item.kind === 'reflect').length)

/** 总耗时：各节点耗时之和（仅作参考，与端到端延迟不同） */
const totalNodeMs = computed<number>(() =>
  props.items.reduce((sum, item) => (item.kind === 'node' ? sum + item.duration_ms : sum), 0),
)
</script>

<template>
  <div class="trace-panel kf-card">
    <div class="trace-panel__head" @click="panelExpanded = !panelExpanded">
      <span class="trace-panel__arrow" :class="{ 'trace-panel__arrow--open': panelExpanded }">▶</span>
      <span class="trace-panel__title">{{ title }}</span>
      <el-tag v-if="nodeCount > 0" size="small" effect="plain">节点 {{ nodeCount }}</el-tag>
      <el-tag v-if="toolCount > 0" size="small" type="success" effect="plain">工具 {{ toolCount }}</el-tag>
      <el-tag v-if="reflectCount > 0" size="small" type="warning" effect="plain">反思 {{ reflectCount }}</el-tag>
      <span v-if="totalNodeMs > 0" class="kf-muted trace-panel__total">节点合计 {{ formatDuration(totalNodeMs) }}</span>
      <span class="kf-spacer" />
      <span v-if="streaming" class="trace-panel__live">● 进行中</span>
    </div>

    <div v-if="panelExpanded" class="trace-panel__body">
      <div v-if="items.length === 0" class="kf-empty-hint trace-panel__empty">
        暂无思考过程。Agent 模式下会显示 analyze / retrieve / grade / rewrite / generate / reflect 各节点。
      </div>

      <ol v-else class="trace-timeline">
        <li v-for="(item, index) in items" :key="keyOf(item, index)" class="trace-item" :class="`trace-item--${item.kind}`">
          <div class="trace-item__line">
            <span class="trace-item__dot" :class="{ 'trace-item__dot--active': item.kind === 'node' && item.node === activeNode }" />
            <span class="trace-item__offset kf-mono">+{{ formatDuration(item.at) }}</span>
          </div>

          <div class="trace-item__content">
            <!-- trace：图节点 -->
            <template v-if="item.kind === 'node'">
              <div class="trace-item__row">
                <el-tag size="small" effect="dark" type="primary">节点</el-tag>
                <span class="trace-item__name kf-mono">{{ item.node }}</span>
                <span class="kf-muted trace-item__desc">{{ nodeLabel(item.node) }}</span>
                <span class="kf-spacer" />
                <el-tag
                  size="small"
                  effect="plain"
                  :type="item.status === 'error' ? 'danger' : 'success'"
                >
                  {{ item.status }}
                </el-tag>
                <span class="kf-mono trace-item__duration">{{ formatDuration(item.duration_ms) }}</span>
              </div>
              <div class="trace-item__bar">
                <span
                  class="trace-item__bar-fill"
                  :style="{ width: `${Math.max(2, (item.duration_ms / maxNodeDuration) * 100)}%` }"
                />
              </div>
            </template>

            <!-- tool：工具调用 -->
            <template v-else-if="item.kind === 'tool'">
              <div class="trace-item__row">
                <el-tag size="small" effect="dark" type="success">工具</el-tag>
                <span class="trace-item__name kf-mono">{{ item.name }}</span>
                <span class="trace-item__hits">命中 {{ item.result_count }} 条</span>
                <span class="kf-spacer" />
                <span class="kf-mono trace-item__duration">{{ formatDuration(item.duration_ms) }}</span>
              </div>
            </template>

            <!-- reflect：自我反思 -->
            <template v-else>
              <div class="trace-item__row">
                <el-tag size="small" effect="dark" :type="item.passed ? 'success' : 'danger'">反思</el-tag>
                <span class="trace-item__name">第 {{ item.round }} 轮</span>
                <el-tag size="small" :type="item.passed ? 'success' : 'danger'" effect="plain">
                  {{ item.passed ? '通过' : '未通过' }}
                </el-tag>
                <el-tag v-if="item.action" size="small" type="info" effect="plain">{{ item.action }}</el-tag>
                <span v-if="item.unsupported.length > 0" class="trace-item__unsupported">
                  未支撑句子 {{ item.unsupported.length }} 条
                </span>
              </div>
              <ul v-if="item.unsupported.length > 0" class="trace-item__sentences">
                <li v-for="(claim, claimIndex) in item.unsupported" :key="claimIndex">
                  <span class="trace-item__sentence">{{ claim.sentence }}</span>
                  <span v-if="claim.reason" class="kf-muted"> —— {{ claim.reason }}</span>
                </li>
              </ul>
            </template>

            <el-button
              v-if="hasDetail(item)"
              link
              size="small"
              class="trace-item__toggle"
              @click="toggleKey(keyOf(item, index))"
            >
              {{ expandedKeys.has(keyOf(item, index)) ? '收起细节' : '查看细节' }}
            </el-button>
            <pre v-if="hasDetail(item) && expandedKeys.has(keyOf(item, index))" class="kf-pre trace-item__detail">{{ detailText(item) }}</pre>
          </div>
        </li>
      </ol>
    </div>
  </div>
</template>

<style scoped>
.trace-panel {
  overflow: hidden;
}

.trace-panel__head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid transparent;
}

.trace-panel__head:hover {
  background: var(--kf-bg-hover);
}

.trace-panel__arrow {
  display: inline-block;
  font-size: 10px;
  color: var(--kf-text-placeholder);
  transition: transform 0.2s ease;
}

.trace-panel__arrow--open {
  transform: rotate(90deg);
}

.trace-panel__title {
  font-weight: 600;
  font-size: 14px;
}

.trace-panel__total {
  font-size: 12px;
}

.trace-panel__live {
  font-size: 12px;
  color: var(--kf-success);
}

.trace-panel__body {
  border-top: 1px solid var(--kf-border);
  padding: 8px 14px 12px;
}

.trace-panel__empty {
  padding: 8px 0;
}

.trace-timeline {
  margin: 0;
  padding: 0;
  list-style: none;
}

.trace-item {
  display: flex;
  gap: 10px;
  padding: 8px 0;
}

.trace-item__line {
  position: relative;
  flex: none;
  width: 56px;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  padding-right: 14px;
}

/* 时间线的竖线与节点圆点 */
.trace-item__line::after {
  content: '';
  position: absolute;
  top: 4px;
  bottom: -12px;
  right: 5px;
  width: 1px;
  background: var(--kf-border);
}

.trace-item:last-child .trace-item__line::after {
  display: none;
}

.trace-item__dot {
  position: absolute;
  top: 3px;
  right: 1px;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  border: 2px solid var(--kf-bg-card);
}

.trace-item--node .trace-item__dot {
  background: var(--kf-span-node);
}

.trace-item--tool .trace-item__dot {
  background: var(--kf-span-tool);
}

.trace-item--reflect .trace-item__dot {
  background: var(--kf-span-llm);
}

.trace-item__dot--active {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--kf-span-node) 35%, transparent);
  animation: trace-pulse 1.2s ease-in-out infinite;
}

@keyframes trace-pulse {
  50% {
    box-shadow: 0 0 0 6px color-mix(in srgb, var(--kf-span-node) 18%, transparent);
  }
}

.trace-item__offset {
  font-size: 11px;
  color: var(--kf-text-placeholder);
  white-space: nowrap;
}

.trace-item__content {
  flex: 1 1 auto;
  min-width: 0;
}

.trace-item__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.trace-item__name {
  font-size: 13px;
  font-weight: 600;
  color: var(--kf-text-primary);
}

.trace-item__desc {
  font-size: 12px;
}

.trace-item__hits {
  font-size: 12px;
  color: var(--kf-success);
}

.trace-item__duration {
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.trace-item__bar {
  margin-top: 4px;
  height: 4px;
  border-radius: 2px;
  background: var(--kf-bg-subtle);
  overflow: hidden;
}

.trace-item__bar-fill {
  display: block;
  height: 100%;
  border-radius: 2px;
  background: var(--kf-span-node);
  transition: width 0.3s ease;
}

.trace-item__unsupported {
  font-size: 12px;
  color: var(--kf-danger);
}

.trace-item__sentences {
  margin: 4px 0 0;
  padding-left: 18px;
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.trace-item__toggle {
  margin-top: 2px;
}

.trace-item__detail {
  margin-top: 4px;
  max-height: 200px;
}
</style>
