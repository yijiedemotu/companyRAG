<script setup lang="ts">
/**
 * 消息气泡。
 *
 * 核心要求：助手回答里的 `[1]` `[2]` 要渲染成**可点击的小角标**，点击后滚动/高亮到对应引用卡片。
 * 实现方式是模板循环渲染「文本片段」与「角标」两种节点（`splitCitations`），
 * **坚决不用 v-html 拼字符串**，从根子上避免 XSS。
 */
import { computed, ref } from 'vue'

import type { ChatMessage } from '@/stores/chat'
import { collectRanks, splitCitations } from '@/utils/citation'
import { formatDateTime, formatDuration, formatUsd, formatInt } from '@/utils/format'

const props = withDefaults(
  defineProps<{
    message: ChatMessage
    /** 当前高亮的引用编号（点击角标后由页面传入） */
    activeRank?: number | null
    /** 是否正在流式生成这条消息 */
    streaming?: boolean
    /** 是否显示反馈按钮（只在已落库的助手消息上显示） */
    feedbackEnabled?: boolean
    /** 当前反馈状态：1 有用 / -1 没用 / 0 未评价 */
    feedbackRating?: number
  }>(),
  {
    activeRank: null,
    streaming: false,
    feedbackEnabled: true,
    feedbackRating: 0,
  },
)

const emit = defineEmits<{
  /** 点击了角标 `[n]` */
  citationClick: [rank: number]
  /** 提交反馈 */
  feedback: [rating: number]
  /** 请求展示这条消息的思考过程 */
  showTrace: []
  /** 重试这条消息（出错的助手消息） */
  retry: []
}>()

const isUser = computed<boolean>(() => props.message.role === 'user')
const isAssistant = computed<boolean>(() => props.message.role === 'assistant')

/** 引用角标：只有落在 sources 里的 rank 才变成可点击角标（避免把 `[2024]` 误判成引用） */
const validRanks = computed<Set<number>>(() => collectRanks(props.message.sources))
const nodes = computed(() => splitCitations(props.message.content, validRanks.value))

const hasTrace = computed<boolean>(() => props.message.trace.length > 0)
const showMeta = computed<boolean>(
  () => isAssistant.value && props.message.status !== 'streaming' && props.message.id !== null,
)

const copied = ref(false)

async function copyContent(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.message.content)
    copied.value = true
    window.setTimeout(() => {
      copied.value = false
    }, 1500)
  } catch {
    /* 剪贴板不可用则忽略 */
  }
}

const statusText = computed<string>(() => {
  switch (props.message.status) {
    case 'streaming':
      return '生成中…'
    case 'stopped':
      return '已停止生成'
    case 'error':
      return '生成失败'
    default:
      return ''
  }
})
</script>

<template>
  <div class="bubble" :class="[isUser ? 'bubble--user' : 'bubble--assistant']">
    <div class="bubble__avatar" aria-hidden="true">{{ isUser ? '🧑' : '🤖' }}</div>

    <div class="bubble__main">
      <div class="bubble__meta">
        <span class="bubble__role">{{ isUser ? '我' : 'KnowFlow' }}</span>
        <span class="kf-muted bubble__time">{{ formatDateTime(message.createdAt) }}</span>
        <template v-if="isAssistant && message.model">
          <el-tag size="small" effect="plain" type="info">{{ message.model }}</el-tag>
        </template>
        <el-tag v-if="message.refusal" size="small" type="warning" effect="plain">已拒答</el-tag>
        <el-tag v-if="message.mode" size="small" effect="plain">{{ message.mode }}</el-tag>
        <span v-if="statusText" class="bubble__status kf-muted">{{ statusText }}</span>
      </div>

      <div class="bubble__body">
        <!-- 助手回答里的 [n] 渲染成角标；文本节点与角标节点分开渲染，不走 v-html -->
        <template v-if="isAssistant">
          <template v-for="node in nodes" :key="node.key">
            <span v-if="node.type === 'text'" class="bubble__text">{{ node.text }}</span>
            <button
              v-else
              type="button"
              class="bubble__citation"
              :class="{ 'bubble__citation--active': node.rank === activeRank }"
              :aria-label="`跳转到引用 ${node.rank}`"
              :title="`查看引用 [${node.rank}]`"
              @click="emit('citationClick', node.rank)"
            >
              {{ node.rank }}
            </button>
          </template>
          <span v-if="streaming" class="bubble__caret" aria-hidden="true" />
          <span v-if="!message.content && streaming" class="kf-muted">正在思考…</span>
        </template>

        <!-- 用户提问按原样展示（白空格保留） -->
        <pre v-else class="bubble__pre">{{ message.content }}</pre>
      </div>

      <div v-if="message.errorText" class="bubble__error">
        <span aria-hidden="true">⚠️</span>
        <span>{{ message.errorText }}</span>
        <el-button link type="primary" size="small" @click="emit('retry')">重试</el-button>
      </div>

      <div v-if="isAssistant && message.status !== 'streaming'" class="bubble__actions">
        <el-button v-if="hasTrace" link size="small" @click="emit('showTrace')">思考过程（{{ message.trace.length }}）</el-button>
        <el-button v-if="message.sources.length > 0" link size="small" @click="emit('citationClick', message.sources[0]?.rank ?? 1)">
          引用来源（{{ message.sources.length }}）
        </el-button>
        <el-button link size="small" @click="copyContent">{{ copied ? '已复制' : '复制' }}</el-button>
        <template v-if="feedbackEnabled && message.id !== null">
          <el-button
            link
            size="small"
            :type="feedbackRating === 1 ? 'primary' : 'default'"
            @click="emit('feedback', 1)"
          >
            👍 有用
          </el-button>
          <el-button
            link
            size="small"
            :type="feedbackRating === -1 ? 'danger' : 'default'"
            @click="emit('feedback', -1)"
          >
            👎 没用
          </el-button>
        </template>
      </div>

      <div v-if="showMeta" class="bubble__stats kf-muted">
        <span v-if="message.latencyMs !== null">延迟 {{ formatDuration(message.latencyMs) }}</span>
        <span v-if="message.usage">
          token 输入 {{ formatInt(message.promptTokens) }} / 输出 {{ formatInt(message.completionTokens) }}
        </span>
        <span v-if="message.costUsd > 0">成本 {{ formatUsd(message.costUsd) }}</span>
        <span v-if="message.retrievalRounds > 0">检索 {{ message.retrievalRounds }} 轮</span>
        <span v-if="message.reflectPassed === false" class="bubble__warn">反思未通过</span>
        <span v-if="message.traceId" class="kf-mono bubble__trace">trace: {{ message.traceId }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bubble {
  display: flex;
  gap: 10px;
  padding: 12px 0;
}

.bubble--user {
  flex-direction: row-reverse;
}

.bubble__avatar {
  flex: none;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: var(--kf-bg-subtle);
  font-size: 15px;
}

.bubble__main {
  max-width: min(760px, 82%);
  min-width: 0;
}

.bubble--user .bubble__main {
  text-align: right;
}

.bubble__meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.bubble--user .bubble__meta {
  justify-content: flex-end;
}

.bubble__role {
  font-weight: 600;
  color: var(--kf-text-regular);
}

.bubble__time {
  font-size: 11px;
}

.bubble__status {
  font-size: 11px;
  font-style: italic;
}

.bubble__body {
  margin-top: 4px;
  padding: 10px 14px;
  border-radius: var(--kf-radius);
  background: var(--kf-bg-card);
  border: 1px solid var(--kf-border);
  font-size: 14px;
  line-height: 1.75;
  color: var(--kf-text-primary);
  text-align: left;
  word-break: break-word;
}

.bubble--user .bubble__body {
  background: var(--kf-primary);
  border-color: var(--kf-primary);
  color: #fff;
}

.bubble__text {
  white-space: pre-wrap;
}

.bubble__pre {
  margin: 0;
  font: inherit;
  white-space: pre-wrap;
  word-break: break-word;
}

/* 引用角标：可点击 */
.bubble__citation {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  margin: 0 2px;
  padding: 0 4px;
  border: none;
  border-radius: var(--kf-radius-sm);
  background: var(--kf-citation-bg);
  color: var(--kf-citation-text);
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
  vertical-align: super;
  cursor: pointer;
  transition: transform 0.15s ease, background-color 0.15s ease;
}

.bubble__citation:hover {
  transform: translateY(-1px);
  background: color-mix(in srgb, var(--kf-citation-text) 30%, var(--kf-citation-bg));
}

.bubble__citation--active {
  background: var(--kf-citation-active);
  color: #1f2937;
}

.bubble__caret {
  display: inline-block;
  width: 7px;
  height: 15px;
  margin-left: 2px;
  background: var(--kf-primary);
  vertical-align: text-bottom;
  animation: bubble-blink 1s steps(2, start) infinite;
}

@keyframes bubble-blink {
  50% {
    opacity: 0;
  }
}

.bubble__error {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  padding: 6px 10px;
  border-radius: var(--kf-radius-sm);
  background: color-mix(in srgb, var(--kf-danger) 10%, transparent);
  color: var(--kf-danger);
  font-size: 12px;
}

.bubble__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}

.bubble--user .bubble__actions {
  justify-content: flex-end;
}

.bubble__stats {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 2px;
  font-size: 11px;
}

.bubble--user .bubble__stats {
  justify-content: flex-end;
}

.bubble__warn {
  color: var(--kf-warning);
}

.bubble__trace {
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
