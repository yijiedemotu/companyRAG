<script setup lang="ts">
/**
 * 智能问答页 `/chat` 与 `/chat/:conversationId`。
 *
 * 三段式布局：
 * - 左：会话列表（新建 / 切换 / 删除）
 * - 中：消息流 + 输入区（模式、top_k、重排、记忆开关、停止生成）
 * - 右：「引用来源」卡片列表 + 「思考过程」折叠面板（AgentTracePanel）
 *
 * 引用角标交互：MessageBubble 里的 `[n]` 点击后滚动并高亮到右侧对应 CitationCard。
 */
import { computed, nextTick, onMounted, ref, watch, type ComponentPublicInstance } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { describeError } from '@/api/client'
import AgentTracePanel from '@/components/AgentTracePanel.vue'
import CitationCard from '@/components/CitationCard.vue'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LoadingBlock from '@/components/LoadingBlock.vue'
import MessageBubble from '@/components/MessageBubble.vue'
import { useChatStream } from '@/composables/useChatStream'
import { useChatStore } from '@/stores/chat'
import { useKbStore } from '@/stores/kb'
import type { ChatMode, Source } from '@/types/models'
import { formatFromNow, formatDuration, formatUsd, formatInt } from '@/utils/format'

const route = useRoute()
const router = useRouter()
const chat = useChatStore()
const kbStore = useKbStore()

/* ---------------------------------------------------------------- 输入与参数 */

const question = ref('')
const mode = ref<ChatMode>('agent')
const topK = ref(5)
const useRerank = ref(true)
const useMemory = ref(true)
const selectedKbId = ref<number | null>(kbStore.currentKbId)

const messagesEl = ref<HTMLDivElement>()
const citationRefs = new Map<number, HTMLElement>()
const activeRank = ref<number | null>(null)
const feedbackMap = ref<Record<number, number>>({})
const rightPanelVisible = ref(true)

const routeConversationId = computed<number | null>(() => {
  const raw = route.params.conversationId
  if (typeof raw !== 'string' || raw === '') return null
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? parsed : null
})

const { ask, stop, streaming, activeNode } = useChatStream({
  onMeta: (conversationId) => {
    // 新建会话时把 URL 补上，刷新后还能回到同一会话
    if (routeConversationId.value !== conversationId) {
      void router.replace(`/chat/${conversationId}`)
    }
    void chat.fetchConversations({ kbId: selectedKbId.value ?? undefined })
  },
  onError: (message) => ElMessage.error(message),
  onSettled: ({ aborted, error }) => {
    if (aborted) ElMessage.info('已停止生成')
    if (error) ElMessage.error(error)
    void nextTick(scrollToBottom)
  },
})

/* ---------------------------------------------------------------- 会话列表 */

const currentMessages = computed(() => chat.messages)
const lastSources = computed<Source[]>(() => {
  const last = chat.lastAssistant
  return last?.sources ?? []
})
const lastTrace = computed(() => chat.lastAssistant?.trace ?? [])

async function loadConversationList(): Promise<void> {
  try {
    await chat.fetchConversations({ kbId: selectedKbId.value ?? undefined })
  } catch {
    /* 列表失败不阻塞主流程 */
  }
}

async function openConversation(id: number): Promise<void> {
  if (chat.streaming) chat.abortStream()
  if (routeConversationId.value === id && chat.messages.length > 0) return
  await router.push(`/chat/${id}`)
}

async function handleNewConversation(): Promise<void> {
  if (chat.streaming) chat.abortStream()
  chat.startNewChat()
  if (routeConversationId.value !== null) await router.push('/chat')
}

async function handleDeleteConversation(id: number): Promise<void> {
  try {
    await ElMessageBox.confirm('删除后该会话的消息与引用将不可见（软删除）。确定继续吗？', '删除会话', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
  } catch {
    return
  }
  try {
    await chat.removeConversation(id)
    ElMessage.success('已删除')
    if (routeConversationId.value === id) await router.push('/chat')
  } catch (error) {
    ElMessage.error(describeError(error))
  }
}

/* ---------------------------------------------------------------- 发问 */

async function handleSend(): Promise<void> {
  const text = question.value.trim()
  if (text === '') return
  if (chat.streaming) {
    ElMessage.warning('正在生成中，请先停止或等待完成')
    return
  }

  question.value = ''
  const ok = await ask(text, {
    kbId: selectedKbId.value,
    mode: mode.value,
    topK: topK.value,
    useRerank: useRerank.value,
    useMemory: useMemory.value,
  })

  // 首轮问答后服务端已造好会话，刷新列表让标题及时更新
  if (ok) await loadConversationList()
}

function handleKeydown(event: KeyboardEvent): void {
  // Enter 发送，Shift+Enter 换行
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    void handleSend()
  }
}

/* ---------------------------------------------------------------- 引用交互 */

/** 模板 ref 回调：把每张引用卡片的根 DOM 存起来，供角标点击时滚动定位 */
function setCitationRef(rank: number, el: Element | ComponentPublicInstance | null): void {
  const node = el instanceof HTMLElement ? el : el && '$el' in el ? (el.$el as unknown) : null
  if (node instanceof HTMLElement) citationRefs.set(rank, node)
  else citationRefs.delete(rank)
}

async function focusCitation(rank: number): Promise<void> {
  rightPanelVisible.value = true
  activeRank.value = rank
  await nextTick()
  const el = citationRefs.get(rank)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  } else {
    ElMessage.info(`引用 [${rank}] 不在当前来源列表里`)
  }
  // 高亮 2.4 秒后自动褪去
  window.setTimeout(() => {
    if (activeRank.value === rank) activeRank.value = null
  }, 2400)
}

async function handleFeedback(messageId: number, rating: number): Promise<void> {
  let comment: string | undefined
  if (rating === -1) {
    // 后端 `FeedbackRequest` 有一条契约之外的校验：rating=-1 时 comment 必填。
    // 点踩是改进检索质量最重要的信号，只说「没用」无法定位是召回错了还是答案编了。
    try {
      const result = await ElMessageBox.prompt(
        '点踩需要说明原因（例如：召回的片段答的是另一个问题 / 答案有编造）：',
        '告诉我们哪里不对',
        {
          confirmButtonText: '提交',
          cancelButtonText: '取消',
          inputType: 'textarea',
          inputPlaceholder: '一句话就够',
          inputValidator: (value: string) => (value && value.trim() !== '' ? true : '原因不能为空'),
        },
      )
      comment = result.value.trim()
    } catch {
      // 用户取消点踩
      return
    }
  }

  try {
    await chat.submitFeedback(messageId, rating, comment)
    feedbackMap.value = { ...feedbackMap.value, [messageId]: rating }
    ElMessage.success(rating === 1 ? '感谢反馈：有用' : '感谢反馈：没用')
  } catch (error) {
    ElMessage.error(describeError(error))
  }
}

function scrollToBottom(): void {
  const el = messagesEl.value
  if (el) el.scrollTop = el.scrollHeight
}

/* ---------------------------------------------------------------- 生命周期 */

/** 消息内容 / 思考过程变化时保持滚动在底部（流式期间跟随） */
watch(
  () => [chat.messages.length, chat.lastAssistant?.content.length ?? 0, chat.lastAssistant?.trace.length ?? 0],
  () => {
    void nextTick(scrollToBottom)
  },
)

/** 路由参数变化 → 载入对应会话；没有 id 就是新会话 */
watch(
  routeConversationId,
  async (id) => {
    if (id === null) {
      chat.startNewChat()
      return
    }
    if (chat.currentConversationId === id && chat.messages.length > 0) return
    try {
      await chat.loadMessages(id)
      await nextTick(scrollToBottom)
    } catch (error) {
      ElMessage.error(describeError(error))
    }
  },
  { immediate: true },
)

/** 切换「当前 KB」时同步下拉并刷新会话列表 */
watch(selectedKbId, (value) => {
  kbStore.setCurrentKbId(value)
  void loadConversationList()
})

onMounted(async () => {
  if (kbStore.items.length === 0) await kbStore.fetchList({ size: 100 }).catch(() => undefined)
  await loadConversationList()
})
</script>

<template>
  <div class="chat-page">
    <!-- 左：会话列表 -->
    <aside class="chat-sidebar">
      <div class="chat-sidebar__head">
        <el-button type="primary" class="chat-sidebar__new" @click="handleNewConversation">
          ＋ 新建会话
        </el-button>
      </div>

      <div class="chat-sidebar__filter">
        <el-select v-model="selectedKbId" placeholder="全部知识库（全局检索）" clearable size="small">
          <el-option v-for="kb in kbStore.items" :key="kb.id" :label="kb.name" :value="kb.id" />
        </el-select>
      </div>

      <div v-loading="chat.conversationsLoading" class="chat-sidebar__list">
        <EmptyState
          v-if="chat.conversations.length === 0"
          icon="💬"
          title="还没有会话"
          description="在右侧输入问题即可开始（Agent 模式会展示完整思考过程）。"
        />
        <div
          v-for="item in chat.conversations"
          :key="item.id"
          class="conv-item"
          :class="{ 'conv-item--active': item.id === chat.currentConversationId }"
          @click="openConversation(item.id)"
        >
          <div class="conv-item__main">
            <div class="conv-item__title kf-truncate" :title="item.title">{{ item.title }}</div>
            <div class="conv-item__meta kf-muted">
              {{ item.message_count }} 条 · {{ formatFromNow(item.updated_at) }}
              <el-tag size="small" effect="plain" class="conv-item__mode">{{ item.mode }}</el-tag>
            </div>
          </div>
          <el-button
            link
            size="small"
            class="conv-item__delete"
            @click.stop="handleDeleteConversation(item.id)"
          >
            🗑
          </el-button>
        </div>
      </div>
    </aside>

    <!-- 中：消息流 + 输入区 -->
    <section class="chat-main">
      <div ref="messagesEl" class="chat-thread">
        <EmptyState
          v-if="currentMessages.length === 0"
          icon="🕸️"
          title="问点什么吧"
          description="Agent 模式：analyze → retrieve → grade →（rewrite → retrieve）* → generate → reflect。每一步都会展示在右侧「思考过程」里。"
        />

        <LoadingBlock v-if="chat.messagesLoading && currentMessages.length === 0" height="180px" />

        <MessageBubble
          v-for="message in currentMessages"
          :key="message.localId"
          :message="message"
          :active-rank="activeRank"
          :streaming="streaming && message.localId === chat.streamMessageId"
          :feedback-rating="message.id !== null ? (feedbackMap[message.id] ?? 0) : 0"
          @citation-click="focusCitation"
          @feedback="(rating: number) => message.id !== null && handleFeedback(message.id, rating)"
          @show-trace="rightPanelVisible = true"
          @retry="handleSend"
        />
      </div>

      <div class="chat-composer">
        <div class="chat-composer__controls">
          <el-radio-group v-model="mode" size="small">
            <el-radio-button value="agent">Agent</el-radio-button>
            <el-radio-button value="rag">RAG 快路径</el-radio-button>
          </el-radio-group>

          <span class="kf-muted chat-composer__label">top_k</span>
          <el-input-number v-model="topK" size="small" :min="1" :max="20" controls-position="right" style="width: 100px" />

          <el-checkbox v-model="useRerank" size="small">重排</el-checkbox>
          <el-checkbox v-model="useMemory" size="small">多轮记忆</el-checkbox>

          <span class="kf-spacer" />
          <el-button link size="small" @click="rightPanelVisible = !rightPanelVisible">
            {{ rightPanelVisible ? '隐藏右侧面板' : '显示右侧面板' }}
          </el-button>
        </div>

        <div class="chat-composer__input">
          <el-input
            v-model="question"
            type="textarea"
            :rows="3"
            resize="none"
            maxlength="4000"
            show-word-limit
            placeholder="例如：一线城市住宿标准是多少？（Enter 发送，Shift+Enter 换行）"
            @keydown="handleKeydown"
          />
          <div class="chat-composer__actions">
            <el-button v-if="streaming" type="danger" @click="stop">停止生成</el-button>
            <el-button v-else type="primary" :disabled="question.trim() === ''" @click="handleSend">
              发送
            </el-button>
          </div>
        </div>
      </div>
    </section>

    <!-- 右：引用来源 + 思考过程 -->
    <aside v-if="rightPanelVisible" class="chat-right">
      <AgentTracePanel
        :items="lastTrace"
        :active-node="activeNode"
        :streaming="streaming"
        :default-expanded="false"
      />

      <div class="chat-right__citations kf-card">
        <div class="kf-card-title">
          <h3>引用来源</h3>
          <span class="kf-muted">{{ lastSources.length }} 条</span>
        </div>
        <div class="kf-card-body citations-body">
          <EmptyState
            v-if="lastSources.length === 0"
            icon="📎"
            title="暂无引用"
            description="sources 事件先于 token 到达，收到引用后这里会立刻显示。"
          />
          <CitationCard
            v-for="source in lastSources"
            :key="`${source.rank}-${source.chunk_id}`"
            :ref="(el) => setCitationRef(source.rank, el as Element | ComponentPublicInstance | null)"
            :source="source"
            :active="activeRank === source.rank"
            compact
          />
        </div>
      </div>

      <div v-if="chat.lastAssistant" class="chat-right__stats kf-card">
        <div class="kf-card-title"><h3>本轮统计</h3></div>
        <div class="kf-card-body stats-body">
          <div class="stats-row">
            <span class="kf-muted">端到端延迟</span>
            <span>{{ formatDuration(chat.lastAssistant.latencyMs) }}</span>
          </div>
          <div class="stats-row">
            <span class="kf-muted">token（输入/输出）</span>
            <span>{{ formatInt(chat.lastAssistant.promptTokens) }} / {{ formatInt(chat.lastAssistant.completionTokens) }}</span>
          </div>
          <div class="stats-row">
            <span class="kf-muted">成本</span>
            <span>{{ formatUsd(chat.lastAssistant.costUsd) }}</span>
          </div>
          <div class="stats-row">
            <span class="kf-muted">检索轮数</span>
            <span>{{ chat.lastAssistant.retrievalRounds }}</span>
          </div>
          <div class="stats-row">
            <span class="kf-muted">反思</span>
            <span>
              <el-tag v-if="chat.lastAssistant.reflectPassed === null" size="small" type="info" effect="plain">未执行</el-tag>
              <el-tag v-else size="small" :type="chat.lastAssistant.reflectPassed ? 'success' : 'danger'" effect="plain">
                {{ chat.lastAssistant.reflectPassed ? '通过' : '未通过' }}
              </el-tag>
            </span>
          </div>
          <div v-if="chat.lastAssistant.rewrittenQueries.length > 0" class="stats-rewrites">
            <div class="kf-muted">改写后的查询</div>
            <ul>
              <li v-for="(item, index) in chat.lastAssistant.rewrittenQueries" :key="index">{{ item }}</li>
            </ul>
          </div>
          <div v-if="chat.lastAssistant.traceId" class="stats-trace">
            <el-button link size="small" type="primary" @click="router.push(`/obs/traces/${chat.lastAssistant?.traceId}`)">
              查看链路详情 →
            </el-button>
          </div>
        </div>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.chat-page {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 360px;
  height: 100%;
  overflow: hidden;
}

/* ---------------------------------------------------------------- 左侧会话列表 */

.chat-sidebar {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--kf-border);
  background: var(--kf-bg-card);
  overflow: hidden;
}

.chat-sidebar__head {
  padding: 12px;
  border-bottom: 1px solid var(--kf-border);
}

.chat-sidebar__new {
  width: 100%;
}

.chat-sidebar__filter {
  padding: 8px 12px;
  border-bottom: 1px solid var(--kf-border);
}

.chat-sidebar__list {
  flex: 1 1 auto;
  overflow: auto;
  padding: 6px;
}

.conv-item {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 10px;
  border-radius: var(--kf-radius-sm);
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.conv-item:hover {
  background: var(--kf-bg-hover);
}

.conv-item--active {
  background: color-mix(in srgb, var(--kf-primary) 14%, transparent);
}

.conv-item__main {
  flex: 1 1 auto;
  min-width: 0;
}

.conv-item__title {
  font-size: 13px;
  color: var(--kf-text-primary);
}

.conv-item__meta {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
}

.conv-item__mode {
  transform: scale(0.85);
}

.conv-item__delete {
  opacity: 0;
  transition: opacity 0.15s ease;
}

.conv-item:hover .conv-item__delete {
  opacity: 1;
}

/* ---------------------------------------------------------------- 中间消息流 */

.chat-main {
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}

.chat-thread {
  flex: 1 1 auto;
  overflow: auto;
  padding: 16px 24px;
}

.chat-composer {
  border-top: 1px solid var(--kf-border);
  background: var(--kf-bg-card);
  padding: 8px 16px 12px;
}

.chat-composer__controls {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 8px;
  flex-wrap: wrap;
}

.chat-composer__label {
  font-size: 12px;
  margin-left: 4px;
}

.chat-composer__input {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

.chat-composer__actions {
  flex: none;
}

/* ---------------------------------------------------------------- 右侧面板 */

.chat-right {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 12px;
  border-left: 1px solid var(--kf-border);
  background: var(--kf-bg-body);
  overflow: auto;
}

.chat-right__citations {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.citations-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 46vh;
  overflow: auto;
}

.stats-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 13px;
}

.stats-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.stats-rewrites {
  font-size: 12px;
}

.stats-rewrites ul {
  margin: 4px 0 0;
  padding-left: 18px;
  color: var(--kf-text-secondary);
}

.stats-trace {
  margin-top: 4px;
}

@media (max-width: 1400px) {
  .chat-page {
    grid-template-columns: 220px minmax(0, 1fr) 320px;
  }
}
</style>
