/**
 * `useChatStream` —— 把 `api/sse.ts` 与 chat store 串起来，处理全部 9 种事件
 * （meta / trace / tool / sources / reflect / token / done / error / end）。
 *
 * 为什么单独抽 composable：
 * - 流式调用的状态机（建泡 → 追加 token → 回填 done → 收尾/取消）比一个请求复杂得多；
 * - 页面只关心「问一个问题」和「停止生成」两个动作；
 * - 事件到 store 的映射集中在一处，方便对着契约 5.6 节逐条核对。
 */

import { computed, onScopeDispose, type ComputedRef } from 'vue'

import { ApiError, isApiError } from '@/api/client'
import { isAbortLike, streamChat } from '@/api/sse'
import { useChatStore } from '@/stores/chat'
import type { ChatMode } from '@/types/models'
import type { SSEHandlers } from '@/types/sse'

export interface AskOptions {
  /** 知识库 id；null = 全局检索 */
  kbId?: number | null
  /** rag | agent */
  mode?: ChatMode
  topK?: number
  useRerank?: boolean
  useMemory?: boolean
}

export interface UseChatStreamOptions {
  /** 流正常结束（含被用户取消）后的回调 */
  onSettled?: (info: { aborted: boolean; error: string | null }) => void
  /** 收到 meta 帧（拿到 conversation_id）的回调，页面据此更新路由 */
  onMeta?: (conversationId: number) => void
  /** 出错回调；不提供时只写进 store */
  onError?: (message: string, error: unknown) => void
}

export interface UseChatStreamReturn {
  /** 发起一轮问答；返回是否成功完成 */
  ask: (question: string, options?: AskOptions) => Promise<boolean>
  /** 停止生成（用户点「停止生成」按钮） */
  stop: () => void
  /** 是否正在流式生成 */
  streaming: ComputedRef<boolean>
  /** 当前正在执行的图节点（AgentTracePanel 据此自动展开） */
  activeNode: ComputedRef<string | null>
  /** 本轮是否需要取消 */
  canStop: ComputedRef<boolean>
}

/** 把任意错误转成可读文案 */
function toMessage(error: unknown): string {
  if (isApiError(error)) return error.toDisplay()
  if (error instanceof Error) return error.message
  return String(error)
}

export function useChatStream(options: UseChatStreamOptions = {}): UseChatStreamReturn {
  const chat = useChatStore()
  const streaming = computed(() => chat.streaming)
  const activeNode = computed(() => chat.activeNode)
  const canStop = computed(() => chat.streaming)

  // 组件卸载时把还在跑的流掐掉，避免「离开页面还在扣 token」
  onScopeDispose(() => {
    if (chat.streaming) chat.abortStream()
  })

  async function ask(question: string, askOptions: AskOptions = {}): Promise<boolean> {
    const text = question.trim()
    if (text === '') return false
    if (chat.streaming) {
      console.warn('[knowflow] 上一轮还在生成中，忽略本次提问')
      return false
    }

    const {
      kbId = null,
      mode = 'agent',
      topK = 5,
      useRerank = true,
      useMemory = true,
    } = askOptions

    // 1) 乐观插入用户提问 + 建立空的助手气泡，让界面立刻有反馈
    chat.pushUserMessage(text, chat.currentConversationId)
    chat.ensureAssistantMessage(chat.currentConversationId)

    const controller = new AbortController()
    chat.beginStream(controller)

    let aborted = false
    let failure: string | null = null
    const startedAt = Date.now()

    const handlers: SSEHandlers = {
      // 第一帧：建立会话上下文
      meta: (data) => {
        chat.applyMeta(data)
        if (data.conversation_id) options.onMeta?.(data.conversation_id)
      },
      // 图节点开始/结束 → 「思考过程」时间线
      trace: (data) => {
        chat.pushTraceNode({
          kind: 'node',
          node: data.node,
          status: data.status,
          duration_ms: data.duration_ms,
          detail: data.detail,
          at: Date.now() - startedAt,
        })
      },
      // 工具调用（检索）
      tool: (data) => {
        chat.pushTool({
          kind: 'tool',
          name: data.name,
          args: data.args,
          result_count: data.result_count,
          duration_ms: data.duration_ms,
          at: Date.now() - startedAt,
        })
      },
      // 引用来源：先于 token，用户立刻能看到「有据可依」
      sources: (data) => {
        chat.setSources(data.sources)
      },
      // 自我反思（Self-RAG）
      reflect: (data) => {
        chat.pushReflect({
          kind: 'reflect',
          round: data.round,
          passed: data.passed,
          unsupported: data.unsupported,
          action: data.action,
          at: Date.now() - startedAt,
        })
      },
      // 增量文本
      token: (data) => {
        chat.appendToken(data.text)
      },
      // 收尾统计（完整 ChatResponse）
      done: (data) => {
        chat.applyDone(data)
      },
      // 错误帧
      error: (data) => {
        failure = data.message
        chat.applyStreamError(data.code, data.message, data.request_id)
      },
      // 关闭流：正常路径下 done 帧已经收尾，这里只清掉「当前节点」高亮
      end: () => {
        chat.activeNode = null
      },
    }

    try {
      await streamChat({
        body: {
          question: text,
          kb_id: kbId,
          // 会话 id：null 时后端新建会话，并用 meta 帧告知
          conversation_id: chat.currentConversationId,
          mode,
          top_k: topK,
          use_rerank: useRerank,
          use_memory: useMemory,
        },
        handlers,
        signal: controller.signal,
      })
      return failure === null
    } catch (error) {
      if (isAbortLike(error)) {
        aborted = true
        chat.markStopped()
        return false
      }

      failure = toMessage(error)
      // 连接中断 / 非 2xx（响应体是错误信封 JSON 而不是 event-stream）
      const apiError = error instanceof ApiError ? error : null
      chat.applyStreamError(
        apiError?.code ?? 'STREAM_ERROR',
        apiError?.message ?? failure,
        apiError?.requestId ?? '',
      )
      options.onError?.(failure, error)
      return false
    } finally {
      chat.endStream()
      options.onSettled?.({ aborted, error: failure })
    }
  }

  function stop(): void {
    if (!chat.streaming) return
    chat.abortStream()
  }

  return { ask, stop, streaming, activeNode, canStop }
}
