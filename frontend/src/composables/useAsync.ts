/**
 * 异步三态封装：`loading` / `error` / `data`。
 *
 * 用的地方很多（列表加载、详情加载、动作提交），
 * 统一成「不抛异常、把错误落进 error 字段」的形态，页面里就不用到处写 try/catch。
 */

import { ref, type Ref } from 'vue'

import { describeError } from '@/api/client'

export interface UseAsyncOptions<T> {
  /** 初始数据 */
  initial?: T
  /** 出错时是否把错误信息打到 console（默认 true，便于排障） */
  logError?: boolean
  /** 成功回调 */
  onSuccess?: (data: T) => void
  /** 失败回调 */
  onError?: (message: string, error: unknown) => void
}

export interface UseAsyncReturn<T, A extends unknown[]> {
  data: Ref<T | undefined>
  loading: Ref<boolean>
  error: Ref<string>
  /** 最近一次的错误原文（含 code / request_id，便于排障） */
  rawError: Ref<unknown>
  /** 执行；返回数据或 undefined（失败时不抛，除非 throwOnError） */
  run: (...args: A) => Promise<T | undefined>
  /** 手动清空错误 */
  clearError: () => void
  /** 手动重置（清数据 + 清错误） */
  reset: (value?: T) => void
}

/**
 * @param task 要执行的异步函数
 * @param options 初始值与回调
 */
export function useAsync<T, A extends unknown[] = []>(
  task: (...args: A) => Promise<T>,
  options: UseAsyncOptions<T> = {},
): UseAsyncReturn<T, A> {
  const data = ref<T | undefined>(options.initial) as Ref<T | undefined>
  const loading = ref(false)
  const error = ref('')
  const rawError = ref<unknown>(null)

  async function run(...args: A): Promise<T | undefined> {
    loading.value = true
    error.value = ''
    rawError.value = null
    try {
      const result = await task(...args)
      data.value = result
      options.onSuccess?.(result)
      return result
    } catch (err) {
      const message = describeError(err)
      error.value = message
      rawError.value = err
      if (options.logError !== false) console.error('[knowflow] 异步任务失败：', err)
      options.onError?.(message, err)
      return undefined
    } finally {
      loading.value = false
    }
  }

  function clearError(): void {
    error.value = ''
    rawError.value = null
  }

  function reset(value?: T): void {
    data.value = value
    error.value = ''
    rawError.value = null
    loading.value = false
  }

  return { data, loading, error, rawError, run, clearError, reset }
}
