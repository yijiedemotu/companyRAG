/**
 * 认证 store：token、user、登录/登出、localStorage 持久化。
 *
 * 契约（docs/01 第七节）：
 * - token 存 localStorage，key = `knowflow.token`（这个名字是契约，不能改）；
 * - 401 处理：清 token → 跳登录页，并 toast「登录已过期，请重新登录」。
 *
 * 401 只弹一次的实现放在 api/client.ts（`triggerUnauthorizedOnce`），
 * 这里只注册回调，避免 client ↔ store 循环依赖。
 */

import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import { clearToken, getToken, onUnauthorized, resetUnauthorizedFlag, setToken } from '@/api/client'
import { login as loginApi, me as meApi, register as registerApi } from '@/api/auth'
import type { LoginRequest, RegisterRequest } from '@/types/dto'
import type { TokenOut, UserOut } from '@/types/models'

const USER_KEY = 'knowflow.user'
const UNAUTHORIZED_MESSAGE = '登录已过期，请重新登录'

/** 跳登录页：整页替换，避免 401 后还在业务路由上渲染半截页面 */
function redirectToLogin(): void {
  if (typeof window === 'undefined') return
  if (window.location.pathname === '/login') return
  window.location.replace(`/login?redirect=${encodeURIComponent(window.location.pathname)}`)
}

/** 读取本地缓存的用户信息（只为刷新页面时少一次 /auth/me 的空窗期，不作鉴权依据） */
function loadCachedUser(): UserOut | null {
  try {
    const raw = window.localStorage.getItem(USER_KEY)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed === 'object' && parsed !== null && 'username' in parsed) {
      return parsed as UserOut
    }
    return null
  } catch {
    return null
  }
}

function cacheUser(user: UserOut | null): void {
  try {
    if (user) window.localStorage.setItem(USER_KEY, JSON.stringify(user))
    else window.localStorage.removeItem(USER_KEY)
  } catch {
    /* 隐私模式下忽略 */
  }
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string>(getToken())
  const user = ref<UserOut | null>(loadCachedUser())
  const loading = ref(false)
  /** 登录页要展示的一次性提示（例如 401 跳转过来） */
  const notice = ref<string>('')
  /** 已在别处消费过 401 提示？用于「只提示一次」 */
  let unauthorizedNotified = false

  const isAuthenticated = computed<boolean>(() => token.value !== '')
  const isAdmin = computed<boolean>(() => user.value?.role === 'admin')
  const displayName = computed<string>(() => user.value?.display_name || user.value?.username || '未登录')

  /** 落地 token（内存 + localStorage 双向） */
  function applyToken(next: string): void {
    token.value = next
    if (next) setToken(next)
    else clearToken()
  }

  function applyTokenOut(payload: TokenOut): void {
    applyToken(payload.access_token)
    user.value = payload.user
    cacheUser(payload.user)
    // 登录成功要复位 401 标记，否则第二次过期不会被处理
    unauthorizedNotified = false
    resetUnauthorizedFlag()
  }

  async function login(payload: LoginRequest): Promise<UserOut> {
    loading.value = true
    try {
      const result = await loginApi(payload)
      applyTokenOut(result)
      notice.value = ''
      return result.user
    } finally {
      loading.value = false
    }
  }

  async function register(payload: RegisterRequest): Promise<UserOut> {
    loading.value = true
    try {
      const result = await registerApi(payload)
      applyTokenOut(result)
      notice.value = ''
      return result.user
    } finally {
      loading.value = false
    }
  }

  /** 拉取当前用户；失败时清 token（说明 token 已失效） */
  async function fetchMe(): Promise<UserOut | null> {
    if (!token.value) return null
    try {
      const result = await meApi()
      user.value = result
      cacheUser(result)
      return result
    } catch (error) {
      applyToken('')
      user.value = null
      cacheUser(null)
      throw error
    }
  }

  /** 本地登出：清 token、清用户 */
  function clearSession(): void {
    applyToken('')
    user.value = null
    cacheUser(null)
  }

  function logout(): void {
    clearSession()
    notice.value = ''
    redirectToLogin()
  }

  /** 401 全局回调：清 token → 跳登录 → 提示（只做一次） */
  function handleUnauthorized(): void {
    if (unauthorizedNotified) return
    unauthorizedNotified = true
    clearSession()
    notice.value = UNAUTHORIZED_MESSAGE
    redirectToLogin()
  }

  function consumeNotice(): string {
    const message = notice.value
    notice.value = ''
    return message
  }

  // 注册 401 处理器（模块级 Set，重复调用也只会注册一次同名字段的效果）
  onUnauthorized(handleUnauthorized)

  return {
    token,
    user,
    loading,
    notice,
    isAuthenticated,
    isAdmin,
    displayName,
    login,
    register,
    fetchMe,
    logout,
    clearSession,
    handleUnauthorized,
    consumeNotice,
  }
})
