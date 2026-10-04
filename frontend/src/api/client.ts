/**
 * axios 实例与统一错误处理。
 *
 * 契约（docs/01 第七节）：
 * - 鉴权：`Authorization: Bearer <token>`，token 存 localStorage，key = `knowflow.token`；
 * - 所有非 2xx 都是统一错误信封 `{error:{code,message,detail,request_id,timestamp}}`；
 * - 401：清 token → 跳登录页 + toast「登录已过期」，且**只处理一次**（并发请求不能弹多个 toast）。
 *
 * 前端把错误统一解包成 `ApiError`，页面 toast 时把 `request_id` 带出来便于排障。
 */

import axios, {
  AxiosError,
  AxiosHeaders,
  type AxiosInstance,
  type AxiosResponse,
  type InternalAxiosRequestConfig,
} from 'axios'

import type { ApiErrorBody, ApiErrorEnvelope } from '@/types/models'

/* ------------------------------------------------------------------ 常量 */

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

const TOKEN_KEY = 'knowflow.token'

/* ------------------------------------------------------------------ 错误类型 */

/**
 * 统一的 API 错误。
 * `code` / `request_id` 来自后端错误信封，排障时把它们展示到 toast 里。
 */
export class ApiError extends Error {
  /** 后端错误码，如 KB_NOT_FOUND / UNAUTHORIZED / REQUEST_VALIDATION_ERROR */
  readonly code: string
  /** HTTP 状态码；网络层失败时为 0 */
  readonly status: number
  /** 后端 request_id，排障用 */
  readonly requestId: string | null
  /** 附加细节（Pydantic 的字段错误数组会放在这里） */
  readonly detail: unknown
  readonly timestamp: string | null

  constructor(init: {
    code: string
    message: string
    status: number
    requestId?: string | null
    detail?: unknown
    timestamp?: string | null
  }) {
    super(init.message)
    this.name = 'ApiError'
    this.code = init.code
    this.status = init.status
    this.requestId = init.requestId ?? null
    this.detail = init.detail ?? null
    this.timestamp = init.timestamp ?? null
  }

  /** 面向用户的展示文案：带 request_id 便于排障 */
  toDisplay(): string {
    return this.requestId ? `${this.message}（request_id: ${this.requestId}）` : this.message
  }
}

/** 类型守卫：判断任意 catch 到的值是否是 ApiError */
export function isApiError(value: unknown): value is ApiError {
  return value instanceof ApiError
}

/** 把任意错误转成可展示的中文文案 */
export function describeError(value: unknown): string {
  if (isApiError(value)) return value.toDisplay()
  if (value instanceof Error) return value.message
  return String(value)
}

/* ------------------------------------------------------------------ token 读写 */

export function getToken(): string {
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    // 隐私模式下 localStorage 可能抛异常，降级为「无 token」
    return ''
  }
}

export function setToken(token: string): void {
  try {
    window.localStorage.setItem(TOKEN_KEY, token)
  } catch {
    /* 忽略：无法持久化时仅本次会话有效 */
  }
}

export function clearToken(): void {
  try {
    window.localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* 忽略 */
  }
}

/* ------------------------------------------------------------------ 401 全局处理 */

/** 401 只处理一次：这个标记在第一次 401 时置位，避免并发请求弹多个 toast */
let handling401 = false

const unauthorizedHandlers = new Set<() => void>()

/**
 * 注册 401 处理器（auth store 初始化时调用）。
 * 用「订阅」而不是直接 import store，是为了避免 client ↔ store 循环依赖。
 */
export function onUnauthorized(handler: () => void): void {
  unauthorizedHandlers.add(handler)
}

/** 401 处理入口：只执行一次，直到 resetUnauthorizedFlag() 被调用 */
function triggerUnauthorizedOnce(): void {
  if (handling401) return
  handling401 = true
  unauthorizedHandlers.forEach((handler) => {
    try {
      handler()
    } catch (err) {
      console.error('[knowflow] 401 处理器执行失败', err)
    }
  })
}

/** 登录成功后复位标记，否则第二次过期不会再被处理 */
export function resetUnauthorizedFlag(): void {
  handling401 = false
}

/* ------------------------------------------------------------------ 错误解包 */

/** 判断是否是统一错误信封 */
function isErrorEnvelope(body: unknown): body is ApiErrorEnvelope {
  if (typeof body !== 'object' || body === null) return false
  const maybe = (body as { error?: unknown }).error
  if (typeof maybe !== 'object' || maybe === null) return false
  const code = (maybe as { code?: unknown }).code
  const message = (maybe as { message?: unknown }).message
  return typeof code === 'string' && typeof message === 'string'
}

/** Pydantic / FastAPI 的 422 校验错误：`{detail:[{loc,msg,type},...]}` */
interface PydanticIssue {
  loc?: unknown
  msg?: unknown
  type?: unknown
}

function isPydantic422(body: unknown): body is { detail: PydanticIssue[] } {
  if (typeof body !== 'object' || body === null) return false
  const detail = (body as { detail?: unknown }).detail
  return Array.isArray(detail)
}

/** 把 Pydantic 的 loc 数组渲染成 `body.question` 这种可读路径 */
function formatLoc(loc: unknown): string {
  if (!Array.isArray(loc)) return ''
  // 去掉首位的 body/query/path 等来源标记
  const parts = loc.filter((item) => typeof item === 'string' || typeof item === 'number')
  return parts.length > 1 ? parts.slice(1).join('.') : parts.join('.')
}

function formatPydanticMessage(issues: PydanticIssue[]): string {
  if (issues.length === 0) return '请求参数校验失败'
  return issues
    .slice(0, 3)
    .map((issue) => {
      const loc = formatLoc(issue.loc)
      const msg = typeof issue.msg === 'string' ? issue.msg : '校验失败'
      return loc ? `${loc}: ${msg}` : msg
    })
    .join('；')
}

/**
 * 把任意 axios 错误解包成 ApiError。
 * 覆盖三种情况：
 * 1. 后端有响应且是统一错误信封；
 * 2. 后端有响应但是 Pydantic 422 格式；
 * 3. 压根没响应（网络断了 / 后端没起来）。
 */
export function toApiError(error: unknown): ApiError {
  if (isApiError(error)) return error

  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<unknown>
    const response = axiosError.response
    const requestIdHeader = response?.headers?.['x-request-id']
    const headerRequestId = typeof requestIdHeader === 'string' ? requestIdHeader : null

    if (response) {
      const body = response.data

      // 情况 1：统一错误信封
      if (isErrorEnvelope(body)) {
        const err: ApiErrorBody = body.error
        return new ApiError({
          code: err.code,
          message: err.message,
          status: response.status,
          requestId: err.request_id ?? headerRequestId,
          detail: err.detail,
          timestamp: err.timestamp,
        })
      }

      // 情况 2：Pydantic 422
      if (response.status === 422) {
        const issues = isPydantic422(body) ? body.detail : []
        return new ApiError({
          code: 'REQUEST_VALIDATION_ERROR',
          message: formatPydanticMessage(issues),
          status: 422,
          requestId: headerRequestId,
          detail: body,
        })
      }

      // 情况 3：有响应但不是已知信封（网关 HTML、代理错误页等）
      const text = typeof body === 'string' ? body.slice(0, 200) : ''
      return new ApiError({
        code: `HTTP_${response.status}`,
        message: text ? `请求失败（HTTP ${response.status}）：${text}` : `请求失败（HTTP ${response.status}）`,
        status: response.status,
        requestId: headerRequestId,
        detail: body,
      })
    }

    // 情况 4：没有响应体
    if (axiosError.code === 'ECONNABORTED' || axiosError.code === 'ETIMEDOUT') {
      return new ApiError({ code: 'TIMEOUT', message: '请求超时，请稍后重试', status: 0 })
    }
    if (axiosError.code === 'ERR_CANCELED') {
      return new ApiError({ code: 'CANCELED', message: '请求已取消', status: 0 })
    }
    return new ApiError({
      code: 'NETWORK_ERROR',
      message: '无法连接后端服务，请确认后端已启动（http://127.0.0.1:8000）',
      status: 0,
      detail: axiosError.message,
    })
  }

  if (error instanceof Error) {
    return new ApiError({ code: 'UNKNOWN_ERROR', message: error.message, status: 0 })
  }
  return new ApiError({ code: 'UNKNOWN_ERROR', message: String(error), status: 0 })
}

/* ------------------------------------------------------------------ 实例与拦截器 */

export const http: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json; charset=utf-8' },
})

// 请求拦截器：注入 Bearer token
http.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = getToken()
  if (token) {
    const headers = config.headers instanceof AxiosHeaders ? config.headers : new AxiosHeaders(config.headers)
    headers.set('Authorization', `Bearer ${token}`)
    config.headers = headers
  }
  return config
})

// 响应拦截器：解包错误信封 + 401 全局处理
http.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: unknown) => {
    const apiError = toApiError(error)
    // 401 只处理一次：清 token → 跳登录页 + toast
    if (apiError.status === 401) {
      triggerUnauthorizedOnce()
    }
    return Promise.reject(apiError)
  },
)

/* ------------------------------------------------------------------ 便捷方法 */

/** GET，返回已解包的响应体 */
export async function get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const res = await http.get<T>(url, { params })
  return res.data
}

/** POST，返回已解包的响应体 */
export async function post<T>(url: string, data?: unknown, config?: Parameters<AxiosInstance['post']>[2]): Promise<T> {
  const res = await http.post<T>(url, data, config)
  return res.data
}

/** PATCH，返回已解包的响应体 */
export async function patch<T>(url: string, data?: unknown): Promise<T> {
  const res = await http.patch<T>(url, data)
  return res.data
}

/** DELETE，忽略响应体 */
export async function del(url: string): Promise<void> {
  await http.delete(url)
}

/** 上传文件（multipart/form-data） */
export async function upload<T>(
  url: string,
  file: File,
  fieldName = 'file',
  extra?: Record<string, string>,
): Promise<T> {
  const form = new FormData()
  form.append(fieldName, file)
  if (extra) {
    Object.entries(extra).forEach(([key, value]) => form.append(key, value))
  }
  const res = await http.post<T>(url, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    // 上传可能较慢，单独放宽超时
    timeout: 300_000,
  })
  return res.data
}
