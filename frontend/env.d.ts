/// <reference types="vite/client" />

/** 环境变量类型声明：与 .env.example 保持一致 */
interface ImportMetaEnv {
  /** API 基址，默认 /api/v1（开发环境由 vite proxy 转发到 127.0.0.1:8000） */
  readonly VITE_API_BASE_URL?: string
  /** 应用标题（顶栏展示） */
  readonly VITE_APP_TITLE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

/** .vue 单文件组件模块声明（vue-tsc 会接管，这里给纯 tsc 兜底） */
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<Record<string, unknown>, Record<string, unknown>, unknown>
  export default component
}

declare module '*.scss' {
  const css: string
  export default css
}
