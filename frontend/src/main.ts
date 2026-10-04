/**
 * 应用入口：挂载 Pinia → Router → Element Plus → 全局样式。
 *
 * ⚠️ 顺序很重要：**必须先 `app.use(pinia)` 再 `app.use(router)`**。
 * 否则路由守卫/路由组件里调用 `useXxxStore()` 时拿不到 pinia 实例，
 * 会报 `getActivePinia() was called but there was no active Pinia`
 * —— 这是 Vue 3 + Pinia + Router 组合下最常见的坑。
 *
 * （本项目路由守卫里没直接用 store，但保持这个顺序能避免以后有人加 store 时踩坑。）
 */

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import 'element-plus/dist/index.css'

import App from './App.vue'
import router from './router'
import './styles/index.scss'

const app = createApp(App)

// 1) Pinia 必须最先安装
app.use(createPinia())
// 2) 再装 Router（守卫里若用 store，此时已可用）
app.use(router)
// 3) UI 库：中文语言包
app.use(ElementPlus, { locale: zhCn })

/** 全局兜底：未捕获的 Promise 异常打到控制台，避免线上静默失败 */
window.addEventListener('unhandledrejection', (event) => {
  console.error('[knowflow] 未处理的 Promise 异常：', event.reason)
})

app.mount('#app')
