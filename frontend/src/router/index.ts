/**
 * 路由表 + 登录守卫。
 *
 * 契约（docs/01 第七节）规定页路由：
 * `/login`、`/kbs`、`/kbs/:id`、`/chat`、`/chat/:conversationId`、
 * `/search`、`/eval`、`/eval/runs/:id`、`/obs`、`/obs/traces/:traceId`
 */

import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

import AppLayout from '@/components/AppLayout.vue'
import { getToken } from '@/api/client'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'login',
    component: () => import('@/pages/LoginPage.vue'),
    meta: { public: true, title: '登录' },
  },
  {
    path: '/',
    component: AppLayout,
    redirect: '/kbs',
    children: [
      {
        path: 'kbs',
        name: 'kb-list',
        component: () => import('@/pages/KBListPage.vue'),
        meta: { title: '知识库' },
      },
      {
        path: 'kbs/:id',
        name: 'kb-detail',
        component: () => import('@/pages/KBDetailPage.vue'),
        props: true,
        meta: { title: '文档管理' },
      },
      {
        path: 'chat',
        name: 'chat',
        component: () => import('@/pages/ChatPage.vue'),
        meta: { title: '智能问答' },
      },
      {
        path: 'chat/:conversationId',
        name: 'chat-conversation',
        component: () => import('@/pages/ChatPage.vue'),
        props: true,
        meta: { title: '智能问答' },
      },
      {
        path: 'search',
        name: 'search-debug',
        component: () => import('@/pages/SearchDebugPage.vue'),
        meta: { title: '检索调试' },
      },
      {
        path: 'eval',
        name: 'eval',
        component: () => import('@/pages/EvalPage.vue'),
        meta: { title: '评测' },
      },
      {
        path: 'eval/runs/:id',
        name: 'eval-run-detail',
        component: () => import('@/pages/EvalRunDetailPage.vue'),
        props: true,
        meta: { title: '评测运行详情' },
      },
      {
        path: 'obs',
        name: 'obs',
        component: () => import('@/pages/ObsPage.vue'),
        meta: { title: '可观测' },
      },
      {
        path: 'obs/traces/:traceId',
        name: 'trace-detail',
        component: () => import('@/pages/TraceDetailPage.vue'),
        props: true,
        meta: { title: '链路详情' },
      },
    ],
  },
  // 兜底：未知路径回到知识库列表
  { path: '/:pathMatch(.*)*', redirect: '/kbs' },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})

/**
 * 登录守卫。
 *
 * ⚠️ 这里**不使用 pinia store**，而是直接读 token（`getToken()`）：
 * `main.ts` 里必须先 `app.use(pinia)` 再 `app.use(router)`，
 * 但如果守卫里调用 `useAuthStore()`，在「首次导航发生在 pinia 安装之前」的场景
 * （例如某些插件或同步 resolve 的初始导航）会抛「getActivePinia() was called but there was no active Pinia」。
 * 直接读 localStorage 既避开这个 Vue 3 常见坑，也不会有循环依赖。
 * pinia 的 `app.use` 顺序在 main.ts 中仍然严格遵守（并写了注释说明原因）。
 */
router.beforeEach((to) => {
  const hasToken = getToken() !== ''

  if (to.meta.public === true) {
    // 已登录还去登录页 → 直接回首页
    if (hasToken && to.name === 'login') {
      return { path: '/kbs' }
    }
    return true
  }

  if (!hasToken) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }

  return true
})

router.afterEach((to) => {
  const base = 'KnowFlow'
  const title = typeof to.meta.title === 'string' ? to.meta.title : ''
  document.title = title ? `${title} · ${base}` : base
})

export default router
