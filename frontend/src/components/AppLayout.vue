<script setup lang="ts">
/**
 * 应用外壳：左侧导航 + 顶栏 + 用户菜单 + 路由出口。
 *
 * 用户信息在挂载时补拉一次（`/auth/me`），
 * 这样刷新页面后即使只有 localStorage 里的缓存，也能拿到服务端的权威信息。
 */
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'

import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { health } from '@/api/search'
import { formatShortTime } from '@/utils/format'
import type { HealthResponse } from '@/types/models'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const chatStore = useChatStore()

const collapsed = ref(false)
const dark = ref(document.documentElement.classList.contains('dark'))
/** 后端健康状态：离线/降级必须如实显示出来（这个项目的硬规则：不骗人） */
const backendHealth = ref<HealthResponse | null>(null)

const activeMenu = computed<string>(() => {
  const path = route.path
  if (path.startsWith('/kbs')) return '/kbs'
  if (path.startsWith('/chat')) return '/chat'
  if (path.startsWith('/search')) return '/search'
  if (path.startsWith('/eval')) return '/eval'
  if (path.startsWith('/obs')) return '/obs'
  return path
})

function toggleTheme(): void {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  document.documentElement.classList.toggle('light', !dark.value)
  try {
    window.localStorage.setItem('knowflow.theme', dark.value ? 'dark' : 'light')
  } catch {
    /* 忽略 */
  }
}

/** 最近会话：顶栏下拉里能快速跳回上一轮对话（数据来自 chat store，聊天页首次加载后会填充） */
const recentConversations = computed(() => chatStore.conversations.slice(0, 6))

function handleUserCommand(command: string): void {
  if (command === 'logout') {
    confirmLogout()
    return
  }
  if (command === 'chat') {
    void router.push('/chat')
    return
  }
  if (command === 'kbs') {
    void router.push('/kbs')
  }
}

function handleRecentCommand(id: number): void {
  void router.push(`/chat/${id}`)
}

function confirmLogout(): void {
  ElMessageBox.confirm('确定要退出登录吗？', '退出登录', {
    confirmButtonText: '退出',
    cancelButtonText: '取消',
    type: 'warning',
  })
    .then(() => {
      auth.logout()
      ElMessage.success('已退出登录')
    })
    .catch(() => {
      /* 用户取消 */
    })
}

onMounted(async () => {
  const saved = window.localStorage.getItem('knowflow.theme')
  if (saved === 'dark' || (!saved && window.matchMedia?.('(prefers-color-scheme: dark)').matches)) {
    dark.value = true
    document.documentElement.classList.add('dark')
  }
  if (auth.isAuthenticated && !auth.user) {
    await auth.fetchMe().catch(() => undefined)
  }
  // 顶栏的「最近会话」需要会话列表；失败不影响主流程
  if (chatStore.conversations.length === 0) {
    await chatStore.fetchConversations().catch(() => undefined)
  }
  // 拉一次 /health：离线或 embedding 降级时在顶栏亮出来（静默失败，后端没起也不报错）
  await health()
    .then((result) => {
      backendHealth.value = result
    })
    .catch(() => {
      backendHealth.value = null
    })
})
</script>

<template>
  <el-container class="app-layout">
    <el-aside class="app-layout__aside" :width="collapsed ? '64px' : 'var(--kf-sidebar-width)'">
      <div class="app-brand" :class="{ 'app-brand--mini': collapsed }">
        <span class="app-brand__logo" aria-hidden="true">🕸️</span>
        <span v-if="!collapsed" class="app-brand__text">
          <strong>KnowFlow</strong>
          <small>知识库 Agent 控制台</small>
        </span>
      </div>

      <el-menu
        :default-active="activeMenu"
        :collapse="collapsed"
        :collapse-transition="false"
        class="app-menu"
        router
      >
        <el-menu-item index="/kbs">
          <span class="app-menu__icon">📚</span>
          <template #title>知识库</template>
        </el-menu-item>
        <el-menu-item index="/chat">
          <span class="app-menu__icon">💬</span>
          <template #title>智能问答</template>
        </el-menu-item>
        <el-menu-item index="/search">
          <span class="app-menu__icon">🔍</span>
          <template #title>检索调试</template>
        </el-menu-item>
        <el-menu-item index="/eval">
          <span class="app-menu__icon">🧪</span>
          <template #title>评测</template>
        </el-menu-item>
        <el-menu-item index="/obs">
          <span class="app-menu__icon">📈</span>
          <template #title>可观测</template>
        </el-menu-item>
      </el-menu>

      <div class="app-layout__aside-footer">
        <el-button link class="app-collapse-btn" @click="collapsed = !collapsed">
          {{ collapsed ? '➡' : '⬅ 收起' }}
        </el-button>
      </div>
    </el-aside>

    <el-container>
      <el-header class="app-header" height="var(--kf-header-height)">
        <div class="app-header__title">{{ route.meta.title ?? 'KnowFlow' }}</div>
        <div class="app-header__spacer" />

        <!-- 后端运行期真实模式：离线 / embedding 降级必须可见 -->
        <template v-if="backendHealth">
          <el-tooltip placement="bottom" :content="`version ${backendHealth.version} · 运行 ${Math.round(backendHealth.uptime_s)}s · 向量 ${backendHealth.vector_count} · BM25 ${backendHealth.bm25_doc_count}`">
            <el-tag v-if="backendHealth.offline" size="small" type="warning" effect="dark">离线模式</el-tag>
            <el-tag v-else size="small" type="success" effect="plain">在线</el-tag>
          </el-tooltip>
          <el-tooltip
            v-if="backendHealth.embedding_mode && backendHealth.embedding_mode !== 'local'"
            placement="bottom"
            content="embedding 已降级：检索质量会下降，务必知情"
          >
            <el-tag size="small" type="danger" effect="plain">
              embedding: {{ backendHealth.embedding_mode }}
            </el-tag>
          </el-tooltip>
          <el-tooltip
            v-if="backendHealth.warnings.length > 0"
            placement="bottom"
            :content="backendHealth.warnings.join('；')"
          >
            <el-tag size="small" type="warning" effect="plain">告警 {{ backendHealth.warnings.length }}</el-tag>
          </el-tooltip>
        </template>

        <el-dropdown v-if="recentConversations.length > 0" trigger="click" @command="handleRecentCommand">
          <el-button link size="small">最近会话 ▾</el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item
                v-for="item in recentConversations"
                :key="item.id"
                :command="item.id"
              >
                <span class="kf-truncate app-recent__title">{{ item.title }}</span>
                <span class="kf-muted app-recent__time">{{ formatShortTime(item.updated_at) }}</span>
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <el-tooltip :content="dark ? '切换到亮色' : '切换到暗色'" placement="bottom">
          <el-button link class="app-header__icon-btn" @click="toggleTheme">
            {{ dark ? '🌙' : '☀️' }}
          </el-button>
        </el-tooltip>

        <el-dropdown trigger="click" @command="handleUserCommand">
          <span class="app-user">
            <el-avatar :size="26" class="app-user__avatar">
              {{ auth.displayName.slice(0, 1).toUpperCase() }}
            </el-avatar>
            <span class="app-user__name">{{ auth.displayName }}</span>
            <el-tag v-if="auth.isAdmin" size="small" type="warning" effect="plain">admin</el-tag>
            <span class="app-user__caret">▾</span>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item disabled>
                <span class="kf-muted">{{ auth.user?.username ?? '未登录' }}</span>
              </el-dropdown-item>
              <el-dropdown-item command="chat" divided>开始新对话</el-dropdown-item>
              <el-dropdown-item command="kbs">管理知识库</el-dropdown-item>
              <el-dropdown-item command="logout" divided>
                <span style="color: var(--kf-danger)">退出登录</span>
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </el-header>

      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-layout {
  height: 100vh;
  overflow: hidden;
}

.app-layout__aside {
  display: flex;
  flex-direction: column;
  background: var(--kf-bg-card);
  border-right: 1px solid var(--kf-border);
  transition: width 0.2s ease;
}

.app-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  height: var(--kf-header-height);
  padding: 0 14px;
  border-bottom: 1px solid var(--kf-border);
  overflow: hidden;
}

.app-brand--mini {
  justify-content: center;
  padding: 0;
}

.app-brand__logo {
  font-size: 20px;
}

.app-brand__text {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
  white-space: nowrap;
}

.app-brand__text strong {
  font-size: 15px;
  color: var(--kf-text-primary);
}

.app-brand__text small {
  font-size: 11px;
  color: var(--kf-text-secondary);
}

.app-menu {
  flex: 1 1 auto;
  border-right: none;
  background: transparent;
}

.app-menu__icon {
  margin-right: 8px;
}

.app-layout__aside-footer {
  padding: 8px;
  border-top: 1px solid var(--kf-border);
}

.app-collapse-btn {
  width: 100%;
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.app-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 16px;
  background: var(--kf-bg-card);
  border-bottom: 1px solid var(--kf-border);
}

.app-header__title {
  font-size: 15px;
  font-weight: 600;
  color: var(--kf-text-primary);
}

.app-header__spacer {
  flex: 1 1 auto;
}

.app-header__icon-btn {
  font-size: 16px;
}

.app-user {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  outline: none;
}

.app-user__avatar {
  background: var(--kf-primary);
  color: #fff;
  font-size: 12px;
}

.app-user__name {
  font-size: 13px;
  color: var(--kf-text-regular);
}

.app-user__caret {
  font-size: 10px;
  color: var(--kf-text-placeholder);
}

.app-recent__title {
  display: inline-block;
  max-width: 260px;
  vertical-align: bottom;
}

.app-recent__time {
  margin-left: 8px;
  font-size: 11px;
}

.app-main {
  padding: 0;
  background: var(--kf-bg-body);
  overflow: auto;
}
</style>
