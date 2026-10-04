<script setup lang="ts">
/**
 * 登录页（`/login`，免登录路由）。
 *
 * 契约（5.1）：`username` 3–32 位 `[A-Za-z0-9_]`；`password` ≥ 6 位；
 * **第一个注册的用户自动成为 admin**（方便本地演示与 CI），所以页面上要写清楚这一点。
 */
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'

import { describeError } from '@/api/client'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const mode = ref<'login' | 'register'>('login')
const submitting = ref(false)
const formRef = ref<FormInstance>()

const form = reactive({
  username: '',
  password: '',
  confirmPassword: '',
  display_name: '',
})

const rules = computed<FormRules>(() => ({
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    {
      pattern: /^[A-Za-z0-9_]{3,32}$/,
      message: '用户名需为 3–32 位字母、数字或下划线',
      trigger: 'blur',
    },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码至少 6 位', trigger: 'blur' },
  ],
  confirmPassword:
    mode.value === 'register'
      ? [
          { required: true, message: '请再次输入密码', trigger: 'blur' },
          {
            validator: (_rule, value: string, callback: (error?: Error) => void) => {
              if (value !== form.password) callback(new Error('两次输入的密码不一致'))
              else callback()
            },
            trigger: 'blur',
          },
        ]
      : [],
}))

const submitText = computed<string>(() => (mode.value === 'login' ? '登录' : '注册并登录'))

function switchMode(next: 'login' | 'register'): void {
  mode.value = next
  formRef.value?.clearValidate()
}

async function handleSubmit(): Promise<void> {
  const instance = formRef.value
  if (!instance) return
  const valid = await instance.validate().catch(() => false)
  if (!valid) return

  submitting.value = true
  try {
    if (mode.value === 'login') {
      await auth.login({ username: form.username.trim(), password: form.password })
      ElMessage.success(`欢迎回来，${auth.displayName}`)
    } else {
      await auth.register({
        username: form.username.trim(),
        password: form.password,
        display_name: form.display_name.trim() || null,
      })
      ElMessage.success('注册成功，已自动登录')
    }
    await redirectAfterLogin()
  } catch (error) {
    // 统一错误信封：带上 request_id 便于排障
    ElMessage.error(describeError(error))
  } finally {
    submitting.value = false
  }
}

/** 登录后跳回来源页（守卫在 query 里塞了 redirect） */
async function redirectAfterLogin(): Promise<void> {
  const raw = route.query.redirect
  const target = typeof raw === 'string' && raw.startsWith('/') && !raw.startsWith('//') ? raw : '/kbs'
  await router.replace(target)
}

onMounted(() => {
  // 401 被踢回登录页时，auth store 会留下一条一次性提示
  const notice = auth.consumeNotice()
  if (notice) ElMessage.warning(notice)
  // 已登录（例如手动访问 /login）直接回业务页
  if (auth.isAuthenticated) void redirectAfterLogin()
})
</script>

<template>
  <div class="login-page">
    <div class="login-card kf-card">
      <div class="login-card__brand">
        <span class="login-card__logo" aria-hidden="true">🕸️</span>
        <div>
          <h1 class="login-card__title">KnowFlow</h1>
          <p class="login-card__subtitle">企业知识库 RAG + LangGraph Agent 问答控制台</p>
        </div>
      </div>

      <el-tabs :model-value="mode" class="login-card__tabs" @update:model-value="switchMode($event as 'login' | 'register')">
        <el-tab-pane label="登录" name="login" />
        <el-tab-pane label="注册" name="register" />
      </el-tabs>

      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
        size="large"
        @submit.prevent="handleSubmit"
      >
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" placeholder="3–32 位字母、数字或下划线" autocomplete="username" />
        </el-form-item>

        <el-form-item label="密码" prop="password">
          <el-input
            v-model="form.password"
            type="password"
            show-password
            placeholder="至少 6 位"
            autocomplete="current-password"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <el-form-item v-if="mode === 'register'" label="确认密码" prop="confirmPassword">
          <el-input
            v-model="form.confirmPassword"
            type="password"
            show-password
            placeholder="再输入一次"
            autocomplete="new-password"
            @keyup.enter="handleSubmit"
          />
        </el-form-item>

        <el-form-item v-if="mode === 'register'" label="显示名（可选）" prop="display_name">
          <el-input v-model="form.display_name" placeholder="留空则用用户名" />
        </el-form-item>

        <el-button
          type="primary"
          size="large"
          class="login-card__submit"
          :loading="submitting"
          @click="handleSubmit"
        >
          {{ submitText }}
        </el-button>
      </el-form>

      <el-alert
        v-if="mode === 'register'"
        class="login-card__tip"
        type="info"
        :closable="false"
        show-icon
        title="第一个注册的用户会自动成为 admin"
        description="后续注册的用户是普通 user；管理接口只有 admin 可访问。"
      />
      <el-alert
        v-else
        class="login-card__tip"
        type="info"
        :closable="false"
        show-icon
        title="后端需先启动"
        description="请确认 http://127.0.0.1:8000 已就绪（开发环境由 Vite proxy 转发 /api 请求）。"
      />
    </div>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: linear-gradient(135deg, var(--kf-bg-body), color-mix(in srgb, var(--kf-primary) 12%, var(--kf-bg-body)));
}

.login-card {
  width: 100%;
  max-width: 420px;
  padding: 28px;
}

.login-card__brand {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.login-card__logo {
  font-size: 32px;
}

.login-card__title {
  margin: 0;
  font-size: 22px;
  color: var(--kf-text-primary);
}

.login-card__subtitle {
  margin: 2px 0 0;
  font-size: 12.5px;
  color: var(--kf-text-secondary);
}

.login-card__tabs {
  margin-top: 8px;
}

.login-card__submit {
  width: 100%;
  margin-top: 4px;
}

.login-card__tip {
  margin-top: 16px;
}
</style>
