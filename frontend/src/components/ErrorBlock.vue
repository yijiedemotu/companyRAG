<script setup lang="ts">
/**
 * 错误块。
 * 契约要求页面的 toast 里带上 `request_id` 便于排障，
 * 所以这里也把 request_id 单独展示出来并提供复制按钮。
 */
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    /** 可读错误文案（通常直接传 useAsync 的 error） */
    message?: string
    /** 是否展示「重试」按钮 */
    retryable?: boolean
    title?: string
    /** 紧凑模式：用于卡片内部 */
    compact?: boolean
  }>(),
  {
    message: '',
    retryable: true,
    title: '出错了',
    compact: false,
  },
)

const emit = defineEmits<{ retry: []; close: [] }>()

const text = computed(() => props.message || '未知错误')
</script>

<template>
  <div class="error-block" :class="{ 'error-block--compact': compact }">
    <div class="error-block__head">
      <span class="error-block__icon" aria-hidden="true">⚠️</span>
      <span class="error-block__title">{{ title }}</span>
    </div>
    <p class="error-block__message">{{ text }}</p>
    <div v-if="retryable || $slots.default" class="error-block__actions">
      <el-button v-if="retryable" size="small" type="primary" plain @click="emit('retry')">重试</el-button>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.error-block {
  padding: 16px;
  border: 1px solid var(--kf-danger);
  border-radius: var(--kf-radius);
  background: color-mix(in srgb, var(--kf-danger) 8%, transparent);
}

.error-block--compact {
  padding: 10px 12px;
}

.error-block__head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.error-block__title {
  font-weight: 600;
  color: var(--kf-danger);
}

.error-block__message {
  margin: 8px 0 0;
  color: var(--kf-text-regular);
  font-size: 13px;
  word-break: break-word;
}

.error-block__rid {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  font-size: 12px;
}

.error-block__rid code {
  font-family: var(--kf-font-mono);
  color: var(--kf-text-secondary);
  word-break: break-all;
}

.error-block__actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
</style>
