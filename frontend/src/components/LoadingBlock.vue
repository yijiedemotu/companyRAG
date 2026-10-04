<script setup lang="ts">
/** 加载占位块：统一骨架屏样式，避免每个页面各写一套 loading */
withDefaults(
  defineProps<{
    /** 文案 */
    text?: string
    /** 高度 */
    height?: string
  }>(),
  {
    text: '加载中…',
    height: '120px',
  },
)
</script>

<template>
  <div class="loading-block" :style="{ minHeight: height }">
    <span class="loading-block__spinner" aria-hidden="true" />
    <span class="loading-block__text">{{ text }}</span>
  </div>
</template>

<style scoped>
.loading-block {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--kf-text-secondary);
  font-size: 13px;
}

.loading-block__spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--kf-border-strong);
  border-top-color: var(--kf-primary);
  border-radius: 50%;
  animation: loading-block-spin 0.8s linear infinite;
}

.loading-block__text {
  line-height: 1;
}

@keyframes loading-block-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
