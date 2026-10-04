<script setup lang="ts">
/** 空状态占位：没有数据时给一句明确的下一步，而不是一片空白 */
withDefaults(
  defineProps<{
    /** 主标题 */
    title?: string
    /** 补充说明 */
    description?: string
    /** 图标（emoji 或字符） */
    icon?: string
  }>(),
  {
    title: '暂无数据',
    description: '',
    icon: '📭',
  },
)
</script>

<template>
  <div class="empty-state">
    <div class="empty-state__icon" aria-hidden="true">{{ icon }}</div>
    <div class="empty-state__title">{{ title }}</div>
    <div v-if="description" class="empty-state__desc">{{ description }}</div>
    <div v-if="$slots.default" class="empty-state__action">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 40px 16px;
  text-align: center;
  color: var(--kf-text-secondary);
}

.empty-state__icon {
  font-size: 32px;
  line-height: 1;
}

.empty-state__title {
  font-size: 15px;
  font-weight: 600;
  color: var(--kf-text-regular);
}

.empty-state__desc {
  font-size: 13px;
  max-width: 420px;
}

.empty-state__action {
  margin-top: 8px;
}
</style>
