<script setup lang="ts">
/**
 * 指标卡。
 * 可观测页的四类卡（请求量 / token / 成本 / 延迟分位）与评测页的指标卡都用它。
 */
withDefaults(
  defineProps<{
    /** 指标名 */
    label: string
    /** 主数值（已格式化好的字符串） */
    value: string | number
    /** 单位或后缀 */
    suffix?: string
    /** 次要说明 */
    hint?: string
    /** 语义色调 */
    tone?: 'default' | 'primary' | 'success' | 'warning' | 'danger'
    /** 顶部图标 */
    icon?: string
    /** 是否加载中 */
    loading?: boolean
  }>(),
  {
    suffix: '',
    hint: '',
    tone: 'default',
    icon: '',
    loading: false,
  },
)
</script>

<template>
  <div class="stat-card kf-card" :class="`stat-card--${tone}`">
    <div class="stat-card__head">
      <span v-if="icon" class="stat-card__icon" aria-hidden="true">{{ icon }}</span>
      <span class="stat-card__label">{{ label }}</span>
    </div>
    <div class="stat-card__value">
      <span v-if="loading" class="stat-card__loading">…</span>
      <template v-else>
        <span class="stat-card__number">{{ value }}</span>
        <span v-if="suffix" class="stat-card__suffix">{{ suffix }}</span>
      </template>
    </div>
    <div v-if="hint || $slots.hint" class="stat-card__hint">
      <slot name="hint">{{ hint }}</slot>
    </div>
  </div>
</template>

<style scoped>
.stat-card {
  padding: 14px 16px;
  border-left: 3px solid var(--kf-border-strong);
}

.stat-card--primary {
  border-left-color: var(--kf-primary);
}

.stat-card--success {
  border-left-color: var(--kf-success);
}

.stat-card--warning {
  border-left-color: var(--kf-warning);
}

.stat-card--danger {
  border-left-color: var(--kf-danger);
}

.stat-card__head {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--kf-text-secondary);
  font-size: 13px;
}

.stat-card__icon {
  font-size: 14px;
}

.stat-card__value {
  display: flex;
  align-items: baseline;
  gap: 4px;
  margin-top: 6px;
}

.stat-card__number {
  font-size: 24px;
  font-weight: 600;
  line-height: 1.2;
  color: var(--kf-text-primary);
  font-variant-numeric: tabular-nums;
}

.stat-card__suffix {
  font-size: 13px;
  color: var(--kf-text-secondary);
}

.stat-card__loading {
  font-size: 24px;
  color: var(--kf-text-placeholder);
}

.stat-card__hint {
  margin-top: 6px;
  font-size: 12px;
  color: var(--kf-text-secondary);
}
</style>
