<script setup lang="ts">
/** 柱状图轻封装（ECharts）。用途：消融对比、按 mode/model 分组的对比。 */
import { computed, watch } from 'vue'

import { chartTheme, useECharts, type EChartsOption } from '@/composables/useECharts'

export interface BarSeries {
  name: string
  data: Array<number | null>
  /** 水平柱状图（类别名很长时更好读） */
  horizontal?: boolean
}

const props = withDefaults(
  defineProps<{
    labels: string[]
    series: BarSeries[]
    height?: string
    yName?: string
    /** 是否堆叠 */
    stack?: boolean
    /** 数值格式化用的最大小数位 */
    decimals?: number
    loading?: boolean
  }>(),
  {
    height: '280px',
    yName: '',
    stack: false,
    decimals: 4,
    loading: false,
  },
)

const { container, setOption } = useECharts()

const horizontal = computed<boolean>(() => props.series.some((item) => item.horizontal))

const option = computed<EChartsOption>(() => {
  const theme = chartTheme()
  const categoryAxis = {
    type: 'category' as const,
    data: props.labels,
    axisLine: { lineStyle: { color: theme.axisColor } },
    axisLabel: { color: theme.textColor, hideOverlap: true },
  }
  const valueAxis = {
    type: 'value' as const,
    name: props.yName,
    nameTextStyle: { color: theme.textColor },
    axisLine: { show: false, lineStyle: { color: theme.axisColor } },
    axisLabel: { color: theme.textColor },
    splitLine: { lineStyle: { color: theme.splitColor } },
  }

  return {
    color: theme.palette,
    grid: { left: 8, right: 16, top: 32, bottom: 8, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: (value) =>
        typeof value === 'number' ? value.toFixed(props.decimals) : String(value ?? ''),
    },
    legend: {
      show: props.series.length > 1,
      top: 0,
      textStyle: { color: theme.textColor },
    },
    xAxis: horizontal.value ? valueAxis : categoryAxis,
    yAxis: horizontal.value ? categoryAxis : valueAxis,
    series: props.series.map((item, index) => ({
      name: item.name,
      type: 'bar' as const,
      stack: props.stack ? 'total' : undefined,
      barMaxWidth: 36,
      itemStyle: { borderRadius: [4, 4, 0, 0] },
      data: item.data,
      color: theme.palette[index % theme.palette.length],
    })),
  }
})

watch(option, (value) => setOption(value), { immediate: true })
</script>

<template>
  <div class="chart-wrap" :style="{ height }" v-loading="loading">
    <div ref="container" class="chart-wrap__canvas" />
  </div>
</template>

<style scoped>
.chart-wrap {
  width: 100%;
  position: relative;
}

.chart-wrap__canvas {
  width: 100%;
  height: 100%;
}
</style>
