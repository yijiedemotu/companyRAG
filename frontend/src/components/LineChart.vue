<script setup lang="ts">
/**
 * 折线图轻封装（ECharts）。
 * 用途：可观测页的请求量 / token / 成本 / 延迟时间序列，评测页的指标趋势。
 */
import { computed, watch } from 'vue'

import { chartTheme, useECharts, type EChartsOption } from '@/composables/useECharts'

/** 一条线 */
export interface LineSeries {
  name: string
  data: Array<number | null>
  /** 是否画在第二个 Y 轴（量纲差异大时用，例如「请求数」和「延迟毫秒」） */
  yAxisIndex?: number
}

const props = withDefaults(
  defineProps<{
    /** X 轴标签（时间点） */
    labels: string[]
    series: LineSeries[]
    /** 容器高度 */
    height?: string
    /** Y 轴名称（左） */
    yName?: string
    /** Y 轴名称（右，只有存在 yAxisIndex=1 的系列时才显示） */
    yNameRight?: string
    /** 面积填充（单条线时好看） */
    area?: boolean
    /** 是否平滑 */
    smooth?: boolean
    loading?: boolean
  }>(),
  {
    height: '280px',
    yName: '',
    yNameRight: '',
    area: false,
    smooth: true,
    loading: false,
  },
)

const { container, setOption } = useECharts()

const hasRightAxis = computed<boolean>(() => props.series.some((item) => item.yAxisIndex === 1))

const option = computed<EChartsOption>(() => {
  const theme = chartTheme()
  return {
    color: theme.palette,
    grid: {
      left: 8,
      right: hasRightAxis.value ? 8 : 16,
      top: 32,
      bottom: 8,
      containLabel: true,
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
    },
    legend: {
      show: props.series.length > 1,
      top: 0,
      textStyle: { color: theme.textColor },
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: props.labels,
      axisLine: { lineStyle: { color: theme.axisColor } },
      axisLabel: { color: theme.textColor, hideOverlap: true },
    },
    yAxis: [
      {
        type: 'value',
        name: props.yName,
        nameTextStyle: { color: theme.textColor },
        axisLine: { show: false, lineStyle: { color: theme.axisColor } },
        axisLabel: { color: theme.textColor },
        splitLine: { lineStyle: { color: theme.splitColor } },
      },
      ...(hasRightAxis.value
        ? [
            {
              type: 'value' as const,
              name: props.yNameRight,
              nameTextStyle: { color: theme.textColor },
              axisLine: { show: false, lineStyle: { color: theme.axisColor } },
              axisLabel: { color: theme.textColor },
              splitLine: { show: false },
            },
          ]
        : []),
    ],
    series: props.series.map((item, index) => ({
      name: item.name,
      type: 'line' as const,
      smooth: props.smooth,
      symbol: 'circle',
      symbolSize: 5,
      showSymbol: props.labels.length <= 40,
      yAxisIndex: item.yAxisIndex ?? 0,
      data: item.data,
      areaStyle: props.area && props.series.length === 1 ? { opacity: 0.15 } : undefined,
      lineStyle: { width: 2 },
      emphasis: { focus: 'series' as const },
      // 用 seriesIndex 兜底避免同名系列配色重复
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
