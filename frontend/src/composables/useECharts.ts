/**
 * ECharts 轻封装：管理实例生命周期（init / setOption / resize / dispose）。
 *
 * 为什么不用 vue-echarts：本项目只有折线图、柱状图、瀑布图三类，
 * 自己包一层 60 行足够，且能完全掌控 resize 与暗色主题的响应。
 */

import { onBeforeUnmount, onMounted, ref, watch, type Ref } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart, LineChart as EChartsLine, CustomChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

echarts.use([
  EChartsLine,
  BarChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  DataZoomComponent,
  MarkLineComponent,
  CanvasRenderer,
])

export interface UseEChartsReturn {
  /** 绑到容器 div 上的 ref */
  container: Ref<HTMLDivElement | undefined>
  /** 设置配置项（内部会做 notMerge 合并） */
  setOption: (option: EChartsOption) => void
  /** 手动重绘 */
  resize: () => void
  /** 拿到原始实例（瀑布图等需要精细控制时用） */
  getInstance: () => echarts.ECharts | null
}

/** 判断当前是否处于暗色主题 */
export function isDarkTheme(): boolean {
  if (typeof document === 'undefined') return false
  const root = document.documentElement
  if (root.classList.contains('dark')) return true
  if (root.classList.contains('light')) return false
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

/** 暗色/亮色下都要可读的公共配色 */
export function chartTheme(): {
  textColor: string
  axisColor: string
  splitColor: string
  palette: string[]
} {
  const dark = isDarkTheme()
  return {
    textColor: dark ? '#cbd5e1' : '#334155',
    axisColor: dark ? '#475569' : '#cbd5e1',
    splitColor: dark ? '#334155' : '#e2e8f0',
    palette: ['#2563eb', '#16a34a', '#d97706', '#dc2626', '#7c3aed', '#0891b2'],
  }
}

export function useECharts(): UseEChartsReturn {
  const container = ref<HTMLDivElement>()
  let instance: echarts.ECharts | null = null
  let pendingOption: EChartsOption | null = null

  function ensureInstance(): echarts.ECharts | null {
    if (!container.value) return null
    if (!instance || instance.isDisposed()) {
      instance = echarts.init(container.value)
    }
    return instance
  }

  function setOption(option: EChartsOption): void {
    pendingOption = option
    const chart = ensureInstance()
    if (!chart) return
    // notMerge=true：每次都整体替换，避免切换数据后残留旧系列
    chart.setOption(option, true)
  }

  function resize(): void {
    instance?.resize()
  }

  let observer: ResizeObserver | null = null

  onMounted(() => {
    // 容器尺寸变化时重绘（侧边栏折叠、窗口缩放都会触发）
    if (container.value && typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => resize())
      observer.observe(container.value)
    }
    // 挂载后再落一次 option（此前容器还不存在）
    if (pendingOption) {
      const chart = ensureInstance()
      chart?.setOption(pendingOption, true)
    }
  })

  onBeforeUnmount(() => {
    observer?.disconnect()
    observer = null
    instance?.dispose()
    instance = null
  })

  // 主题切换（html 上的 class 变化）时重绘
  watch(
    () => document.documentElement.className,
    () => {
      if (pendingOption) setOption(pendingOption)
    },
  )

  return { container, setOption, resize, getInstance: () => instance }
}

export { echarts }
export type { EChartsOption }
