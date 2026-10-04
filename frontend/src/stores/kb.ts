/**
 * 知识库 store：列表 + 当前 KB。
 *
 * 说明：`currentKbId` 会持久化到 localStorage，
 * 让「检索调试」「对话」页面在刷新后还停留在同一个 KB，少点几次。
 */

import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import { createKB, deleteKB, getKB, getKBStats, listKBs, updateKB } from '@/api/kbs'
import type { KBCreateRequest, KBUpdateRequest } from '@/types/dto'
import type { KBOut, KBStatsOut, Page } from '@/types/models'

const CURRENT_KB_KEY = 'knowflow.currentKbId'

function loadCurrentKbId(): number | null {
  try {
    const raw = window.localStorage.getItem(CURRENT_KB_KEY)
    if (!raw) return null
    const parsed = Number(raw)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null
  } catch {
    return null
  }
}

export const useKbStore = defineStore('kb', () => {
  const items = ref<KBOut[]>([])
  const total = ref<number>(0)
  const page = ref<number>(1)
  const size = ref<number>(20)
  const pages = ref<number>(0)
  const keyword = ref<string>('')
  const loading = ref<boolean>(false)
  const stats = ref<KBStatsOut | null>(null)
  const currentKbId = ref<number | null>(loadCurrentKbId())

  const currentKb = computed<KBOut | null>(
    () => items.value.find((kb) => kb.id === currentKbId.value) ?? null,
  )

  /** 后端 chunk_count 与向量库数量是否一致；不一致要在 UI 上显红点（把静默降级变可见） */
  const inconsistent = computed<boolean>(() => stats.value !== null && !stats.value.consistent)

  function setCurrentKbId(kbId: number | null): void {
    currentKbId.value = kbId
    try {
      if (kbId === null) window.localStorage.removeItem(CURRENT_KB_KEY)
      else window.localStorage.setItem(CURRENT_KB_KEY, String(kbId))
    } catch {
      /* 忽略 */
    }
  }

  async function fetchList(options: { page?: number; size?: number; keyword?: string } = {}): Promise<Page<KBOut>> {
    loading.value = true
    try {
      const nextPage = options.page ?? page.value
      const nextSize = options.size ?? size.value
      const nextKeyword = options.keyword ?? keyword.value
      const result = await listKBs({ page: nextPage, size: nextSize, keyword: nextKeyword })
      items.value = result.items
      total.value = result.total
      page.value = result.page
      size.value = result.size
      pages.value = result.pages
      keyword.value = nextKeyword
      // 当前 KB 已被删除时清理选中态
      if (currentKbId.value !== null && !result.items.some((kb) => kb.id === currentKbId.value)) {
        const stillExists = await getKB(currentKbId.value).catch(() => null)
        if (!stillExists) setCurrentKbId(null)
      }
      return result
    } finally {
      loading.value = false
    }
  }

  async function fetchOne(kbId: number): Promise<KBOut> {
    const kb = await getKB(kbId)
    const index = items.value.findIndex((item) => item.id === kbId)
    if (index >= 0) items.value[index] = kb
    return kb
  }

  async function fetchStats(kbId: number): Promise<KBStatsOut> {
    const result = await getKBStats(kbId)
    stats.value = result
    return result
  }

  async function create(payload: KBCreateRequest): Promise<KBOut> {
    const kb = await createKB(payload)
    items.value = [kb, ...items.value]
    total.value += 1
    return kb
  }

  async function update(kbId: number, payload: KBUpdateRequest): Promise<KBOut> {
    const kb = await updateKB(kbId, payload)
    const index = items.value.findIndex((item) => item.id === kbId)
    if (index >= 0) items.value[index] = kb
    return kb
  }

  async function remove(kbId: number): Promise<void> {
    await deleteKB(kbId)
    items.value = items.value.filter((item) => item.id !== kbId)
    total.value = Math.max(0, total.value - 1)
    if (currentKbId.value === kbId) setCurrentKbId(null)
  }

  return {
    items,
    total,
    page,
    size,
    pages,
    keyword,
    loading,
    stats,
    currentKbId,
    currentKb,
    inconsistent,
    setCurrentKbId,
    fetchList,
    fetchOne,
    fetchStats,
    create,
    update,
    remove,
  }
})
