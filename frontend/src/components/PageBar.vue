<script setup lang="ts">
/**
 * 分页条。
 *
 * 契约里所有列表都是 `{items,total,page,size,pages}`，
 * Element Plus 的 `el-pagination` 用的是 `page-count`（总页数），所以这里做 `pages → page-count` 映射。
 */
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    /** 当前页（从 1 开始） */
    page: number
    /** 每页条数 */
    size: number
    /** 总条数 */
    total: number
    /** 总页数（契约里的 pages），不传则由 total/size 推导 */
    pages?: number
    /** 可选每页条数 */
    pageSizes?: number[]
    /** 是否显示「共 N 条」 */
    showTotal?: boolean
    disabled?: boolean
  }>(),
  {
    pages: undefined,
    pageSizes: () => [10, 20, 50, 100],
    showTotal: true,
    disabled: false,
  },
)

const emit = defineEmits<{
  'update:page': [value: number]
  'update:size': [value: number]
  change: [payload: { page: number; size: number }]
}>()

/** pages → page-count；pages 缺省或为 0 时用 total/size 向上取整兜底 */
const pageCount = computed<number>(() => {
  if (props.pages !== undefined && props.pages > 0) return props.pages
  if (props.size > 0) return Math.max(1, Math.ceil(props.total / props.size))
  return 1
})

function handlePageChange(nextPage: number): void {
  emit('update:page', nextPage)
  emit('change', { page: nextPage, size: props.size })
}

function handleSizeChange(nextSize: number): void {
  emit('update:size', nextSize)
  // 每页条数变化后回到第一页，否则可能落在不存在的页上
  emit('update:page', 1)
  emit('change', { page: 1, size: nextSize })
}
</script>

<template>
  <div class="page-bar">
    <span v-if="showTotal" class="page-bar__total kf-muted">共 {{ total }} 条</span>
    <el-pagination
      :current-page="page"
      :page-size="size"
      :page-count="pageCount"
      :page-sizes="pageSizes"
      :disabled="disabled"
      background
      layout="sizes, prev, pager, next, jumper"
      @current-change="handlePageChange"
      @size-change="handleSizeChange"
    />
  </div>
</template>

<style scoped>
.page-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  padding: 12px 0 4px;
}

.page-bar__total {
  font-size: 13px;
}
</style>
