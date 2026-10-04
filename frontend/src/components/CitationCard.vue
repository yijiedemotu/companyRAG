<script setup lang="ts">
/**
 * 引用卡片：文件名 / 页码 / 小节路径 / 分数 / 片段。
 *
 * 与 MessageBubble 里的 `[n]` 角标通过 `rank` 关联：
 * 点击角标后由页面滚动到这里并高亮（`active` 控制高亮态）。
 */
import { computed } from 'vue'

import type { CitationOut, Source } from '@/types/models'

const props = withDefaults(
  defineProps<{
    /** 引用来源（流式期来自 sources 事件） */
    source?: Source | null
    /** 历史消息里的引用（只有 message_citations 的字段） */
    citation?: CitationOut | null
    /** 是否高亮（点击角标后） */
    active?: boolean
    /** 是否紧凑模式 */
    compact?: boolean
  }>(),
  {
    source: null,
    citation: null,
    active: false,
    compact: false,
  },
)

const rank = computed<number>(() => props.source?.rank ?? props.citation?.rank ?? 0)
const docName = computed<string>(() => {
  if (props.source) return props.source.doc_name
  if (props.citation?.doc_name) return props.citation.doc_name
  if (props.citation?.doc_id) return `文档 #${props.citation.doc_id}`
  return '未知文档'
})
const pageNo = computed<number | null>(() => props.source?.page_no ?? props.citation?.page_no ?? null)
const sectionPath = computed<string | null>(
  () => props.source?.section_path ?? props.citation?.section_path ?? null,
)
const score = computed<number | null>(() => props.source?.score ?? props.citation?.score ?? null)
const snippet = computed<string>(() => props.source?.snippet ?? props.citation?.snippet ?? '')
const chunkId = computed<number | null>(() => {
  if (props.source) return props.source.chunk_id || null
  return props.citation?.chunk_id ?? null
})

/** 分数展示：向量/重排分数都在 0~1，用三位小数足够 */
const scoreText = computed<string>(() => (score.value === null ? '-' : score.value.toFixed(3)))
</script>

<template>
  <div class="citation-card" :class="{ 'citation-card--active': active, 'citation-card--compact': compact }">
    <div class="citation-card__head">
      <span class="citation-card__badge">[{{ rank }}]</span>
      <span class="citation-card__doc kf-truncate" :title="docName">{{ docName }}</span>
      <el-tag v-if="pageNo !== null" size="small" type="info" effect="plain">P{{ pageNo }}</el-tag>
      <span class="kf-spacer" />
      <el-tooltip content="检索/重排分数" placement="top">
        <span class="citation-card__score">{{ scoreText }}</span>
      </el-tooltip>
    </div>

    <div v-if="sectionPath" class="citation-card__section" :title="sectionPath">
      <span class="citation-card__section-icon" aria-hidden="true">📑</span>
      {{ sectionPath }}
    </div>

    <p v-if="snippet" class="citation-card__snippet">{{ snippet }}</p>

    <div class="citation-card__foot kf-muted">
      <span v-if="chunkId !== null" class="kf-mono">chunk_id: {{ chunkId }}</span>
    </div>
  </div>
</template>

<style scoped>
.citation-card {
  padding: 10px 12px;
  border: 1px solid var(--kf-border);
  border-left: 3px solid var(--kf-primary);
  border-radius: var(--kf-radius-sm);
  background: var(--kf-bg-card);
  transition: background-color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
  scroll-margin-top: 80px;
}

.citation-card--compact {
  padding: 8px 10px;
}

.citation-card--active {
  border-color: var(--kf-citation-active);
  border-left-color: var(--kf-citation-active);
  background: color-mix(in srgb, var(--kf-citation-active) 12%, var(--kf-bg-card));
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--kf-citation-active) 35%, transparent);
}

.citation-card__head {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.citation-card__badge {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 20px;
  padding: 0 5px;
  border-radius: var(--kf-radius-sm);
  background: var(--kf-citation-bg);
  color: var(--kf-citation-text);
  font-size: 12px;
  font-weight: 600;
}

.citation-card__doc {
  font-size: 13px;
  font-weight: 600;
  color: var(--kf-text-primary);
  max-width: 220px;
}

.citation-card__score {
  font-family: var(--kf-font-mono);
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.citation-card__section {
  margin-top: 4px;
  font-size: 12px;
  color: var(--kf-text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.citation-card__section-icon {
  margin-right: 2px;
}

.citation-card__snippet {
  margin: 6px 0 0;
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--kf-text-regular);
  display: -webkit-box;
  -webkit-line-clamp: 4;
  line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.citation-card__foot {
  margin-top: 6px;
  font-size: 11px;
}
</style>
