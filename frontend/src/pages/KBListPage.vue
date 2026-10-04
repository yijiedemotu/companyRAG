<script setup lang="ts">
/**
 * 知识库列表页 `/kbs`。
 *
 * 能力：搜索、分页、新建、编辑、删除、设为「当前 KB」、跳转文档管理。
 * 契约 5.2：GET/POST/PATCH/DELETE `/kbs` 与 `/kbs/{kb_id}/stats`。
 */
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'

import { describeError } from '@/api/client'
import { useKbStore } from '@/stores/kb'
import { useAsync } from '@/composables/useAsync'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LoadingBlock from '@/components/LoadingBlock.vue'
import PageBar from '@/components/PageBar.vue'
import type { KBOut } from '@/types/models'
import { formatDateTime, formatInt } from '@/utils/format'

const router = useRouter()
const kbStore = useKbStore()

const keyword = ref('')
const dialogVisible = ref(false)
const dialogMode = ref<'create' | 'edit'>('create')
const editingId = ref<number | null>(null)
const submitting = ref(false)
const formRef = ref<FormInstance>()

const form = reactive({
  name: '',
  description: '',
  chunk_size: 600,
  chunk_overlap: 120,
  is_active: true,
})

const rules: FormRules = {
  name: [
    { required: true, message: '请输入知识库名称', trigger: 'blur' },
    { max: 128, message: '名称不超过 128 字符', trigger: 'blur' },
  ],
  chunk_size: [{ required: true, message: '请输入 chunk 大小', trigger: 'blur' }],
  chunk_overlap: [
    {
      validator: (_rule, value: number, callback: (error?: Error) => void) => {
        // 契约 400 VALIDATION_ERROR：chunk_overlap ≥ chunk_size
        if (typeof value === 'number' && typeof form.chunk_size === 'number' && value >= form.chunk_size) {
          callback(new Error('重叠长度必须小于 chunk 大小'))
        } else {
          callback()
        }
      },
      trigger: 'blur',
    },
  ],
}

const page = ref(1)
const size = ref(20)

const listAsync = useAsync(async () => {
  await kbStore.fetchList({ page: page.value, size: size.value, keyword: keyword.value.trim() })
})

const items = computed<KBOut[]>(() => kbStore.items)

async function load(): Promise<void> {
  await listAsync.run()
}

function handleSearch(): void {
  page.value = 1
  void load()
}

function handleReset(): void {
  keyword.value = ''
  page.value = 1
  void load()
}

function handlePageChange(payload: { page: number; size: number }): void {
  page.value = payload.page
  size.value = payload.size
  void load()
}

function openCreate(): void {
  dialogMode.value = 'create'
  editingId.value = null
  form.name = ''
  form.description = ''
  form.chunk_size = 600
  form.chunk_overlap = 120
  form.is_active = true
  formRef.value?.clearValidate()
  dialogVisible.value = true
}

function openEdit(kb: KBOut): void {
  dialogMode.value = 'edit'
  editingId.value = kb.id
  form.name = kb.name
  form.description = kb.description ?? ''
  form.chunk_size = kb.chunk_size
  form.chunk_overlap = kb.chunk_overlap
  form.is_active = kb.is_active
  formRef.value?.clearValidate()
  dialogVisible.value = true
}

async function handleSubmit(): Promise<void> {
  const instance = formRef.value
  if (!instance) return
  const valid = await instance.validate().catch(() => false)
  if (!valid) return

  submitting.value = true
  try {
    if (dialogMode.value === 'create') {
      const created = await kbStore.create({
        name: form.name.trim(),
        description: form.description.trim() || null,
        chunk_size: form.chunk_size,
        chunk_overlap: form.chunk_overlap,
      })
      ElMessage.success(`知识库「${created.name}」已创建`)
      dialogVisible.value = false
      kbStore.setCurrentKbId(created.id)
      await router.push(`/kbs/${created.id}`)
      return
    }

    if (editingId.value !== null) {
      await kbStore.update(editingId.value, {
        name: form.name.trim(),
        description: form.description.trim() || null,
        chunk_size: form.chunk_size,
        chunk_overlap: form.chunk_overlap,
        is_active: form.is_active,
      })
      ElMessage.success('已保存')
    }
    dialogVisible.value = false
    await load()
  } catch (error) {
    ElMessage.error(describeError(error))
  } finally {
    submitting.value = false
  }
}

async function handleDelete(kb: KBOut): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `删除「${kb.name}」会同时清空它的向量集合与 BM25 索引，且不可恢复。确定继续吗？`,
      '删除知识库',
      { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' },
    )
  } catch {
    return
  }

  try {
    await kbStore.remove(kb.id)
    ElMessage.success('已删除')
    await load()
  } catch (error) {
    ElMessage.error(describeError(error))
  }
}

function openDetail(kb: KBOut): void {
  kbStore.setCurrentKbId(kb.id)
  void router.push(`/kbs/${kb.id}`)
}

function openSearch(kb: KBOut): void {
  kbStore.setCurrentKbId(kb.id)
  void router.push('/search')
}

function openChat(kb: KBOut): void {
  kbStore.setCurrentKbId(kb.id)
  void router.push('/chat')
}

onMounted(() => {
  void load()
})
</script>

<template>
  <div class="kf-page">
    <div class="kf-page-header">
      <div>
        <h2 class="kf-page-title">知识库</h2>
        <p class="kf-page-subtitle">
          知识库是检索边界：换 embedding 模型会换向量维度，所以 embedding 信息在创建时被快照下来。
        </p>
      </div>
      <div class="kf-row">
        <el-input
          v-model="keyword"
          placeholder="按名称搜索"
          clearable
          style="width: 220px"
          @keyup.enter="handleSearch"
          @clear="handleReset"
        />
        <el-button type="primary" @click="handleSearch">搜索</el-button>
        <el-button type="success" @click="openCreate">新建知识库</el-button>
      </div>
    </div>

    <ErrorBlock
      v-if="listAsync.error.value"
      :message="listAsync.error.value"
      @retry="load"
    />

    <LoadingBlock v-else-if="listAsync.loading.value && items.length === 0" height="220px" />

    <EmptyState
      v-else-if="items.length === 0"
      icon="📚"
      title="还没有知识库"
      description="创建一个知识库，上传 markdown / txt / pdf / csv / json 文档，就可以开始检索与问答了。"
    >
      <el-button type="primary" @click="openCreate">新建知识库</el-button>
    </EmptyState>

    <template v-else>
      <div class="kb-grid">
        <div
          v-for="kb in items"
          :key="kb.id"
          class="kb-card kf-card"
          :class="{ 'kb-card--current': kbStore.currentKbId === kb.id }"
        >
          <div class="kb-card__head">
            <span class="kb-card__name kf-truncate" :title="kb.name">{{ kb.name }}</span>
            <el-tag v-if="kbStore.currentKbId === kb.id" size="small" type="primary" effect="dark">当前</el-tag>
            <el-tag v-if="!kb.is_active" size="small" type="info" effect="plain">已停用</el-tag>
          </div>

          <p class="kb-card__desc">{{ kb.description || '（无描述）' }}</p>

          <div class="kb-card__stats">
            <div class="kb-card__stat">
              <span class="kb-card__stat-value">{{ formatInt(kb.doc_count) }}</span>
              <span class="kb-card__stat-label">文档</span>
            </div>
            <div class="kb-card__stat">
              <span class="kb-card__stat-value">{{ formatInt(kb.chunk_count) }}</span>
              <span class="kb-card__stat-label">chunk</span>
            </div>
            <div class="kb-card__stat">
              <span class="kb-card__stat-value">{{ kb.chunk_size }}</span>
              <span class="kb-card__stat-label">chunk_size</span>
            </div>
            <div class="kb-card__stat">
              <span class="kb-card__stat-value">{{ kb.chunk_overlap }}</span>
              <span class="kb-card__stat-label">overlap</span>
            </div>
          </div>

          <div class="kb-card__meta kf-muted">
            <div>
              <span class="kf-mono">{{ kb.embedding_provider }}</span>
              ·
              <span class="kf-mono kf-truncate" :title="kb.embedding_model">{{ kb.embedding_model }}</span>
              · {{ kb.embedding_dim }} 维
            </div>
            <div>创建于 {{ formatDateTime(kb.created_at) }}</div>
          </div>

          <div class="kb-card__actions">
            <el-button size="small" type="primary" @click="openDetail(kb)">文档管理</el-button>
            <el-button size="small" @click="openSearch(kb)">检索调试</el-button>
            <el-button size="small" @click="openChat(kb)">去问答</el-button>
            <span class="kf-spacer" />
            <el-button size="small" link @click="openEdit(kb)">编辑</el-button>
            <el-button size="small" link type="danger" @click="handleDelete(kb)">删除</el-button>
          </div>
        </div>
      </div>

      <PageBar
        :page="kbStore.page"
        :size="kbStore.size"
        :total="kbStore.total"
        :pages="kbStore.pages"
        @change="handlePageChange"
      />
    </template>

    <el-dialog
      v-model="dialogVisible"
      :title="dialogMode === 'create' ? '新建知识库' : '编辑知识库'"
      width="520px"
      destroy-on-close
    >
      <el-form ref="formRef" :model="form" :rules="rules" label-width="110px">
        <el-form-item label="名称" prop="name">
          <el-input v-model="form.name" placeholder="例如：员工手册知识库" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input v-model="form.description" type="textarea" :rows="2" placeholder="可选" />
        </el-form-item>
        <el-form-item label="chunk 大小" prop="chunk_size">
          <el-input-number v-model="form.chunk_size" :min="100" :max="4000" :step="50" />
          <span class="kf-muted dialog-hint">默认 600</span>
        </el-form-item>
        <el-form-item label="重叠长度" prop="chunk_overlap">
          <el-input-number v-model="form.chunk_overlap" :min="0" :max="2000" :step="10" />
          <span class="kf-muted dialog-hint">必须小于 chunk 大小（默认 120）</span>
        </el-form-item>
        <el-form-item v-if="dialogMode === 'edit'" label="启用">
          <el-switch v-model="form.is_active" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="handleSubmit">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.kb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}

.kb-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
}

.kb-card--current {
  border-color: var(--kf-primary);
  box-shadow: 0 0 0 1px var(--kf-primary);
}

.kb-card__head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.kb-card__name {
  font-size: 16px;
  font-weight: 600;
  color: var(--kf-text-primary);
  max-width: 100%;
}

.kb-card__desc {
  margin: 0;
  font-size: 13px;
  color: var(--kf-text-secondary);
  min-height: 20px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.kb-card__stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  padding: 8px 0;
  border-top: 1px solid var(--kf-border);
  border-bottom: 1px solid var(--kf-border);
}

.kb-card__stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}

.kb-card__stat-value {
  font-size: 16px;
  font-weight: 600;
  color: var(--kf-text-primary);
  font-variant-numeric: tabular-nums;
}

.kb-card__stat-label {
  font-size: 11px;
  color: var(--kf-text-secondary);
}

.kb-card__meta {
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.kb-card__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.dialog-hint {
  margin-left: 8px;
  font-size: 12px;
}
</style>
