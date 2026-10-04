<script setup lang="ts">
/**
 * 知识库详情页 `/kbs/:id` —— 文档管理。
 *
 * 契约 5.2/5.3：
 * - `GET /kbs/{id}` + `GET /kbs/{id}/stats`（`consistent` 为 false 时要**显红点**，把静默降级变可见）；
 * - `POST /kbs/{id}/documents`（multipart `file`）、`GET /kbs/{id}/documents`、`DELETE /documents/{id}`；
 * - `GET /documents/{id}/chunks` 给「chunk 预览抽屉」。
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type UploadRequestOptions } from 'element-plus'

import { describeError } from '@/api/client'
import * as documentsApi from '@/api/documents'
import * as kbsApi from '@/api/kbs'
import EmptyState from '@/components/EmptyState.vue'
import ErrorBlock from '@/components/ErrorBlock.vue'
import LoadingBlock from '@/components/LoadingBlock.vue'
import PageBar from '@/components/PageBar.vue'
import type { ChunkOut, DocumentOut, KBOut, KBStatsOut } from '@/types/models'
import {
  documentStatusTag,
  documentStatusText,
  formatBytes,
  formatDateTime,
  formatDuration,
  formatInt,
  isDocumentProcessing,
} from '@/utils/format'

const props = defineProps<{ id: string }>()

const router = useRouter()
const kbId = computed<number>(() => Number(props.id))

const kb = ref<KBOut | null>(null)
const stats = ref<KBStatsOut | null>(null)
const loadingKb = ref(false)
const kbError = ref('')

const documents = ref<DocumentOut[]>([])
const docTotal = ref(0)
const docPage = ref(1)
const docSize = ref(10)
const docPages = ref(0)
const statusFilter = ref('')
const loadingDocs = ref(false)
const docsError = ref('')

/* chunk 预览抽屉 */
const chunkDrawerVisible = ref(false)
const chunkDoc = ref<DocumentOut | null>(null)
const chunks = ref<ChunkOut[]>([])
const chunkTotal = ref(0)
const chunkPage = ref(1)
const chunkSize = ref(10)
const chunkPages = ref(0)
const loadingChunks = ref(false)

/** 只要还有文档在解析/切分/向量化，就轮询刷新状态 */
const hasProcessing = computed<boolean>(() => documents.value.some((doc) => isDocumentProcessing(doc.status)))
let pollTimer: number | null = null

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function syncPolling(): void {
  if (hasProcessing.value && pollTimer === null) {
    pollTimer = window.setInterval(() => {
      void loadDocuments(true)
    }, 5000)
  } else if (!hasProcessing.value) {
    stopPolling()
  }
}

async function loadKb(): Promise<void> {
  loadingKb.value = true
  kbError.value = ''
  try {
    const [detail, kbStats] = await Promise.all([kbsApi.getKB(kbId.value), kbsApi.getKBStats(kbId.value)])
    kb.value = detail
    stats.value = kbStats
  } catch (error) {
    kbError.value = describeError(error)
  } finally {
    loadingKb.value = false
  }
}

async function loadDocuments(silent = false): Promise<void> {
  if (!silent) loadingDocs.value = true
  docsError.value = ''
  try {
    const result = await documentsApi.listDocuments(kbId.value, {
      page: docPage.value,
      size: docSize.value,
      status: statusFilter.value || undefined,
    })
    documents.value = result.items
    docTotal.value = result.total
    docPages.value = result.pages
    docPage.value = result.page
    docSize.value = result.size
    syncPolling()
  } catch (error) {
    if (!silent) docsError.value = describeError(error)
  } finally {
    loadingDocs.value = false
  }
}

function handleDocPageChange(payload: { page: number; size: number }): void {
  docPage.value = payload.page
  docSize.value = payload.size
  void loadDocuments()
}

function handleStatusChange(): void {
  docPage.value = 1
  void loadDocuments()
}

/** el-upload 的 http-request 钩子：走我们自己的 axios 实例（带 Authorization 与统一错误信封） */
async function handleUpload(options: UploadRequestOptions): Promise<void> {
  const file = options.file
  try {
    const created = await documentsApi.uploadDocument(kbId.value, file)
    ElMessage.success(`「${created.filename}」已上传，正在后台解析`)
    docPage.value = 1
    await Promise.all([loadDocuments(), loadKb()])
  } catch (error) {
    // 契约里的错误码在这里很有用：FILE_TOO_LARGE / UNSUPPORTED_FILE_TYPE / DOCUMENT_DUPLICATE
    ElMessage.error(describeError(error))
  }
}

async function handleDeleteDocument(doc: DocumentOut): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `删除「${doc.filename}」会清掉它的 chunks、向量与 BM25 索引。确定继续吗？`,
      '删除文档',
      { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' },
    )
  } catch {
    return
  }

  try {
    await documentsApi.deleteDocument(doc.id)
    ElMessage.success('已删除')
    await Promise.all([loadDocuments(), loadKb(), refreshStats()])
  } catch (error) {
    ElMessage.error(describeError(error))
  }
}

async function refreshStats(): Promise<void> {
  try {
    stats.value = await kbsApi.getKBStats(kbId.value)
  } catch {
    /* 统计失败不影响主流程 */
  }
}

async function openChunks(doc: DocumentOut): Promise<void> {
  chunkDoc.value = doc
  chunkPage.value = 1
  chunkDrawerVisible.value = true
  await loadChunks()
}

async function loadChunks(): Promise<void> {
  if (!chunkDoc.value) return
  loadingChunks.value = true
  try {
    const result = await documentsApi.listChunks(chunkDoc.value.id, chunkPage.value, chunkSize.value)
    chunks.value = result.items
    chunkTotal.value = result.total
    chunkPages.value = result.pages
  } catch (error) {
    ElMessage.error(describeError(error))
  } finally {
    loadingChunks.value = false
  }
}

function handleChunkPageChange(payload: { page: number; size: number }): void {
  chunkPage.value = payload.page
  chunkSize.value = payload.size
  void loadChunks()
}

onMounted(async () => {
  await loadKb()
  await loadDocuments()
})

onBeforeUnmount(stopPolling)
</script>

<template>
  <div class="kf-page">
    <div class="kf-page-header">
      <div class="kf-row">
        <el-button link @click="router.push('/kbs')">← 知识库列表</el-button>
        <h2 class="kf-page-title">{{ kb?.name ?? '加载中…' }}</h2>
        <el-tag v-if="kb && !kb.is_active" type="info" size="small">已停用</el-tag>
        <!-- consistent = chunk_count == vector_count；不一致时必须肉眼可见 -->
        <el-tooltip
          v-if="stats"
          :content="
            stats.consistent
              ? `一致：chunk ${formatInt(stats.chunk_count)} = 向量 ${formatInt(stats.vector_count)}`
              : `不一致！chunk ${formatInt(stats.chunk_count)} ≠ 向量 ${formatInt(stats.vector_count)}，请检查入库流程`
          "
          placement="bottom"
        >
          <span
            class="consistency-dot"
            :class="stats.consistent ? 'consistency-dot--ok' : 'consistency-dot--bad'"
          />
        </el-tooltip>
      </div>

      <div class="kf-row">
        <el-button @click="loadKb(); loadDocuments()">刷新</el-button>
        <el-button v-if="kb" type="primary" plain @click="router.push('/search')">检索调试</el-button>
        <el-button v-if="kb" type="success" plain @click="router.push('/chat')">去问答</el-button>
      </div>
    </div>

    <ErrorBlock v-if="kbError" :message="kbError" @retry="loadKb" />

    <template v-else>
      <div class="stat-row">
        <div class="info-tile kf-card">
          <div class="info-tile__label">文档</div>
          <div class="info-tile__value">
            {{ formatInt(stats?.doc_count ?? kb?.doc_count) }}
            <small class="kf-muted">/ 就绪 {{ formatInt(stats?.ready_doc_count) }}</small>
          </div>
        </div>
        <div class="info-tile kf-card">
          <div class="info-tile__label">chunk 数</div>
          <div class="info-tile__value">{{ formatInt(stats?.chunk_count ?? kb?.chunk_count) }}</div>
        </div>
        <div class="info-tile kf-card" :class="{ 'info-tile--bad': stats && !stats.consistent }">
          <div class="info-tile__label">向量数</div>
          <div class="info-tile__value">
            {{ formatInt(stats?.vector_count) }}
            <small v-if="stats && !stats.consistent" class="info-tile__bad-text">不一致</small>
          </div>
        </div>
        <div class="info-tile kf-card">
          <div class="info-tile__label">BM25 文档数</div>
          <div class="info-tile__value">{{ formatInt(stats?.bm25_doc_count) }}</div>
        </div>
        <div class="info-tile kf-card">
          <div class="info-tile__label">总字符 / 总 token</div>
          <div class="info-tile__value">
            {{ formatInt(stats?.total_chars) }}
            <small class="kf-muted">/ {{ formatInt(stats?.total_tokens) }}</small>
          </div>
        </div>
        <div class="info-tile kf-card">
          <div class="info-tile__label">切分参数</div>
          <div class="info-tile__value">
            {{ kb?.chunk_size ?? '-' }}
            <small class="kf-muted">/ overlap {{ kb?.chunk_overlap ?? '-' }}</small>
          </div>
        </div>
      </div>

      <el-alert
        v-if="stats && !stats.consistent"
        class="inconsistent-alert"
        type="error"
        :closable="false"
        show-icon
        title="向量库与关系库数量不一致"
        :description="`chunks 表 ${formatInt(stats.chunk_count)} 条，向量库 ${formatInt(stats.vector_count)} 条。静默降级会让「列表看着成功、实际搜不到」，请先修复入库流程再看检索效果。`"
      />

      <el-alert
        v-if="kb"
        class="embedding-alert"
        type="info"
        :closable="false"
        show-icon
        title="embedding 快照（创建时锁定）"
        :description="`provider=${kb.embedding_provider} · model=${kb.embedding_model} · dim=${kb.embedding_dim}。换模型会换维度，混用会让检索彻底错乱，所以不支持在此处修改。`"
      />

      <div class="kf-card upload-card">
        <div class="kf-card-title">
          <h3>上传文档</h3>
          <span class="kf-muted">允许：md / markdown / txt / pdf / csv / json</span>
        </div>
        <div class="kf-card-body">
          <el-upload
            drag
            multiple
            :show-file-list="false"
            :http-request="handleUpload"
            accept=".md,.markdown,.txt,.pdf,.csv,.json"
          >
            <div class="upload-card__inner">
              <div class="upload-card__icon" aria-hidden="true">📤</div>
              <div class="upload-card__text">把文件拖到这里，或<em>点击选择</em></div>
              <div class="upload-card__hint kf-muted">
                上传后后台异步解析 → 切分 → 向量化，状态列会自动刷新
              </div>
            </div>
          </el-upload>
        </div>
      </div>

      <div class="kf-card">
        <div class="kf-card-title">
          <h3>文档列表</h3>
          <div class="kf-row">
            <el-select v-model="statusFilter" placeholder="全部状态" clearable style="width: 140px" @change="handleStatusChange">
              <el-option label="已上传" value="UPLOADED" />
              <el-option label="解析中" value="PARSING" />
              <el-option label="切分中" value="CHUNKING" />
              <el-option label="向量化中" value="EMBEDDING" />
              <el-option label="就绪" value="READY" />
              <el-option label="失败" value="FAILED" />
            </el-select>
            <el-tag v-if="hasProcessing" size="small" type="warning" effect="plain">处理中，自动刷新</el-tag>
            <el-button size="small" @click="loadDocuments()">刷新</el-button>
          </div>
        </div>

        <ErrorBlock v-if="docsError" class="kf-card-body" :message="docsError" @retry="loadDocuments()" />

        <LoadingBlock v-else-if="loadingDocs && documents.length === 0" height="200px" />

        <EmptyState
          v-else-if="documents.length === 0"
          icon="📄"
          title="还没有文档"
          description="上传第一个文档，系统会解析 → 切分（子块检索 / 父块喂模型）→ 向量化 → 写入 Chroma。"
        />

        <template v-else>
          <el-table :data="documents" size="small" class="kf-card-body">
            <el-table-column prop="filename" label="文件名" min-width="200" show-overflow-tooltip />
            <el-table-column prop="ext" label="类型" width="80" />
            <el-table-column label="大小" width="100">
              <template #default="{ row }: { row: DocumentOut }">{{ formatBytes(row.size_bytes) }}</template>
            </el-table-column>
            <el-table-column label="状态" width="120">
              <template #default="{ row }: { row: DocumentOut }">
                <el-tag :type="documentStatusTag(row.status)" size="small" effect="plain">
                  <span v-if="isDocumentProcessing(row.status)" class="doc-status__spin">◌</span>
                  {{ documentStatusText(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="页/字符" width="130">
              <template #default="{ row }: { row: DocumentOut }">
                <span class="kf-mono">{{ row.page_count ?? '-' }} / {{ formatInt(row.char_count) }}</span>
              </template>
            </el-table-column>
            <el-table-column label="chunk" width="80">
              <template #default="{ row }: { row: DocumentOut }">{{ formatInt(row.chunk_count) }}</template>
            </el-table-column>
            <el-table-column label="token" width="90">
              <template #default="{ row }: { row: DocumentOut }">{{ formatInt(row.token_count) }}</template>
            </el-table-column>
            <el-table-column label="入库耗时" width="100">
              <template #default="{ row }: { row: DocumentOut }">
                {{ row.ingest_ms === null ? '-' : formatDuration(row.ingest_ms) }}
              </template>
            </el-table-column>
            <el-table-column label="创建时间" width="170">
              <template #default="{ row }: { row: DocumentOut }">{{ formatDateTime(row.created_at) }}</template>
            </el-table-column>
            <el-table-column label="操作" width="150" fixed="right">
              <template #default="{ row }: { row: DocumentOut }">
                <el-button link size="small" type="primary" @click="openChunks(row)">chunk 预览</el-button>
                <el-button link size="small" type="danger" @click="handleDeleteDocument(row)">删除</el-button>
              </template>
            </el-table-column>
            <!-- 失败原因独占一行展示，避免被 tooltip 藏起来 -->
            <el-table-column type="expand" width="1">
              <template #default="{ row }: { row: DocumentOut }">
                <div class="doc-error">
                  <template v-if="row.error_code || row.error_message">
                    <el-tag type="danger" size="small" effect="plain">{{ row.error_code }}</el-tag>
                    <span>{{ row.error_message }}</span>
                  </template>
                  <span v-else class="kf-muted">无错误信息</span>
                  <div class="kf-mono kf-muted">parser: {{ row.parser ?? '-' }} · doc_id: {{ row.id }}</div>
                </div>
              </template>
            </el-table-column>
          </el-table>

          <PageBar
            :page="docPage"
            :size="docSize"
            :total="docTotal"
            :pages="docPages"
            @change="handleDocPageChange"
          />
        </template>
      </div>
    </template>

    <el-drawer v-model="chunkDrawerVisible" size="52%" :title="`chunk 预览 · ${chunkDoc?.filename ?? ''}`">
      <div class="chunk-drawer">
        <div class="kf-row">
          <span class="kf-muted">
            共 {{ formatInt(chunkTotal) }} 个 chunk（检索单位是子块，喂模型时用 parent_content 父块）
          </span>
        </div>

        <LoadingBlock v-if="loadingChunks" height="200px" />

        <EmptyState v-else-if="chunks.length === 0" icon="🧩" title="没有 chunk" description="该文档可能还在处理中，或解析后无有效文本。" />

        <div v-else class="chunk-list">
          <div v-for="chunk in chunks" :key="chunk.id" class="chunk-item kf-card">
            <div class="chunk-item__head">
              <el-tag size="small" type="primary" effect="dark">#{{ chunk.chunk_index }}</el-tag>
              <el-tag v-if="chunk.page_no !== null" size="small" effect="plain" type="info">P{{ chunk.page_no }}</el-tag>
              <span v-if="chunk.section_path" class="chunk-item__section kf-truncate" :title="chunk.section_path">
                📑 {{ chunk.section_path }}
              </span>
              <span class="kf-spacer" />
              <span class="kf-mono kf-muted">{{ formatInt(chunk.char_count) }} 字 / {{ formatInt(chunk.token_count) }} token</span>
            </div>
            <p class="chunk-item__content">{{ chunk.content }}</p>
            <div class="kf-mono kf-muted chunk-item__vector">vector_id: {{ chunk.vector_id }}</div>
          </div>
        </div>

        <PageBar :page="chunkPage" :size="chunkSize" :total="chunkTotal" :pages="chunkPages" @change="handleChunkPageChange" />
      </div>
    </el-drawer>
  </div>
</template>

<style scoped>
.stat-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}

.info-tile {
  padding: 12px 14px;
}

.info-tile--bad {
  border-color: var(--kf-danger);
}

.info-tile__label {
  font-size: 12px;
  color: var(--kf-text-secondary);
}

.info-tile__value {
  margin-top: 4px;
  font-size: 20px;
  font-weight: 600;
  color: var(--kf-text-primary);
  font-variant-numeric: tabular-nums;
}

.info-tile__value small {
  font-size: 12px;
  font-weight: 400;
}

.info-tile__bad-text {
  color: var(--kf-danger);
  font-size: 12px;
}

.consistency-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.consistency-dot--ok {
  background: var(--kf-success);
}

.consistency-dot--bad {
  background: var(--kf-danger);
  animation: consistency-blink 1s steps(2, start) infinite;
}

@keyframes consistency-blink {
  50% {
    opacity: 0.25;
  }
}

.inconsistent-alert,
.embedding-alert {
  margin-top: 16px;
}

.upload-card__inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 16px;
}

.upload-card__icon {
  font-size: 28px;
}

.upload-card__text em {
  color: var(--kf-primary);
  font-style: normal;
}

.upload-card__hint {
  font-size: 12px;
}

.doc-status__spin {
  display: inline-block;
  animation: doc-spin 1.2s linear infinite;
}

@keyframes doc-spin {
  to {
    transform: rotate(360deg);
  }
}

.doc-error {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 12px;
  font-size: 12px;
  color: var(--kf-text-regular);
}

.chunk-drawer {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.chunk-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 62vh;
  overflow: auto;
}

.chunk-item {
  padding: 10px 12px;
}

.chunk-item__head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.chunk-item__section {
  font-size: 12px;
  color: var(--kf-text-secondary);
  max-width: 260px;
}

.chunk-item__content {
  margin: 6px 0 0;
  font-size: 13px;
  line-height: 1.7;
  color: var(--kf-text-regular);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow: auto;
}

.chunk-item__vector {
  margin-top: 6px;
  font-size: 11px;
}
</style>
