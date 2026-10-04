# KnowFlow 前端

企业知识库 RAG + LangGraph Agent 问答控制台。Vue 3 + Vite + TypeScript(strict) + Pinia + Element Plus + ECharts。

> 接口、字段名、事件名一律以 [`docs/01-数据库与接口契约.md`](../docs/01-数据库与接口契约.md) 为准。
> 本 README 末尾单独列出「**契约里有歧义 / 缺失、前端做了假设的地方**」，供后端对齐。

---

## 一、启动

```bash
cd frontend
pnpm install          # 用了 pnpm，不要用 npm
cp .env.example .env.local   # 可选；默认 /api/v1 已够用
pnpm dev              # http://127.0.0.1:5173
```

后端需先跑在 `http://127.0.0.1:8000`（开发环境由 Vite proxy 转发 `/api`，无跨域问题）。

### 其它脚本

| 命令 | 作用 |
| --- | --- |
| `pnpm build` | **类型检查（`vue-tsc --noEmit`）+ 生产构建**，类型报错会让构建失败 |
| `pnpm typecheck` | 只做类型检查 |
| `pnpm preview` | 预览 `dist/` 产物 |
| `node scripts/sse-selftest.mjs` | SSE 增量解析自测（不需要浏览器，14 项断言） |

### 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `/api/v1` | API 基址；开发环境走 Vite proxy |
| `VITE_APP_TITLE` | `KnowFlow` | 顶栏标题 |

---

## 二、页面清单

| 路由 | 页面 | 说明 |
| --- | --- | --- |
| `/login` | `LoginPage` | 登录 / 注册（登录守卫免检路由）。第一个注册的用户自动成为 admin |
| `/kbs` | `KBListPage` | 知识库列表：搜索、分页、新建 / 编辑 / 删除、设为「当前 KB」 |
| `/kbs/:id` | `KBDetailPage` | 文档管理：拖拽上传、状态轮询、删除、**chunk 预览抽屉**、`consistent` 红点告警 |
| `/chat` · `/chat/:conversationId` | `ChatPage` | 智能问答：会话列表 + 消息流（`[n]` 可点击角标）+ 引用卡片 + **思考过程面板** + 本轮统计 |
| `/search` | `SearchDebugPage` | 检索调试：**全部参数可调**，结果表并列展示 vector / bm25 / rerank 分数与排名、闸门判定、召回漏斗 |
| `/eval` | `EvalPage` | 数据集 + 运行列表 + **消融对比**（纯向量 / 纯 BM25 / 混合 / 混合+重排） |
| `/eval/runs/:id` | `EvalRunDetailPage` | 指标卡 + 逐条结果表（含 retrieved_json 快照）+ 与另一个 run 的对比（delta + 95% CI + 显著性） |
| `/obs` | `ObsPage` | 请求量 / token / 成本(USD+CNY) / 延迟分位卡 + 质量卡 + 三张趋势图 + 按 mode/model 分组 + trace 列表 |
| `/obs/traces/:traceId` | `TraceDetailPage` | span **瀑布图**（靠 `start_offset_ms` + `duration_ms` 定位，颜色按 `span_type`）+ input/output JSON + 关联消息 |

---

## 三、目录结构

```
frontend/
├─ index.html
├─ vite.config.ts          # proxy: /api,/health,/ready,/metrics -> 127.0.0.1:8000
├─ tsconfig.json           # strict: true
├─ tsconfig.node.json
├─ env.d.ts
├─ .env.example
├─ scripts/
│  └─ sse-selftest.mjs     # 纯 Node 跑的 SSE 解析自测
└─ src/
   ├─ main.ts              # Pinia → Router → ElementPlus 的挂载顺序（有注释说明为什么）
   ├─ App.vue
   ├─ styles/index.scss    # CSS 变量 + 暗色（跟随系统 / html.dark 手动切换）
   ├─ router/index.ts      # 路由表 + 登录守卫
   ├─ api/
   │  ├─ client.ts         # axios：注入 Bearer、解包错误信封、401 只处理一次
   │  ├─ sse-parser.ts     # ★ 纯解析函数 parseSSEChunk（可在 Node 里测）
   │  ├─ sse.ts            # ★ fetch + ReadableStream 的流式客户端（9 种事件 + 取消）
   │  ├─ auth.ts kbs.ts documents.ts search.ts chat.ts conversations.ts eval.ts obs.ts
   ├─ stores/              # auth（token/user）、kb（列表 + 当前 KB）、chat（会话/消息/流式状态）
   ├─ types/               # dto.ts（出参）、models.ts（入参）、sse.ts（事件）、sse-guards.ts（运行期收窄）
   ├─ composables/         # useChatStream（串 sse + chat store）、useAsync、useECharts
   ├─ components/          # AppLayout、MessageBubble、CitationCard、AgentTracePanel、
   │                       # EmptyState、LoadingBlock、ErrorBlock、PageBar、StatCard、LineChart、BarChart
   ├─ utils/               # format.ts（dayjs 等）、citation.ts（[n] 切分）
   └─ pages/               # 9 个页面
```

---

## 四、几个关键实现说明

### 1. 为什么 SSE 要自己解析

`POST /api/v1/chat/stream` 需要带 `Authorization` 头**且是 POST**，浏览器原生 `EventSource` 两者都不支持。
所以走 `fetch` + `response.body.getReader()`：

- `TextDecoder(..., { stream: true })` 增量解码，一个 UTF-8 汉字被字节切断也不会乱码；
- 按 `\n\n` 分帧，`data:` 跨多行时用 `\n` join 再 `JSON.parse`（SSE 规范）；
- 兼容 CRLF（sse-starlette 默认发 `\r\n`）、注释心跳行、未知事件名（忽略而不崩）；
- 9 种事件全部支持：`meta` `trace` `tool` `sources` `reflect` `token` `done` `error` `end`；
- `AbortController` 取消（「停止生成」按钮），取消不会弹红色错误提示；
- 两类异常给可读错误：**连接中断**（没收到 `end`/`done` → `STREAM_INTERRUPTED`）、
  **非 2xx 且响应体是错误信封 JSON**（手工解包出 `code`/`message`/`request_id`）。

纯解析函数抽到 `src/api/sse-parser.ts`，`scripts/sse-selftest.mjs` **直接 import 这份在浏览器里跑的同一份源码**
（Node 24 原生类型擦除），喂入**在 `data:` 中间切断**的分片做验证，零复制粘贴。

### 2. `[n]` 引用角标（不用 `v-html`）

`utils/citation.ts` 的 `splitCitations()` 把回答按 `/\[(\d+)\]/g` 切成「文本节点 / 角标节点」，
模板里分别渲染成 `<span>` 与 `<button>`。只有出现在 `sources[].rank` 里的编号才变角标，
避免正文里的 `[2024]` 被误判。点击后由 `ChatPage` 滚动并高亮右侧对应 `CitationCard`。

### 3. Agent「思考过程」

`AgentTracePanel` 按到达顺序渲染时间线：`trace`（节点名 + 中文说明 + 耗时横条）、
`tool`（工具名 + 命中数）、`reflect`（第几轮 + 通过与否 + **未支撑句子列表**）。
默认折叠；流式过程中 `activeNode` 变化会**自动展开当前节点**并高亮圆点。

### 4. 统一错误信封与 401

- 所有非 2xx 在 `api/client.ts` 的响应拦截器里解包成 `ApiError`（`code` / `status` / `requestId` / `detail`）；
- **422 单独兼容** Pydantic 的 `{detail:[{loc,msg,type}]}`，渲染成 `body.question: 字段必填` 这种可读文案；
- 网络不通（后端没起）给的是「无法连接后端服务」而不是 `Network Error`；
- 401：清 token → 跳 `/login` → 提示「登录已过期，请重新登录」，用模块级 `handling401` 标记保证**并发请求只处理一次**；
- 页面 toast 用 `describeError()`，自动把 `request_id` 带出来便于排障。

### 5. 分页

契约里所有列表都是 `{items,total,page,size,pages}`，`PageBar` 把 `pages` 映射到 `el-pagination` 的 `page-count`。
（注意 `/eval/datasets` 在契约里返回的是**数组**而不是 `Page`，代码按数组处理。）

### 6. 挂载顺序（Vue 3 常见坑）

`main.ts` 里**必须先 `app.use(createPinia())` 再 `app.use(router)`**，否则守卫/路由组件里 `useStore()` 会抛
`getActivePinia() was called but there was no active Pinia`。代码里有注释说明；路由守卫本身直接读
`localStorage` 的 token（`api/client.ts` 的 `getToken()`），既没有循环依赖也不受挂载顺序影响。

---

## 五、契约里有歧义 / 缺失、前端做了假设的地方

> 这一节是为了和后端对齐用的。
>
> **校验状态**：前端在实现后读到了后端的 Pydantic schema（`backend/src/knowflow/schemas/*.py`），
> 下表的 ✅ 表示**已按后端 schema 逐字对齐**（不再是猜的），⚠️ 表示**契约文档里确实没写、后端 schema 里也仍未提供**，
> 前端做了容错降级并在此列出，仍未提供的那几项会影响体验但不影响功能。
> **没有一项是靠猜字段名硬编码的**：凡是拿不到的数据，前端显示 `—` 或空列表，绝不崩。

### 5.1 ✅ 已按后端 schema 对齐（契约缺、后端已定义）

| # | 位置 | 契约的问题 | 后端的实际定义 | 前端做法 |
| --- | --- | --- | --- | --- |
| 1 | 5.9 `ObsStatsOut` | 只给了**语义**，没给嵌套字段名 | `{hours, requests, tokens:{prompt,completion,total}, cost_usd, cost_cny, latency:{p50,p95,p99,max,avg}, by_mode:[{mode,requests,tokens,cost_usd,avg_latency_ms,refusal_rate}], by_model:[{model,requests,prompt_tokens,completion_tokens,cost_usd,avg_latency_ms}], timeline:[{bucket,requests,tokens,cost_usd,avg_latency_ms,refusal_rate}]}` | 全部按此对齐。`api/obs.ts` 只保留「数组缺失 → 空数组」的兜底 |
| 2 | 5.9 时间序列的键 | 未指定 | `timeline[].bucket`（时间桶起点，UTC ISO8601 带 Z） | 横轴直接用 `bucket` 格式化 |
| 3 | 5.9 `series` 里有 P95 吗 | 未说明 | `timeline` 点里只有 `avg_latency_ms`，**没有 P95** | 「延迟趋势」图画的是**平均延迟**；P50/P95/P99 在顶部延迟卡里 |
| 4 | 5.9 质量接口 | 契约没写 `hours`/`total` | `QualityOut` 多了 `hours` 与 `total`（样本数） | 质量卡副标题显示「样本 N 次请求」 |
| 5 | 5.8 `AblationGroup` | 只写了名字，字段全缺 | **扁平**字段：`{mode, label, top_k, run_id, case_count, hit_rate, recall_at_k, mrr, ndcg, avg_latency_ms, delta_vs_baseline}`（不是 `group.metrics.*`） | 表格与柱状图都按扁平字段读；`label` 用作图表分类名，`delta_vs_baseline.recall_at_k` 单独列一列 |
| 6 | 5.8 `AblationResponse` | 契约只写 `{dataset_id, groups}` | 还有 `top_k`、`baseline_mode` | 前端未展示 `baseline_mode`，但类型已声明 |
| 7 | 5.8 `compare.delta` | 没写 delta 的键 | `delta: dict[str, float]`，键是指标名，语义是 **B − A** | 按 `Record<string, number>` 索引；表里把键映射成中文标签 |
| 8 | 5.8 `ci95` | 契约写 `{low, high}` | 后端是 `dict[str, float]`，实际仍是 `{low, high}` | 按 `{low, high}` 读 |
| 9 | 5.8 `DatasetOut` | — | **没有** `created_at` / `updated_at` | 数据集卡片不显示创建时间（原先假设有，已去掉） |
| 10 | 5.6 `reflect.unsupported` | 契约写 `[...]`，像是字符串数组 | 后端是 **对象数组** `[{sentence, reason}]`（裸字符串会被 validator 包成 `{sentence}`） | 按对象渲染，展开细节时显示「句子 —— 原因」；解析层兼容裸字符串 |
| 11 | 5.6 `trace.status` | 没说取值 | 只有 `ok` / `error`（**没有 running**）：每个节点在结束时产出一帧并带耗时 | `AgentTracePanel` 不再假设有「开始/结束」两帧；同名节点重复出现且耗时不同才算新的一轮 |
| 12 | 5.6 `tool.args` | 契约写 `args`，未说类型 | `Any \| None`（不保证是对象） | 解析层按 `unknown` 收下，渲染层对对象/字符串分别处理 |
| 13 | 5.7 `FeedbackRequest` | 契约只写 `{rating, comment?}` | **后端多加了一条规则：`rating=-1` 时 `comment` 必填**（校验失败会 422） | 点踩时弹输入框要求填原因，`inputValidator` 保证非空；取消则不提交 |
| 14 | 5.4 `SearchRequest` | 契约列了参数但没写相互关系 | **后端校验 `fetch_k >= top_k`**（否则 422） | UI 上联动纠偏：改 `top_k` 会把 `fetch_k` 顶上去，`fetch_k` 的 `min` 绑定 `top_k` |
| 15 | 5.8 `DatasetCreateRequest.cases` | 契约写 `cases: [...]` | **后端要求至少 1 条**（`min_length=1`） | 提交前检查解析出的用例数，为 0 直接拦下并提示 |
| 16 | 5.5 `EvalMetrics` 的可空性 | 契约的 `EvalMetrics` 看起来全是必填数字 | 除 `case_count` 外**全部可为 `null`**，语义是「没算」（LLM 判官关闭时 faithfulness 无从计算） | 类型改成 `number \| null`，展示成 `—` 而不是 `0` |
| 17 | 5.7 `MessageOut.citations` | 契约只写了表的字段 | `CitationOut = {id, message_id, chunk_id, doc_id, rank, score, snippet}`——**确实没有** `doc_name`/`page_no`/`section_path` | 见下方 ⚠️ 第 1 条 |
| 18 | 5.10 `/health` | 契约列了字段但没说层级 | `db:{ok,dialect,version,error}`、`pool:{size,checkedin,checkedout,overflow,total}`、`warnings[]` | 顶栏用 `offline` / `embedding_mode` / `warnings` 如实显示降级状态（后端 `status` 可为 `ok/degraded/error`，前端只按 `offline` 与 `embedding_mode` 判断） |
| 19 | 5.9 `TraceDetailOut.messages` | 契约写 `messages`，未给类型 | 后端是 `list[dict[str, Any]]`（**裸字典，不是 `MessageOut`**） | 前端按 `MessageOut` 的形状读并全部容错（字段缺失只是不显示） |

### 5.2 ⚠️ 仍未解决（契约没写、后端也还没提供）

| # | 位置 | 问题 | 影响 | 前端的容错 / 建议 |
| --- | --- | --- | --- | --- |
| 1 | 5.7 `MessageOut.citations` | 没有 `doc_name` / `page_no` / `section_path`，而引用卡片要显示「文件名 / 页码 / 小节路径」 | **历史消息**（刷新页面后）的引用卡片只能显示 `文档 #doc_id` + 分数 + 片段；**只有流式当次**（`sources` 事件）能显示完整信息 | 类型里这三个字段是**可选**的：后端在 `MessageOut.citations` 里补上 `doc_name`/`page_no`/`section_path`（JOIN `chunks` + `documents` 即可）后，前端**无需改动**就会自动显示。建议补 |
| 2 | 5.8 评测运行 | 契约里**没有 `DELETE /eval/runs/{id}`** | 前端不提供删除，点击时弹窗说明契约未定义该接口 | 如需删除请补接口 |
| 3 | 5.8 `POST /eval/datasets` | 契约写「上传 `{...}` **或** JSONL 文件」，没有明确的 multipart 端点定义 | 前端只实现了 JSON body 方式 + 文本框粘贴（每行一条：问题 + 可选 Tab/`\|` 分隔标准答案） | 若要支持 JSONL 文件上传，需要补端点与字段名 |
| 4 | 5.7 `/conversations/{id}/messages` | 没说排序方向与单次上限 | 前端按 `created_at` 升序 + `size=100` 取最近一屏；超长会话只显示最后一屏 | 需要上滑分页时前端再改（接口已支持 `page/size`） |
| 5 | 5.7 `meta` 事件 | 不含会话标题 | 新建会话后要再拉一次 `/conversations` 才能看到标题 | 已实现（`onMeta` 里刷新列表）；若 `meta` 带上 `title` 可省一次请求 |
| 6 | 5.4 `SearchResponse` | `gate` 只在顶层，hit 里没有 per-hit 闸门判定 | 表格里没有 per-hit「是否被闸门拦下」列 | 顶层的 `gate` 已完整展示；如需 per-hit 判定请补字段 |
| 7 | 5.10 `/health` | 没有「错误率」类字段 | 可观测页的请求量卡片不再显示错误次数（原先假设有 `error_count`，已去掉） | `traces?status=error` 可近似看到错误链路 |
| 8 | 5.1 认证 | 契约没写登出接口、也没写 refresh token | 前端登出只清本地 token（不发请求）；token 过期后靠 401 跳登录 | 若后端加 `POST /auth/logout` 或 refresh，前端再接 |
| 9 | 5.9 `ObsStatsOut.requests` | 没有「按小时补齐的空桶」相关说明 | `timeline` 若只包含有数据的桶，折线图会出现时间跳变 | 前端按后端返回的桶原样画（不做补齐）；建议后端补零桶 |

---

## 六、已知取舍（诚实清单）

| 取舍 | 现状 | 如果要改 |
| --- | --- | --- |
| 会话消息一次取 100 条 | 超长会话只显示最近一屏 | 改成上滑分页（`/messages` 已支持 `page/size`） |
| **没有联调测试** | 后端当时还没跑起来，只做了类型级 + 解析级验证 | 后端起来后按 README 第五节逐项回归，重点是 SSE 9 种事件与错误信封 |
| trace 列表 30 秒轮询 | 页面开着就会一直请求 | 换成 SSE / WebSocket 推送，或加「自动刷新」开关 |
| 上传没有进度条 | 解析是后端异步任务，传输进度意义有限 | 传 `onUploadProgress` 至少显示上传阶段进度 |
| 没有单元测试框架 | 只有 `scripts/sse-selftest.mjs` 覆盖最容易出错的 SSE 分帧 | 引入 vitest，把 `utils/citation.ts`、`types/sse-guards.ts`、`stores/chat.ts` 也覆盖上 |
| 暗色主题 | 跟随系统 + 顶栏手动切换（存 localStorage） | 若要与 Element Plus 的暗色变量体系完全统一，改用 `@element-plus/theme-chalk/dark` |
| `element-plus` / `echarts` 体积大 | 已手动分包（`element-plus` 917KB、`echarts` 614KB，gzip 后 296KB / 208KB） | 需要更小可改成按需引入（`unplugin-vue-components` + `unplugin-auto-import`） |
