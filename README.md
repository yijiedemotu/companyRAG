# KnowFlow

> **企业知识库 RAG + LangGraph Agent 问答服务。**
> 上传制度/手册/表格 → 混合检索（向量 + 自研 BM25 + RRF 融合 + 重排 + 相关性闸门）→
> CRAG/Self-RAG 状态图编排 → 带 `[n]` 引用的回答，且**每一个数字都能在 trace 里查到出处**。

```
你  一线城市住宿标准是多少

    [思考过程]
    ✓ analyze    判断为「事实型 + 需检索」
    ✓ retrieve   向量召回 20 条 · BM25 召回 12 条 · RRF 融合 20 条 · autocut 后 12 条
    ✓ sources    员工报销制度.md › 差旅报销标准（score 0.722 · 向量 0.732 · BM25 31.30）

    [回答]
    一线城市住宿标准为每晚 600 元 [1]。

    [1] 员工报销制度.md › 差旅报销标准
        "…住宿标准（每晚）一线城市 600 元，二线城市 450 元…"

    实测：检索 177 ms（其中向量化 168 ms · 向量查询 3 ms · BM25 0 ms · 重排 1 ms）
         问答 573 ms · 4304 tokens · $0.000696
```

> ⚠ **上面这组数字全部来自本机实测**（`docs/03-实测证据.md` 的 E6.3，可用
> `.\tasks.ps1 smoke-http` 复现）。
> 但**答案文本本身是"真实大模型生成"的效果，本机未实测**（开发机未配 API Key，
> 离线模式走的是抽取式 Mock 回答）—— 所以上面那段流畅的中文是**预期效果**，
> 不是本机跑出来的原文。这一点在「已知边界」里也标注了。

回答里的 `[n]` 不是装饰：它落到 `message_citations.chunk_id`，能反查原文、小节路径与页码。
点开右边就能看到上面那些分数——**从提问到引用到 token 成本，整条链路可回溯**。

---

## 一、它解决什么问题

把文档塞进 prompt 的 demo 是这样的：

```python
prompt = open("员工报销制度.md").read() + question   # ← 迟早会炸
```

四个必然翻车的地方，**换更强的模型一个都救不了**：

| 问题 | 为什么换模型没用 |
| --- | --- |
| **塞不进去** | 制度文档 3 万字，企业库 3000 份。上下文窗口再大也是有限资源，而且**越长越贵越慢**——成本随 token 线性涨，延迟随长度涨 |
| **找不到** | 90% 的「答不准」死在检索，不在生成。模型只是把手里那份资料讲清楚；**手里那份是错的，它就认真地错** |
| **答了没依据** | 检索到相似但无关的片段时，模型会自信地编。字面相似 ≠ 能回答这个问题（实测：BGE-M3 上无关文本的余弦相似度基线约 **0.57**，不是 0） |
| **坏了没人知道** | 换个切分参数、换个嵌入模型，效果是变好还是变坏？靠「我感觉好点了」改参数，改坏了也发现不了 |

KnowFlow 把这四个问题各自做了一层，而不是指望模型自己搞定：

- **塞不进去** → 切分 + 父子块 + 上下文预算（`ingest/chunkers.py`、`retrieval/autocut.py`）
- **找不到** → 混合检索 + 重排 + 融合（`retrieval/`）
- **没依据** → 相关性闸门 + CRAG 判官 + Self-RAG 反思（`retrieval/autocut.py`、`agent/nodes.py`）
- **没人知道** → 全链路 trace + 可回归评测集（`observability/`、`evaluation/`）

> 一句话：**这个项目的重点不是「调用大模型」，是把大模型放进一条可观测、可兜底、可测试的流水线。**

---

## 二、核心亮点（每条对应到代码位置）

### 1. 混合检索：向量 + 自研 BM25 + RRF 融合

三路信号的盲区互补——稠密向量强在**同义改写**（"出差住房能报多少" ≈ "住宿标准"），
稀疏 BM25 强在**精确词面**（错误码、专有名词、型号）。

关键细节：**中文字段不能用空格分词**。`一线城市住宿标准` 用 `split()` 切出来是 **1 个词**，
BM25 直接失效。本项目用 **1-gram + 2-gram 混合切分**（不引 jieba）：

```
一线城市住宿标准 → 一 线 城 市 住 宿 标 准 一线 线城 城市 市住 住宿 宿标 标准
                    └─ 1-gram 保证不漏 ─┘  └─ 2-gram 提供区分度 ─┘
```

融合默认走 **RRF**：`score = Σ 1/(k + rank)`，只用排名不用分数，对某一路的**分数尺度漂移免疫**
（向量余弦在 `[0,1]`，BM25 无上界——两个量纲不能直接加权）。

- `retrieval/bm25.py` — 自研 BM25 倒排索引（约 120 行，k1=1.5 / b=0.75）
- `retrieval/text.py` — 中文 1/2-gram 切分与查询词覆盖率
- `retrieval/fusion.py` — RRF 与加权 min-max 归一化**两种都实现**（配置可切）
- `retrieval/engine.py` — 四模式编排：`vector` / `bm25` / `hybrid` / `hybrid_rerank`

### 2. 重排 + 相关性闸门：防幻觉的最后一道

**重排**把「相关的」排到「相似的」前面：有 API Key 时用 LLM 打分，没有时退到
「查询词覆盖率 + 标题命中」的启发式兜底——链路不中断。

**闸门**是双阈值：`向量相似度 ≥ 阈值` **或** `查询词覆盖率 ≥ 阈值` 才算「有料」。
两条都不过就**不调大模型**，直接回「知识库中没有找到相关内容」。
省钱是一方面，更重要的是**防幻觉**：模型手里没资料时最容易编，工程上把钱花在
「确保它手里有对的资料」比花在「让它别编」有效得多。

> ⚠️ 阈值不能拍脑袋。实测 BGE-M3 上「无关文本」的相似度基线约 0.57，
> 直觉设成 0.35 等于闸门完全失效。`scripts/calibrate_threshold.py`
> 在 43 条标注用例上扫参数，输出「阈值 vs 召回率/误放行率」曲线取 F1 最优点。
> 结果落在 `data/threshold_calibration.json`。

- `retrieval/rerank.py` — LLM 重排 + 启发式降级
- `retrieval/autocut.py` — 分数断崖截断 + 双阈值闸门 + 上下文预算裁剪

### 3. Agent 是 LangGraph 状态图，不是「把控制流交给模型」

控制流画成图，模型只在**每个节点内部**做判断。既有自适应的好处，又保证每一步可观测、可兜底、可测试。

```
START → analyze ─┬─(不需要检索)──────────────────────────→ generate → reflect ─┬─→ END
                 │                                                             │
                 └─(需要检索)→ retrieve → grade ─┬─(不相关 & 次数<上限)→ rewrite ─┘
                                                │                          └→ retrieve（回到上面）
                                                └─(相关)→ generate → reflect ─(引用不通过 & 次数<上限)→ generate
```

三个刻意设计：

- **`grade` 用 LLM 判定而不是复用检索分数**：检索分数是「和 query 的相似度」，
  不等于「能不能回答这个问题」。实测经常出现「高分片段答的是另一个问题」。
  代价是多一次便宜的小模型调用 + 300~800ms，所以做成开关 `AGENT_GRADING_ENABLED`。
- **`rewrite` 有次数上限（默认 2）**：防死循环烧钱。上限用完还没料就**明确拒答**——
  宁可说「知识库中没有足够信息」，也不编。
- **`reflect` 不重新检索**：职责分离。`grade` 管「料找对没」，`reflect` 管「话有没有依据」。
  混在一起会「越反思越跑偏」。

- `agent/graph.py` — `StateGraph` 装配、条件边、checkpointer
- `agent/nodes.py` — 六个节点的实现与判官调用
- `agent/state.py` — 状态定义（含 `Annotated[..., operator.add]` 的累加语义）
- `agent/runner.py` — 把节点事件映射成 SSE 帧

### 4. 会话记忆：LangGraph checkpointer + 双重截断

多轮记忆不该只靠「把历史塞进 prompt」。用 LangGraph 的 checkpointer 按
`thread_id = conversation_id` 持久化状态，**服务重启会话不丢**；
再叠加「**滑动窗口 + 字符预算**」双重截断（取更严者，从最新往回装、装不下就停）控制成本。

再加一条工程约束：**记忆读写要线程安全**——FastAPI 的 async 路由与后台线程会同时碰同一会话。

- `agent/memory.py` — SQLite checkpointer、双重截断、线程安全会话记忆

### 5. 全链路可观测：回答「为什么慢」和「花了多少钱」

「零侵入」实现：`@traced("retrieve")` 装饰器 + contextvar 绑定 `trace_id`，
业务代码不写埋点也能被记录。

三样东西必须能回答：

1. **这次请求为什么慢？** → `trace_spans` 瀑布图，每个节点耗时
2. **这个功能一天花多少钱？** → `messages.token_* + cost_usd` 按天聚合，USD/CNY 双币
3. **线上质量有没有退化？** → P50/P95 延迟 + 拒答率 + 引用数分布

`middleware/context.py` 给每个请求发 `request_id`，并把它注入**每一条日志**——
排查线上问题时，拿到用户截图里的 `X-Request-Id` 就能捞出该请求的全部日志。

- `observability/tracing.py` — span 树、瀑布图数据
- `observability/pricing.py` — token → 成本（`DECIMAL(12,6)`，不用 FLOAT）
- `observability/metrics.py` — 聚合统计 + Prometheus 文本
- `middleware/context.py` — request_id / 耗时头 / 访问日志

### 6. 评测：把「感觉更好了」变成数字

- **标注集**：`data/eval_cases.jsonl` 共 **43 条**，覆盖三类难点——同义改写、专有名词、多跳
- **检索指标**：`recall@k`、`MRR`、`nDCG@k`
- **生成指标**：`faithfulness`、`answer_relevance`、`citation_precision`
- **消融实验**：纯向量 / 纯 BM25 / 混合 / 混合+重排，四组同一数据集跑出对比表
- **统计显著性**：bootstrap 抽样给 95% 置信区间，避免「提升 1.2%」其实是噪声
- **CI 门槛**：`recall@10` 低于基线即失败，防止改坏

- `evaluation/metrics.py` — 检索/生成指标（纯函数，好测）
- `evaluation/stats.py` — bootstrap 置信区间 + 显著性检验
- `evaluation/harness.py` — 消融实验编排
- `evaluation/datasets.py` — 标注集载入与校验

### 7. 降级路径与主路径同等对待（`/health` 不骗人）

`EMBEDDING_PROVIDER=hash` + 无 `OPENAI_API_KEY` 时，全链路依然跑通：
哈希向量兜底、抽取式回答。这让 **clone 下来零成本就能演示**，CI 里不花一分钱。

代价是要额外维护 `/health` **如实上报当前真实模式**——离线时 `offline=true`，
embedding 降级时 `embedding_mode="hash(fallback:local_load_failed)"`，
BM25 索引没重建时 `bm25_doc_count=0`。

> 这是本项目的一条硬规则：**允许降级，不允许瞒报。**
> 静默降级（服务不报错、不掉线，只是检索质量悄悄变差）是最难查的一类 bug。

- `embeddings/factory.py` — 主路径失败 → 记录原因 → 降级，并把原因暴露给 `/health`
- `embeddings/hash_embedder.py` — 零依赖确定性兜底（hashing trick，带符号位降低碰撞破坏力）
- `api/routes/ops.py` — `/health` `/ready` `/metrics`

### 8. 鉴权「默认需要、例外公开」——这是修出来的，不是设计出来的

写 HTTP 冒烟时发现：**`GET /kbs` 不带 token 竟然返回 200 + 正常业务数据**。
写脚本审计后发现 **39 个端点里有 21 个没鉴权**，包括
`GET /kbs/{id}`、`GET /documents/{id}`、`GET /kbs/{id}/stats`、`POST /kbs/{id}/search`、
`GET /eval/*`（8 个）、`GET /obs/*`（4 个）。

**它们全都返回 200 + 正常数据，所以任何"断言字段对不对"的功能测试都发现不了** ——
只有「不带 token 打一次」才会暴露。

修法不是逐个补签名（下次加端点还会忘），而是改成
**路由级"默认需要、例外公开"**：`include_router(..., dependencies=[Depends(get_current_user)])`，
让**将来新增的端点自动受保护**，确实要公开的（注册/登录、运维探针）才显式排除。
并留下一个读 **FastAPI 真实依赖图**的审计脚本当 CI 门禁：**34 个受保护 / 2 个刻意公开 / 0 漏洞**。

> 这个脚本自己也踩了两个坑：**只扫端点自身的依赖会误报**（路由级依赖挂在
> `include_context.dependencies` 上）；**直接遍历 `app.routes` 只看得到 `_IncludedRouter` 占位对象**
> （FastAPI 0.142 的 `include_router` 是惰性的，真正的 `APIRoute` 在 `original_router.routes` 里）——
> 第一版脚本只找到 1 个端点，**差点得出"没有漏洞"的错误结论**。
> 见 `scripts/audit_auth.py` 与 `docs/03-实测证据.md` 的 E6.4。

### 9. 四层验证：为什么不能只有 pytest

| 层 | 命令 | 规模 | 抓到过什么 |
| --- | --- | --- | --- |
| 单元/集成 | `tasks.ps1 test` | **222 个用例 / 4.2 秒** | 边界条件、协议契约、降级分支 |
| 链路冒烟 | `tasks.ps1 smoke` | 41 项断言 | 入库一致性、四种检索、拒答短路 |
| **HTTP 冒烟** | `tasks.ps1 smoke-http` | **70 项断言** | **21 个端点漏鉴权**；**流式 contextvars 跨线程崩溃** |
| 鉴权审计 | `tasks.ps1 audit` | 39 个端点 | 防止将来再加出未鉴权端点 |

**pytest 走 `ASGITransport`，不经过真实服务器、真实中间件栈，也不会触发
Starlette 对同步生成器的分次线程调度。** 上面两个最难的 bug 都是
**在 pytest 全绿的情况下**只被 HTTP 冒烟抓到的 —— 这就是为什么必须真起一次服务。

---

## 三、架构（7 层）

市面上多数 demo 只有 3 层：`前端 → 一个 /chat → 调大模型`。能上生产的是 7 层，**缺哪一层就会在某个场景崩掉**。

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ ① 接入层            frontend/ (Vue3 + Vite + TS)   │  api/ + middleware/         │
│                     Pinia · Element Plus · ECharts │  FastAPI · SSE · JWT · 限流 │
│                     「缺了它」前端拿不到稳定契约，出问题无法定位是哪个请求          │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ② 编排层（Agent）    agent/  —— LangGraph StateGraph                             │
│                     analyze → retrieve → grade →(rewrite→retrieve)* → generate    │
│                              → reflect →(generate)*                              │
│                     「缺了它」只能一问一答，多跳必错，错了也不会自我纠正            │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ③ 检索层            retrieval/  —— 混合检索 + 融合 + 重排 + 闸门                   │
│                     向量(BGE-M3) ┐                                               │
│                     BM25(自研)   ├→ RRF 融合 → 重排 → 双阈值闸门 → 上下文预算     │
│                                  ┘                                               │
│                     「缺了它」检索不准 → 模型再强也答不准（90% 的答不准死在这里）  │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ④ 模型层            llm/ + embeddings/  —— 对话模型 + 向量化模型（都可换）         │
│                     OpenAI 兼容协议：换 DeepSeek/通义/智谱/vLLM/Ollama 只改一行     │
│                     「缺了它」被单一厂商锁死；换模型要改业务代码                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ⑤ 数据层            db/ + vectorstore/ + data/uploads/                           │
│                     MySQL 8（业务事实）· Chroma（语义索引）· 文件（原文件）        │
│                     「缺了它」数据不一致；重启后索引丢失且无人知道（静默降级）      │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ⑥ 可观测层          observability/  —— trace 树 · token 成本归因 · P50/P95        │
│                     「缺了它」答不了「这个功能一天花多少钱」「这次为啥慢 8 秒」    │
├──────────────────────────────────────────────────────────────────────────────────┤
│ ⑦ 评测层            evaluation/ + scripts/  —— 标注集 · 指标 · 消融 · CI 门槛     │
│                     「缺了它」只能靠「我感觉更好了」改参数，改坏了也发现不了       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**一次上传写了几处**（离线、低频）：

```
上传 → data/uploads/{sha256}.{ext}（对象存储位）→ MySQL documents(status=UPLOADED)
     → 解析 loaders.py（保留页码）→ 切分 chunkers.py（子块=检索单位 / 父块=喂模型单位）
     → BGE-M3 向量化(1024维,归一化) → Chroma upsert
     → MySQL chunks（正文 + vector_id 锚点）→ documents(status=READY)
```

> 顺序是刻意设计的：**先写向量库、再写 MySQL chunks、最后置 READY**。
> 中途失败最坏是「向量库里有孤儿 chunk」（可清理），而不是
> 「列表显示成功、实际搜不到」——后者是最难排查的一类 bug。

**一次提问读了几处**（在线、高频）：

```
鉴权 → 落 user message + 起 trace → 取记忆（窗口+字符预算）
     → LangGraph：analyze → retrieve(向量‖BM25 → RRF → 重排 → 闸门 → 上下文)
                  → grade →(rewrite→retrieve)* → generate → reflect →(generate)*
     → SSE 逐帧推给前端：meta/trace/sources/tool/token/reflect/done/error/end
     → 落 assistant message + citations + trace_spans + token/成本
```

---

## 四、10 分钟跑起来

### 前置

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| Python | **3.11+** | 用了 `X \| None`、`tomllib`、`Annotated` |
| MySQL | **8.0+** | 业务库。只想看链路可以跳过，改用 SQLite（见「零 Key 也能跑」） |
| Node.js | **18+** | 前端。用 **pnpm**，不要用 npm |

### 1. 建库

```sql
CREATE DATABASE knowflow DEFAULT CHARSET utf8mb4;
```

### 2. 建虚拟环境 + 装依赖（清华镜像，快很多）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 只想先跑通链路、**不想下 2GB 的 BGE-M3**？再装一份开发依赖后设
> `EMBEDDING_PROVIDER=hash` 即可，`sentence-transformers` / `torch` 都用不上：
> ```powershell
> .\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 3. 配置

```powershell
# Windows
Copy-Item backend\.env.example .env
# Linux / macOS
cp backend/.env.example .env
```

`backend/.env.example` **就是配置文档**——每一项都写了「默认值 / 什么时候要改 / 改了会怎样」。
配置项全表另见 [`docs/01-数据库与接口契约.md`](docs/01-数据库与接口契约.md) 第六节。
所有键的唯一权威定义在 `backend/src/knowflow/core/config.py`。

### 4. 建表

```powershell
.\tasks.ps1 migrate          # = cd backend; python -m alembic upgrade head
```

### 5. 灌演示数据

```powershell
.\tasks.ps1 seed             # = python backend/scripts/seed_demo.py
```

会建好 `admin` / `admin123`、一个演示知识库、6 份不同格式的文档，并导入 43 条评测集。
想先验证链路（不落开发库）也可以：

```powershell
.\tasks.ps1 smoke            # 端到端自检：跑在 knowflow_test 库 + data/_smoke，不碰开发数据
```

### 6. 起后端

```powershell
.\tasks.ps1 dev              # 带 --reload
# 等价于：
# uvicorn knowflow.main:app --host 127.0.0.1 --port 8000 --app-dir backend/src --reload
```

验证：`http://127.0.0.1:8000/health` 应返回 `status: ok`，以及**真实的运行模式**
（`offline` / `embedding_mode` / `vector_count` / `bm25_doc_count`）。

### 7. 起前端

```powershell
cd frontend
pnpm install
pnpm dev
```

### 8. 打开并注册

<http://127.0.0.1:5173> → 注册（用户名 3–32 位 `[A-Za-z0-9_]`，口令 ≥ 6 位）。

> **第一个注册的用户自动成为 `admin`**（方便本地演示与 CI）。

---

## 五、零 Key 也能跑（离线降级）

不配任何 API Key 也能把**完整链路**跑一遍——上传、检索、Agent 编排、SSE、
引用落库、trace、评测全部照常。两条降级路径：

| 缺什么 | 降级成什么 | 你会失去什么 |
| --- | --- | --- |
| `OPENAI_API_KEY` 留空 | **抽取式回答**（不调外部服务） | 自然语言组织能力；答案是从检索片段里抽的 |
| `EMBEDDING_PROVIDER=hash` | **确定性哈希向量** | 语义能力。同义改写查不到（`出差住房` vs `住宿标准` 没有共同字） |
| `VECTOR_BACKEND=memory` | 纯内存向量库 | 进程退出即丢（测试用） |
| `AGENT_CHECKPOINT_BACKEND=memory` | 状态只在进程内 | 服务重启会话记忆丢 |

**`/health` 会如实上报你处在哪种模式**，这是「不骗人」的硬规则：

```jsonc
{
  "status": "ok",
  "offline": true,                                  // 无可用 API Key
  "llm_mode": "offline",                            // 抽取式回答
  "embedding_mode": "hash",                         // 或 "hash(fallback:local_load_failed)"
  "vector_backend": "chroma",
  "vector_count": 128,
  "bm25_doc_count": 0,                              // ← 不是 0 才说明 BM25 索引建好了
  "db": { "ok": true, "dialect": "mysql", "version": "8.0.34" },
  "warnings": []
}
```

> **为什么 `bm25_doc_count` 值得单列**：BM25 索引在内存里，进程退出就没了，
> 启动时要从 `chunks` 表全量重建。漏了这一步**服务不会报错、不会掉线**，
> 只是精确关键词查询悄悄变差——典型的静默降级。把它报出来，降级就可见了。

`hash` 模式的实现（[`embeddings/hash_embedder.py`](backend/src/knowflow/embeddings/hash_embedder.py)）：
用 hashing trick / random projection 把文本映射成 1024 维向量，
`blake2b(token)` 取 8 字节 → 低若干位做桶、**最高位做符号**（`±1`）。
符号位的存在是为了降低哈希碰撞的破坏力：两个 token 撞进同一个桶时有 50% 概率互相抵消，
而不是叠加放大。

---

## 六、本地 BGE-M3 完全离线

默认 `EMBEDDING_PROVIDER=local` 会从 HuggingFace 拉 `BAAI/bge-m3`（约 2GB）。
国内可用 `HF_ENDPOINT=https://hf-mirror.com` 加速。

**想彻底离线**，把 `EMBEDDING_MODEL_PATH` 指向 HF 原生缓存里的 `snapshots/<sha>` 目录
（优先级最高，设了就完全不联网）：

```dotenv
# .env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_MODEL_PATH=D:\models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181
EMBEDDING_DEVICE=auto          # auto = 有 CUDA 用 GPU，否则 CPU
HF_ENDPOINT=https://hf-mirror.com
```

怎么拿到这个路径：

```powershell
# 用 huggingface-cli 下到自定义目录（走镜像）
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\huggingface-cli.exe download BAAI/bge-m3 --local-dir D:\models\bge-m3

# 或者直接用 HF 默认缓存的 snapshots 目录
#   Windows: $env:USERPROFILE\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\<sha>
#   Linux:   ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<sha>
```

⚠️ **`EMBEDDING_DIM` 与模型强绑定**。BGE-M3 是 1024 维；换别的模型维度一变，
**必须重建整个向量库**（`chunks.vector_id` 与 Chroma 里的向量会和新查询向量对不上）。
`hash` 兜底也特意用 1024 维，就是为了降级时不必重建表结构。

---

## 七、接口速览

完整契约（字段级）见 [`docs/01-数据库与接口契约.md`](docs/01-数据库与接口契约.md) 第五节。

| 分组 | 基址 | 关键接口 |
| --- | --- | --- |
| 认证 | `/api/v1/auth` | `POST /register`（**首个用户自动 admin**）· `POST /login` · `GET /me` |
| 知识库 | `/api/v1/kbs` | `GET/POST /kbs` · `GET/PATCH/DELETE /kbs/{id}` · `GET /kbs/{id}/stats` |
| 文档 | `/api/v1` | `POST /kbs/{id}/documents`（multipart）· `GET/DELETE /documents/{id}` · `GET /documents/{id}/chunks` |
| 检索调试 | `/api/v1/kbs/{id}/search` | 四模式检索 + `gate` + `debug` 计数。**不调大模型、不花钱**，排障主力 |
| 对话 | `/api/v1` | `POST /chat`（非流式）· `POST /chat/stream`（SSE） |
| 会话 | `/api/v1` | `GET/POST /conversations` · `GET /conversations/{id}/messages` · `POST /messages/{id}/feedback` |
| 评测 | `/api/v1/eval` | `/eval/datasets` · `/eval/runs` · `/eval/ablation` · `/eval/compare` |
| 可观测 | `/api/v1/obs` | `/obs/stats` · `/obs/traces` · `/obs/traces/{id}` · `/obs/quality` |
| 运维 | 根路径 | `GET /health` · `GET /ready` · `GET /metrics`（**无需鉴权**） |

**SSE 事件顺序有保证**（前端只需处理 8 种事件）：

```
meta → (trace|tool|reflect)* → sources → token* → done → end
                                          ↑
                            先于 token：用户立刻看到引用来源，而不是等答完
```

出错时发 `error` 帧再发 `end`。⚠️ 前端用 **`POST + fetch + ReadableStream`** 手工分帧，
**不能用 `EventSource`**——它不支持自定义 `Authorization` 头和 POST body。

---

## 八、目录结构

```
knowflow/
├── backend/
│   ├── src/knowflow/
│   │   ├── core/           配置（唯一来源 + 启动期交叉校验）、日志、安全、异常
│   │   ├── middleware/     request_id 注入、耗时头、访问日志
│   │   ├── api/            路由层：auth / kbs / documents / search / chat / conversations / eval / obs / ops
│   │   ├── schemas/        Pydantic 请求响应模型（契约的可执行副本）
│   │   ├── services/       业务编排：auth / kb / document / chat / chunks / obs_bridge
│   │   ├── agent/          LangGraph 状态图：graph / nodes / state / runner / tools / memory
│   │   ├── retrieval/      混合检索：engine / bm25 / fusion / rerank / autocut / text / types
│   │   ├── embeddings/     向量化：local_bge / api_embedder / hash_embedder / factory
│   │   ├── vectorstore/    向量库抽象：base(协议) / chroma_store / memory_store / factory
│   │   ├── llm/            对话模型：openai_compat / mock(离线) / prompts / parsing / tokenizer
│   │   ├── ingest/         文档入库：loaders(解析,保留页码) / chunkers(父子块) / pipeline
│   │   ├── db/             SQLAlchemy 2.0：models / session / base / types(BIGINT UNSIGNED, DATETIME(6))
│   │   ├── evaluation/     评测：metrics / stats(bootstrap) / harness(消融) / datasets
│   │   ├── observability/  trace / pricing / metrics
│   │   └── container.py    ★ 全项目唯一装配入口（启动即失败，而不是第一次请求才 500）
│   ├── alembic/            迁移（versions/ 下是历史）
│   ├── scripts/            见下方「scripts 一览」（自检 / 审计 / 评测 / 标定 / 演示数据）
│   ├── tests/              222 个 pytest 用例（ASGITransport 直打 ASGI，不起服务器、零网络、4.2 秒）
│   ├── pyproject.toml      ruff / mypy / pytest 配置（工具链唯一真相来源）
│   ├── MIGRATIONS.md       迁移规范（含 4 个真实踩过的坑）
│   ├── requirements.txt    ★ 依赖版本的唯一权威
│   └── .env.example        ★ 配置项的唯一权威文档
├── frontend/               Vue 3 + Vite + TS(strict) + Pinia + Element Plus + ECharts
├── data/
│   ├── samples/            ★ 6 份样例文档（评测与标定脚本依赖它们）
│   ├── eval_cases.jsonl    ★ 43 条标注用例（同义改写 / 专有名词 / 多跳）
│   ├── threshold_calibration.json  ★ 阈值标定结果（阈值 vs 召回率/误放行率曲线）
│   ├── eval_report.json    ★ 四模式消融报告（含逐条 recall，可复核显著性检验）
│   ├── chroma/             运行期：向量库（不进库）
│   └── uploads/            运行期：上传原文件（不进库）
├── docs/
│   ├── 00-架构与技术选型.md      ★ 设计源头：为什么这么选、面试会追问什么
│   ├── 01-数据库与接口契约.md    ★ 14 张表 DDL + 接口清单 + 配置项全表
│   ├── 03-实测证据.md           ★ **全部文档与简历里的数字都来自这里**（可复核）
│   └── 教程/                    从零动手重建的分阶段教程（17 篇）
├── 知识库学习手册.md          ★ 按章节读代码的教科书（含面试自测题）
├── 03-knowflow-项目实现详解.html  ★ 图解实现（单文件自包含，离线可看）
├── 简历-大模型应用开发实习生-KnowFlow.md
├── Makefile                Linux/macOS/CI 入口
├── tasks.ps1               ★ Windows 入口（等价目标）
├── docker-compose.yml      MySQL + backend + frontend（未实测，见「已知边界」）
└── .github/workflows/ci.yml
```

> 标 ★ 的是「读代码前先读它」的文件。

### `backend/scripts/` 一览

| 脚本 | 用途 | 成本 |
| --- | --- | --- |
| `seed_demo.py` | 一键准备演示环境：admin + 演示 KB + 6 份文档 + 43 条评测集 | 需加载模型 |
| `smoke_pipeline.py` | 41 项链路断言（跑在隔离的库与数据目录上） | 需加载模型 |
| `smoke_http.py` | 70 项 HTTP 断言（真起 uvicorn + 解析 SSE 分帧） | 需加载模型 |
| `audit_auth.py` | 鉴权审计，读 FastAPI 真实依赖图 | 需加载模型 |
| `calibrate_threshold.py` | 闸门阈值标定 | **零成本**（不连库、不调模型，只做本地向量计算） |
| `run_eval.py` | 四模式消融 + 配对 bootstrap 显著性检验 | 需加载模型；`--llm-judge` 才花钱 |
| `check_env.py` | 依赖自检（25 个关键包） | 秒级 |
| `check_models.py` | ORM → 真实 MySQL 建表校验（14 张表 + utf8mb4 emoji） | 秒级 |
| `check_embedding.py` | BGE-M3 加载 + 语义有效性（同义 vs 无关相似度差） | 需加载模型 |
| `check_imports.py` | 分层导入自检（51 个模块） | 秒级 |
| `check_yaml.py` | compose 与 CI 配置的 YAML 语法 | 秒级 |
| `check_api_contract.py` | 前后端路径对齐（前端调的每个路径后端都得有） | 秒级 |
| `check_docs.py` | 文档质量门：教程结构 / HTML 自包含与锚点 / **数字可追溯** / 路径存在 | 秒级 |
| `audit_auth.py` | 鉴权审计（读 FastAPI 真实依赖图） | 秒级 |

---

## 九、常用命令

两个入口目标名**完全一致**：Windows 用 `.\tasks.ps1 <task>`，Linux/macOS/CI 用 `make <task>`。

| 任务 | 作用 | Windows | Linux/macOS |
| --- | --- | --- | --- |
| 安装运行期依赖 | 清华镜像 | `.\tasks.ps1 install` | `make install` |
| 安装开发依赖 | pytest/ruff/mypy | `.\tasks.ps1 install-dev` | `make install-dev` |
| 起后端 | `127.0.0.1:8000` | `.\tasks.ps1 run` | `make run` |
| 起后端（热重载） | 开发用 | `.\tasks.ps1 dev` | `make dev` |
| 建表/升级 | `alembic upgrade head` | `.\tasks.ps1 migrate` | `make migrate` |
| 生成迁移 | `.\tasks.ps1 revision "加 xxx"` | `.\tasks.ps1 revision "…"` | `make revision M="…"` |
| 回退迁移 | `alembic downgrade -1` | `.\tasks.ps1 downgrade` | `make downgrade` |
| 灌演示数据 | admin + 6 文档 + 43 用例 | `.\tasks.ps1 seed` | `make seed` |
| 跑测试 | 只跑 pytest | `.\tasks.ps1 test` | `make test` |
| 全量自检 | pytest+冒烟+ruff+mypy | `.\tasks.ps1 test-all` | `make test-all` |
| 覆盖率 | pytest-cov | `.\tasks.ps1 cov` | `make cov` |
| ruff | check + format --check | `.\tasks.ps1 lint` | `make lint` |
| 自动格式化 | 会改文件 | `.\tasks.ps1 format` | `make format` |
| 类型检查 | mypy strict | `.\tasks.ps1 typecheck` | `make typecheck` |
| **一键检查** | **ruff + mypy + pytest** | **`.\tasks.ps1 check`** | **`make check`** |
| 链路自检 | 41 项断言，不碰开发数据 | `.\tasks.ps1 smoke` | `make smoke` |
| **HTTP 冒烟** | **70 项断言：真起 uvicorn 打 39 个端点 + 校验 SSE 帧序** | **`.\tasks.ps1 smoke-http`** | **`make smoke-http`** |
| **鉴权审计** | **确认没有漏加鉴权的端点（曾漏过 21 个）** | **`.\tasks.ps1 audit`** | **`make audit`** |
| YAML 校验 | compose 与 CI 配置语法 | `.\tasks.ps1 yaml` | `make yaml` |
| 评测/消融 | 四模式对比 + 配对 bootstrap | `.\tasks.ps1 eval` | `make eval` |
| 阈值标定 | 零成本 | `.\tasks.ps1 calibrate` | `make calibrate` |
| 前端 | 安装/开发/构建 | `frontend-install` / `frontend-dev` / `frontend-build` | 同 |
| 清理缓存 | 不动 data/ 与 .venv | `.\tasks.ps1 clean` | `make clean` |
| 帮助 | 列出全部任务 | `.\tasks.ps1 help` | `make help` |

> **`check` 是本地与 CI 的同一个口径**：`ruff check` + `ruff format --check` + `mypy strict` + `pytest`。
> 提交前跑一次，CI 就不会红。
>
> **四层验证的分工**（为什么要四个，不是一个）：
>
> | 层 | 命令 | 覆盖什么 | 耗时 |
> | --- | --- | --- | --- |
> | 单元/集成 | `tasks.ps1 test` | 222 个用例：纯函数、模型、服务、路由 | 4.2 s |
> | 链路 | `tasks.ps1 smoke` | 41 项：入库 → 一致性 → 四种检索 → 拒答 → 流式 → 记忆 → 可观测 → 删除 | ~35 s（要加载模型） |
> | HTTP | `tasks.ps1 smoke-http` | 70 项：**真起 uvicorn**、真打 39 个端点、手写解析 SSE 分帧 | ~25 s |
> | 安全 | `tasks.ps1 audit` | 34 个受保护端点 / 2 个刻意公开 / 0 漏洞 | ~15 s |
>
> **为什么不能只有 pytest**：`pytest` 走 `ASGITransport`，不经过真实服务器、真实端口、
> 真实中间件栈，也**不会**触发 Starlette 对同步生成器的分次线程调度。
> 本项目两个最难查的 bug（**21 个端点漏鉴权**、**流式路径 contextvars 跨线程失效**）
> 都是 **pytest 全绿**的情况下只被 HTTP 冒烟抓到的。

---

## 十、技术栈

版本均为 `backend/requirements.txt` 中实际钉住、且在本机跑通的版本。

| 层 | 技术 | 版本 | 为什么选它 |
| --- | --- | --- | --- |
| 语言 | Python | 3.13 | 类型注解 + `async` + `X \| None` 够用，生态最全 |
| Web 框架 | FastAPI | 0.142.2 | 原生 async；Pydantic v2 校验；自动 OpenAPI；SSE 一个 `StreamingResponse` 搞定 |
| ASGI 服务器 | Uvicorn | 0.54.0 | `--reload` 开发顺手；生产配 gunicorn/多 worker |
| 配置校验 | pydantic-settings | 2.15.0 | 配置和请求用同一套类型系统，**配置错在启动期就炸**，而不是第一次请求 500 |
| 关系库 | MySQL | 8.0.34 | 业务事实需要事务、外键、聚合查询 |
| ORM | SQLAlchemy | 2.0.51 | 2.0 风格 `Mapped[...]` + `mapped_column` 有真类型；`select()` 显式，不用魔法 |
| 迁移 | Alembic | 1.18.5 | 表结构变更可版本化、可回滚、可进 CI |
| 向量库 | Chroma | 1.5.9 | 单机文件型、零运维、`hnsw:space=cosine`；`PersistentClient` 重启不丢 |
| 向量化 | BAAI/bge-m3（本地） | 1024 维 | 中文检索 SOTA 档；**本地跑 = 零 API 成本、离线可复现**；多语言 |
| 向量化框架 | sentence-transformers | 6.1.0 | `modules.json` 里已声明 CLS + Normalize，加载即正确 |
| 稀疏检索 | **自研 BM25** | ~120 行 | 不引依赖；中文 1/2-gram 切分；白板可推导，能讲到底层 |
| 融合 | **RRF / 加权 min-max** | 自研 | 两种都实现，能讲清「为什么工业界选 RRF」 |
| 重排 | LLM rerank + 启发式 | 自研 | 有 Key 用 LLM，没 Key 用覆盖率+标题命中兜底，链路不中断 |
| 对话模型 | LangChain + langchain-openai | 1.4.3 / 1.6.7 | **OpenAI 兼容协议** → 换 DeepSeek/Qwen/GLM 只改 `OPENAI_BASE_URL` |
| 切分 | langchain-text-splitters | 1.1.3 | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |
| Agent 编排 | LangGraph | 1.2.12 | 显式状态图 + 条件边 + checkpointer；比 `create_agent` 更可控、更可测 |
| 流式 | SSE（sse-starlette） | 3.5.0 | 单向推送够用，比 WebSocket 简单 |
| 认证 | PyJWT + bcrypt | 2.15.1 / 5.0.0 | 无状态 JWT；bcrypt 自带 salt，不自己造轮子 |
| 日志 | structlog | 26.1.0 | `request_id` 全链路绑定；本地彩色 / 生产 JSON 一键切 |
| 文档解析 | pypdf | 6.19.0 | 逐页抽取并**保留页码**，引用能精确到页 |
| 测试 | pytest + httpx | 9.1.1 / 0.28.1 | `ASGITransport` 直打 ASGI，不起服务器、不占端口、零网络 |
| 代码质量 | ruff + mypy(strict) | 0.16.10 / 2.4.0 | 「代码质量」变成可验证的事实，而不是自述 |
| 前端 | Vue 3 + Vite + TS | 3.5 / 7 | 组合式 API + 类型；Vite 冷启动秒级 |
| UI 库 | Element Plus | 2.x | 中后台组件全 |
| 状态管理 | Pinia | 3.x | Vue 官方推荐，TS 推导好 |
| 图表 | ECharts | 6.x | 可观测/评测页的延迟分布、指标趋势 |

**三条选型原则**：

1. **协议优先于厂商**——模型能力走 OpenAI 兼容协议，任何一家都能换。
   代价：拿不到厂商独有参数。
2. **能自己写的核心算法就自己写**——BM25、融合、闸门、指标全部自研。
   引库的调试成本 > 收益，而且「我用过 rank_bm25」和「我实现过 BM25 并解释为什么 k1=1.5」是两个层次。
   代价：没有分布式、没有高级重排模型。
3. **降级路径与主路径同等对待**——见上文「零 Key 也能跑」。
   代价：要额外维护 `/health` 如实上报，不能骗人。

---

## 十一、实测环境

本文档与 `docs/` 下所有**性能/指标数字都来自本机实测**，不是引自论文或网上抄来的：

| 项 | 实测值 |
| --- | --- |
| 操作系统 | Windows 11 |
| Python | **3.13.13** |
| MySQL | **8.0.34** |
| Node.js | **24** |
| pnpm | **10.33** |

关于「实测」的诚实说明：

- **阈值标定数字是实测的**：`data/threshold_calibration.json` 里 43 条用例的正负样本
  相似度分布与 `threshold → recall/false_accept/F1` 曲线，全部由
  `scripts/calibrate_threshold.py` 在本机用真实 BGE-M3 跑出来
  （结论：向量阈值 F1 最优点 0.63，F1 0.8193）。
- **检索指标（recall@k / MRR / nDCG）与消融对比表**由 `scripts/run_eval.py` 在本机产出。
  换机器、换模型、换标注集数字都会变——**请以你自己那次运行的输出为准**，不要复制本文档的数字。
- **延迟数字（如 `4.2 s`）是量级示意**，随 CPU/GPU、文档规模、模型供应商波动；
  真实数字请到 `/obs/stats` 的 P50/P95 里看。
- **`docker compose` 未实测**（本机没装 Docker，见下节）。

---

## 十二、已知边界与 roadmap（诚实清单）

主动讲出来，比被追问出来强得多——它证明你知道生产化的下一步在哪。

| 边界 | 现状 | 怎么修 |
| --- | --- | --- |
| 检索路径有同步阻塞调用 | BGE-M3 与 Chroma 是同步 API，跑在线程池里（`run_in_threadpool`），已收口 | 换 GPU 推理服务 / 用 async 客户端 |
| 上传是同步的 | 大文件会占用请求 | 引入任务队列（Celery/arq）+ 进度轮询 |
| BM25 单机内存 | 十万级 chunk 内存占用约数百 MB；进程重启要从 MySQL 全量重建 | 换 Elasticsearch / 只保留热门 KB 索引 |
| 无多租户隔离 | 单库单租户 | 加 `tenant_id` + 行级过滤 |
| Chroma 单机 | 无副本、无水平扩展 | 换 Milvus / pgvector（`VectorStore` 协议已留好接口） |
| 无重排序模型 | 用的是 LLM 重排与启发式 | 接 `bge-reranker-v2-m3` |
| **Docker / compose 未实测** | 本机没装 Docker，`Dockerfile` 与 `docker-compose.yml` **只做了 YAML/TOML 语法校验与人工核对**，没有真的 `docker compose up` 过 | 在有 Docker 的机器上跑一次 `docker compose up --build` 并验证 `/health` |
| 无鉴权刷新令牌 | JWT 到期要重新登录 | 加 refresh token |
| 埋点写库失败静默忽略 | `services/obs_bridge.py` 里 `try/except/pass` 是刻意的（埋点绝不能带崩问答），但**连日志都没有** | 改成 `logger.debug` 记一笔，让「埋点悄悄坏了」可见 |

---

## 十三、深入阅读

| 想了解 | 去哪看 |
| --- | --- |
| 为什么这么选、面试会追问什么 | [`docs/00-架构与技术选型.md`](docs/00-架构与技术选型.md) |
| 14 张表 DDL、接口清单、配置项全表 | [`docs/01-数据库与接口契约.md`](docs/01-数据库与接口契约.md) |
| 迁移怎么写、两个真实踩过的坑 | [`backend/MIGRATIONS.md`](backend/MIGRATIONS.md) |
| 配置每一项是什么意思 | [`backend/.env.example`](backend/.env.example) |
| 前端页面、SSE 自测 | [`frontend/README.md`](frontend/README.md) |

---

## 十四、许可

MIT。仓库内所有样例文档（`data/samples/`）均为虚构的演示数据。
