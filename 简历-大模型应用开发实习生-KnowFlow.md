# 【姓名】

**【求职意向】** 大模型应用开发实习生（RAG / Agent 应用方向）
**【联系方式】** 【138-0000-0000】 ｜ 【your@email.com】 ｜ GitHub：【github.com/yourname】
**【实习信息】** 【每周可实习 5 天，可持续 6 个月以上】 ｜ 期望城市：【北京 / 上海 / 杭州】

> 📌 **这份简历怎么用**：所有 `【】` 里的内容需要你自己填。
> 文中的**每一个数字都来自本仓库的真实运行结果**（可复现命令都写在下面），
> 不是估算 —— 面试官追问时你能当场跑给他看，这是这份简历最大的价值。
> 复现命令见文末「数据可复现性」一节。

---

## 教育背景

**【学校名称】** ｜ 【专业名称】 ｜ 【本科 / 硕士】 ｜ 【2022.09 – 2026.06】

- GPA：【3.7 / 4.0（专业前 10%）】；核心课程：数据结构与算法、操作系统、数据库系统、机器学习、自然语言处理
- 【CET-6 xxx 分；如有竞赛 / 奖学金请补一行】

---

## 专业技能

- **Python 后端工程**：FastAPI 0.142 + Pydantic v2 严格校验；SQLAlchemy 2.0（`Mapped[...]` 类型化 ORM）+ Alembic 迁移（14 张表，autogenerate **零 drift** 已验证）；`pytest` 离线测试；`ruff` + `mypy --strict` **0 error**。
- **RAG 与检索**：自研 **BM25**（倒排索引 + 中文 1-gram/2-gram 切词 + IDF 恒正修正）；**RRF / 加权 min-max** 两种融合；父子块切分；**分数断崖自适应截断（autocut）**；**双阈值相关性闸门**；**阈值标定**（用标注集扫参取 F1 最优点，而不是拍脑袋）。
- **Agent 编排**：**LangGraph 显式状态图**（6 节点 / 4 条条件边）；**CRAG** 式检索相关性判定与查询改写；**Self-RAG** 式答案引用校验与有界重试；模型驱动工具调用循环（含"不支持工具时优雅降级"）；checkpointer 状态回溯。
- **向量与存储**：MySQL 8 关系建模（软删除 / 冗余计数 / `DECIMAL` 金额 / utf8mb4）；Chroma 持久化向量库（`hnsw:space=cosine` + 距离→相似度换算）；**本地 BGE-M3**（1024 维、CPU 约 45 ms/条、可完全离线）；`Protocol` 抽象 + **自动降级**（模型加载失败 → 哈希向量，且降级状态在 `/health` 可见）。
- **可观测与评测**：基于 `contextvars` 的**零侵入** span 埋点；trace/span 瀑布图；token 与成本归因（P50/P95/P99）；检索层指标（Recall@K / MRR / nDCG）与生成层判官；**配对 bootstrap 置信区间与显著性检验**。
- **前端**：Vue 3 + TypeScript（**strict，全仓库零 `any`**）+ Vite + Element Plus + Pinia + ECharts；**手写 SSE 增量解析**（`fetch` + `ReadableStream`，支持跨分片 / 多行 `data:` / 取消）。

---

## 项目经历

### KnowFlow ｜ 企业知识库智能问答服务 ｜ 个人项目 ｜ 【2025.xx – 至今】

**GitHub**：【github.com/yourname/knowflow】
**规模**：14 张表 · 后端 **88 个源文件 / 15,452 行** · 前端 **47 个文件 / 9,856 行** · 脚本 16 个 / 3,318 行 · **43 条标注评测集**

> **一句话**：把公司制度、手册、接口文档传进去，用自然语言提问，得到**带原文出处**的答案；
> 核心不在"调用大模型"，而在**检索质量**与**宁可拒答也不编**——并且每个设计都能拿出实测数字。

**技术栈**：Python 3.13 · FastAPI · LangChain / LangGraph · MySQL 8 + SQLAlchemy 2.0 + Alembic · Chroma · BAAI/bge-m3（本地） · 自研 BM25 · Vue 3 + TypeScript + Element Plus · pytest / ruff / mypy

#### 核心工作

- **搭出可度量的混合检索链路，并用显著性检验决定"哪些结论能写进简历"**：自研 **BM25**（倒排索引 + 中文 1-gram/2-gram 切词 + IDF 恒正修正）与 **BGE-M3 稠密向量**双路召回，用 **RRF 倒数排名融合**（对分数尺度漂移免疫），再叠加重排与自适应截断。在 **43 条标注集**（含同义改写 / 专有名词 / 多跳三类难点）上跑**四种模式的消融**，并用**配对 bootstrap（n_boot=2000, seed=42）**检验：混合检索把 **recall@5 从 0.721 提到 0.849（Δ=+0.128，95% CI [+0.035, +0.244]，p=0.010）**——**这条过了显著性检验，可以写**。同时诚实剔除一条：纯向量比混合高 1.2 个百分点但 **p=0.820、CI 跨 0**，属于噪声（43 条用例翻转 1 条就是 2.3 个点），**不写**。

- **标定防幻觉闸门，并推翻了自己拍脑袋设的默认值**：闸门是"向量相似度 ≥ 阈值 **或** 查询词覆盖率 ≥ 阈值"，两条都不过就**不调用大模型、直接拒答**。写了标定脚本在 43 条标注用例上用**真实 BGE-M3** 实测，发现 **BGE-M3 上"无关文本"的余弦相似度基线高达 0.57（不是 0）**——我最初按直觉设的 `0.35` 让闸门**完全失效（误放行率 100%）**，防幻觉机制形同虚设；而这个 bug 在功能测试里**完全看不出来**（接口 200、答案有引用、流程全通）。按扫描出的 F1 最优点重定为 `0.60 / 0.40`，重跑后无关问题被正确拦下（`reason=below_all_thresholds`）。

- **用测试钉住"防幻觉"这条设计承诺，而不是只写在文档里**：无召回时链路**短路拒答**，不花 token、不给模型编造的机会。这一点用 trace 断言证明：拒答请求的 span 序列是 `['analyze', 'retrieve']`——**没有 `generate.llm` span**，即生成模型确实一次都没被调用。同时把"拒答"落成 `messages.refusal` 独立布尔列（而不是靠字符串匹配），拒答率才能被聚合与回归。

- **把 Agent 的控制流画成图，而不是交给模型**：用 LangGraph 搭 `analyze → retrieve → grade →(rewrite → retrieve)* → generate → reflect →(generate)*` 状态图（6 节点 / 4 条条件边）。**模型只在节点内部做判断**，所以每一步都可观测、可兜底、可单测：4 个条件边判据全部是纯函数并逐条断言；改写与反思都有**硬上限**（各 2 次 / 1 次），从结构上消除"死循环烧钱"。离线用 Mock 模型扮演 6 种角色（规划 / 评分 / 改写 / 反思 / 判官 / 问答），因此**每条分支都能在零成本 CI 里被真实触发**，而不是只测主路径。

- **实现父子块切分与三类切分陷阱的防御**：子块（约 600 字）用于检索、父块（约 1800 字）用于喂模型，解决"切小了检索准但上下文不全、切大了上下文全但检索不准"的两难；markdown 标题栈生成 `section_path`（如 `员工报销制度 > 差旅报销标准`），让引用能精确到小节。切分器显式防御 `separators` 中混入空字符串（langchain 1.1.3 实测会 `break` 掉递归下降、从句中砍断），并保证**每个子块的文本都完整落在其父块内**（用测试断言，这是真实修过的 bug）。

- **零侵入可观测，让"为什么慢 / 一天花多少钱"变成可查数据**：用装饰器 + 显式持有者做 span 埋点，**业务代码零改动**；每个 span 记 `start_offset_ms`（相对 trace 起点，前端画瀑布图只需两个数）与截断后的 input/output；token 与成本按模型价格表估算（`Decimal` 定点，避免浮点误差冻进金额），落 `traces` / `trace_spans` 两张表。可观测落库失败**只记日志不抛异常**——埋点坏了不该让一次成功的问答变成 500。

- **靠"不带 token 打一次"发现并修掉 21 个未鉴权端点**：写 HTTP 冒烟时发现 `GET /kbs` 不带 token 竟然返回 **200 + 正常业务数据**。写脚本审计后发现 **39 个端点里有 21 个没鉴权**（`GET /kbs`、`GET /documents/{id}`、`GET /obs/traces` 等只读但含业务数据的接口）——它们的行为是"返回 200 + 正常数据"，**任何功能测试都发现不了**。修法不是逐个补签名，而是改成**路由级"默认需要、例外公开"**（`include_router(dependencies=[Depends(get_current_user)])`），让**将来新增的端点自动受保护**；并留下一个读 FastAPI 真实依赖图的审计脚本当 CI 门禁（34 个受保护 / 2 个刻意公开 / **0 个漏洞**）。

- **修复流式路径上的 contextvars 跨线程失效**：`/chat/stream` 每次都在最后发 `error` 帧、`done` 永远不出现，服务端报 `ValueError: Token ... was created in a different Context`。根因是 Starlette 用 `iterate_in_threadpool` 迭代**同步生成器**时**每取一个元素就是一次 `run_in_threadpool(next, ...)`**，`ContextVar.set()` 的 token 无法在另一个 context 里 `reset()`。改成把 recorder **挂在依赖对象上显式传递**（`RecorderBox`），与线程无关。**这个 bug 只在流式路径出现，而 222 个 pytest 用例（走 ASGITransport）全绿也照样漏掉它 —— 只有真起 uvicorn 打一次流式接口才会暴露**，这也是我坚持写 HTTP 冒烟的原因。

- **工程化：把"代码质量"变成可验证的事实**：`mypy --strict` 在 **97 个源文件**上 **0 error**；`ruff check` + `ruff format --check` 干净；Alembic 迁移与模型 **autogenerate 零 drift**（跑一次 autogenerate，产出文件里 `op.` 调用数为 0）。**三层验证全绿**：**222 个离线 pytest 用例（4.2 秒）** + **41 项链路冒烟断言** + **70 项 HTTP 冒烟断言**（真起 uvicorn、真打 39 个端点、手写解析 SSE 分帧）。测试跑在**独立数据库与独立数据目录**上（`knowflow_test` + `data/_smoke`），不污染开发数据；同一套断言在 `EMBEDDING_PROVIDER=hash` + `VECTOR_BACKEND=memory` 下也全绿，因此 **CI 零成本、零网络、不下载 2GB 模型**。

#### 实测数据（本机运行结果）

| 项目 | 实测值 | 说明 |
| --- | --- | --- |
| **消融：混合检索 vs 纯 BM25** | recall@5 **0.721 → 0.849**，Δ=+0.128，**p=0.010** | 配对 bootstrap，**显著**，可写 |
| 消融：纯向量 vs 混合 | Δ=+0.0116，**p=0.820**，CI 跨 0 | **不显著 = 噪声，不写** |
| 四种模式对比 | vector 0.860 / bm25 0.721 / hybrid 0.849 / hybrid_rerank 0.849 | 43 条标注用例，top_k=5 |
| 检索延迟构成 | BM25 **5ms** / 向量 **58ms** / 混合 56ms | CPU 推理，34 个切片的小库 |
| **闸门阈值标定** | 正样本均值 **0.676** / 负样本均值 **0.570** | 基线极高 → 原 0.35 阈值**误放行 100%** |
| 入库 6 份文档（md/csv/json） | 34 个切片，三方一致 `34/34/34` ✓ | MySQL / Chroma / BM25 完全一致 |
| BGE-M3 向量化 | 1024 维，约 **45 ms/条**（CPU） | 加载约 1.7 ~ 15 s |
| 语义区分度 | 同义 **0.668** vs 无关 **0.449** | 差值 +0.219，证明向量化有效 |
| Chroma vs 内存向量库 | top-1 一致，score Δ = **1.79e-07** | 两个后端行为等价 |
| 三层验证 | **222** pytest（4.2s）+ **41** 链路 + **70** HTTP | 全绿；`mypy --strict` 97 文件 0 error |
| 鉴权审计 | 34 受保护 / 2 刻意公开 / **0 漏洞** | 修复前 39 个端点里 **21 个漏鉴权** |
| Agent vs 固定链（离线 Mock） | 691 ms vs 317 ms | ⚠ 仅 Mock 下的相对对比 |

> **诚实声明**：上表除"闸门阈值标定""消融实验""语义区分度"外，
> 其余均在**未配置大模型 API Key 的离线模式**下测得（用抽取式 Mock 保证链路可复现、CI 零成本）。
> 因此**没有测"答案质量"**（faithfulness / relevance），只测了
> **检索质量、一致性、链路正确性、参数标定与安全性**——
> 这几项恰好都是离线可测且与真实模型无关的。
> 另外：`hybrid` 与 `hybrid_rerank` 结果相同，是因为离线模式下重排走启发式兜底
>（`MockChatModel.offline=True`），**这一列测不出 LLM 重排的价值**，接真实 Key 后必须重跑。

#### 边界声明（主动交代）

- 检索路径上的本地向量化与向量库查询是**同步阻塞**的，目前收口在线程池里；高并发下线程是瓶颈，替换为 async 客户端或独立推理服务才能真并发。
- 文档上传是**同步**的（契约即如此），大文件会占住一个请求；生产化应引入任务队列 + 进度轮询。
- 向量库与关系库**共用同一个 `DATA_DIR`**：切换 `DATABASE_URL` 必须同时切换 `DATA_DIR`，否则向量索引会串库（这是已知的设计边界，写进了架构文档）。
- `Dockerfile` / `docker-compose.yml` 已编写，但**本机无 Docker，未实测启动**；只做了 YAML 语法校验。
- 重排序用的是 **LLM 重排 + 启发式兜底**，没有接真实的交叉编码器（`bge-reranker-v2-m3`）；路线图上已标注。

---

## 个人优势

习惯把"改进了什么"变成"**证据是什么**"：每个结论都带可复现的命令与实测数字，
并且主动交代边界（离线测的、没实测的、只在原理上成立的）。
最有代表性的一次是**用测量推翻了自己拍脑袋设的参数**——
发现余弦相似度的"无关基线"高达 0.57、原阈值让闸门完全失效，
这比"我调了个参数效果变好了"更接近真实工程。

能在无人指导下把一个 AI 服务从 0 做到**可运行、可度量、可回归**：
后端 88 文件 / 前端 47 文件 / 14 张表 / 41 项端到端断言，
全部在一台 Windows 笔记本上跑通，且零成本可复现。

---

## 数据可复现性（面试时可以当场演示）

```bash
# 0) 前置：MySQL 8 已启动，建库
mysql -uroot -p -e "CREATE DATABASE knowflow DEFAULT CHARSET utf8mb4;"

# 1) 装依赖（国内镜像）
python -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 建表 + 准备演示数据（管理员 / 6 份文档 / 43 条评测集）
copy backend\.env.example .env
cd backend && ..\.venv\Scripts\python.exe -m alembic upgrade head && cd ..
.venv\Scripts\python.exe backend\scripts\seed_demo.py

# 3) 阈值标定（产出上面那张"正/负样本相似度"的表）
.venv\Scripts\python.exe backend\scripts\calibrate_threshold.py --out data/threshold_calibration.json

# 4) 端到端链路自检（41 项断言，跑在隔离的库与目录上）
.venv\Scripts\python.exe backend\scripts\smoke_pipeline.py

# 5) 检索质量评测与消融（四种检索模式对比 + 配对 bootstrap 显著性）
.venv\Scripts\python.exe backend\scripts\run_eval.py --modes vector,bm25,hybrid,hybrid_rerank
```

---

## 【占位】其他经历

> 如有实习、竞赛、论文、开源贡献或学生工作，按「时间 ｜ 角色 ｜ 做了什么 + 量化结果」补在这里；
> 没有可整节删除。

## 【占位】荣誉奖项

> 【奖学金 / 竞赛奖项 / 专利论文，一行一条；没有可整节删除】

---

## 【占位】另一个可讲的项目

> 你名下还有一份「企业知识库智能问答系统」（FastAPI + pgvector/Milvus + React + 187 个测试用例）。
> **建议不要和 KnowFlow 同时放进简历**——两个 RAG 项目会互相稀释。
> 二选一，另一个留作面试时"我还做过一个类似的，但技术选型不同"的备谈素材
> （这反而能体现你能比较技术选型的取舍）。
