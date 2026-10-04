# KnowFlow 开放接口说明

本文档面向接入方开发者，描述 KnowFlow 知识库服务的对外 HTTP 接口。

## 基础信息

- 生产环境基址：`https://api.knowflow.example.com/api/v1`
- 所有接口要求请求头 `Authorization: Bearer <access_token>`
- 请求与响应编码均为 `application/json; charset=utf-8`
- 上传接口使用 `multipart/form-data`

## 鉴权

调用 `POST /auth/login` 获取 token，请求体为：

```json
{"username": "alice", "password": "secret123"}
```

响应中的 `access_token` 有效期为 1440 分钟。token 过期后接口返回 `401 UNAUTHORIZED`，
需要重新登录。**不要在前端硬编码账号密码**。

## 错误码

所有错误响应结构一致：

```json
{"error": {"code": "KB_NOT_FOUND", "message": "知识库不存在", "detail": null, "request_id": "a1b2c3"}}
```

| HTTP | code | 说明 |
| --- | --- | --- |
| 400 | `UNSUPPORTED_FILE_TYPE` | 文件扩展名不在白名单 |
| 400 | `DOCUMENT_PARSE_ERROR` | 文档解析失败（加密 PDF、未知编码、扫描件） |
| 401 | `UNAUTHORIZED` | token 缺失、过期或签名错误 |
| 403 | `FORBIDDEN` | 非资源所有者且非管理员 |
| 404 | `KB_NOT_FOUND` | 知识库不存在或已删除 |
| 409 | `DOCUMENT_DUPLICATE` | 同一知识库内 sha256 重复 |
| 413 | `PAYLOAD_TOO_LARGE` | 请求体超过网关上限（10MB） |
| 422 | `REQUEST_VALIDATION_ERROR` | 参数校验失败 |
| 429 | `RATE_LIMITED` | 超过每分钟 60 次的调用限制 |
| 502 | `UPSTREAM_ERROR` | 上游大模型服务不可用 |
| 503 | `NOT_READY` | 服务尚未完成初始化 |

排查问题时请提供响应里的 `request_id`，服务端可据此定位全链路日志。

## 核心接口

### 创建知识库

`POST /kbs`

```json
{"name": "员工手册", "description": "HR 相关制度", "chunk_size": 600, "chunk_overlap": 120}
```

`chunk_overlap` 必须小于 `chunk_size`，否则返回 `400 VALIDATION_ERROR`。
同名知识库返回 `409 KB_NAME_CONFLICT`。

### 上传文档

`POST /kbs/{kb_id}/documents`，`multipart/form-data`，字段名固定为 `file`。

支持 `md`、`markdown`、`txt`、`pdf`、`csv`、`json` 六种格式，单个文件不超过 20MB。
上传是**同步**的：接口返回时文档已经可以被检索。大文件建议在前端显示进度提示。

响应中的 `status` 字段取值：`UPLOADED`、`PARSING`、`CHUNKING`、`EMBEDDING`、
`READY`、`FAILED`。`FAILED` 时 `error_code` 与 `error_message` 会给出原因。

### 检索（不调用大模型）

`POST /kbs/{kb_id}/search`

```json
{"query": "一线城市住宿标准", "mode": "hybrid_rerank", "top_k": 5, "include_debug": true}
```

`mode` 可选 `vector`、`bm25`、`hybrid`、`hybrid_rerank`。该接口**不产生模型费用**，
适合调参与效果验证。

响应中的 `gate` 字段给出相关性闸门判定结果。`gate.passed` 为 `false` 表示
检索结果都没有达到阈值，此时问答接口会直接返回拒答而不会调用大模型。

### 问答（非流式）

`POST /chat`

```json
{"question": "一线城市住宿标准是多少", "kb_id": 1, "mode": "agent"}
```

`mode` 可选 `rag`（固定链路，延迟低）与 `agent`（LangGraph 状态图，会做相关性判定、
查询改写与答案引用校验）。

### 问答（流式）

`POST /chat/stream`，请求体同上，响应为 `text/event-stream`。

事件序列为：`meta` → (`trace` | `tool` | `reflect`)* → `sources` → `token`* → `done` → `end`。
`sources` 事件**先于** `token` 事件，客户端应立即渲染引用卡片，避免用户空等。

注意：浏览器原生 `EventSource` **不支持** POST 与自定义请求头，
必须使用 `fetch` + `ReadableStream` 手动解析 SSE 分帧。

## 限流与配额

默认每分钟 60 次调用，按 `user_id` 维度计数，超限返回 `429 RATE_LIMITED`，
响应头 `Retry-After` 给出建议重试秒数。流式接口按一次请求计数，与生成时长无关。

## 版本与兼容性

接口版本通过 URL 路径携带（`/api/v1`）。新增字段属于向后兼容变更，会直接生效；
删除或重命名字段会先发布新版本路径，旧版本至少保留 6 个月。
