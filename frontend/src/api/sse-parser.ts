/**
 * SSE 纯解析层（**不依赖浏览器**，可在 Node 里直接跑自测）。
 *
 * 契约（docs/01 5.6 节）：帧格式 `event: <name>\ndata: <json>\n\n`。
 *
 * 为什么必须自己解析：
 * `POST /api/v1/chat/stream` 需要带 `Authorization` 头、且必须是 POST，
 * 浏览器原生 `EventSource` 两者都不支持，所以只能 fetch + ReadableStream 手撕。
 *
 * 关键规范点：
 * - `data:` 可以跨多行，多行之间用 `\n` join（SSE 规范），最后再 JSON.parse；
 * - 帧与帧之间用空行分隔，即 `\n\n`；
 * - `:` 开头是注释行，忽略；
 * - 增量解码：TextDecoder 必须带 `{stream:true}`，否则一个 UTF-8 汉字被切断会变乱码。
 */

/** 已知的 9 个事件名（注意 `end` 也是事件名之一） */
export const KNOWN_SSE_EVENTS = [
  'meta',
  'trace',
  'tool',
  'sources',
  'reflect',
  'token',
  'done',
  'error',
  'end',
] as const

/** 解析结果：`data` 保持 unknown，交由上层用类型守卫收窄，避免在这一层引入 any */
export interface ParsedSSEEvent {
  event: string
  data: unknown
}

export interface ParseSSEChunkResult {
  /** 本次能完整解析出的帧（按到达顺序） */
  events: ParsedSSEEvent[]
  /** 尚未凑齐一帧的残余，必须拼到下一批数据前面 */
  rest: string
}

/**
 * 解析一批增量的 SSE 文本。
 *
 * 纯函数：同样的输入永远得到同样的输出，不碰网络、不碰 DOM。
 * 这是本项目「SSE 必须自己解析」这条契约的可测试落点。
 *
 * @param buffer 上一次解析剩下的 `rest` + 本次新解码出来的文本
 */
export function parseSSEChunk(buffer: string): ParseSSEChunkResult {
  // 统一换行：服务端可能发 CRLF（sse-starlette 默认就是 \r\n）
  const normalized = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

  const events: ParsedSSEEvent[] = []
  const blocks = normalized.split('\n\n')

  // 最后一段永远可能是「半帧」：留在 rest 里等下一批数据补齐
  const rest = blocks.pop() ?? ''

  for (const block of blocks) {
    const parsed = parseEventBlock(block)
    if (parsed) events.push(parsed)
  }

  return { events, rest }
}

/**
 * 解析单个完整帧。
 * 一个「完整帧」= 若干行，用 `\n` 分隔。
 */
function parseEventBlock(block: string): ParsedSSEEvent | null {
  const lines = block.split('\n')
  let eventName = ''
  const dataLines: string[] = []

  for (const line of lines) {
    if (line === '') continue
    // 注释行（心跳常用 `: keep-alive`）
    if (line.startsWith(':')) continue

    const colonIndex = line.indexOf(':')
    const field = colonIndex === -1 ? line : line.slice(0, colonIndex)
    let value = colonIndex === -1 ? '' : line.slice(colonIndex + 1)
    // 规范：冒号后若紧跟一个空格，要去掉这一个空格
    if (value.startsWith(' ')) value = value.slice(1)

    if (field === 'event') {
      eventName = value.trim()
    } else if (field === 'data') {
      dataLines.push(value)
    }
    // id / retry 字段本项目不使用，忽略
  }

  // 无 event 名的帧（例如只有心跳 data）直接丢弃，避免污染业务逻辑
  if (eventName === '') return null

  // SSE 规范：多行 data 用 \n 连接
  const rawData = dataLines.join('\n')

  return {
    event: eventName,
    data: parseDataField(rawData),
  }
}

/**
 * data 字段解析：优先 JSON.parse，失败则退回原始字符串。
 * 后端约定 data 恒为 JSON，退回原始字符串只是一种有损兜底（不抛异常、不断流）。
 */
function parseDataField(raw: string): unknown {
  const text = raw.trim()
  if (text === '') return {}
  try {
    return JSON.parse(text) as unknown
  } catch {
    return text
  }
}
