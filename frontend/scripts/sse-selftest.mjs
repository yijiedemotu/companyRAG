/**
 * SSE 增量解析自测（不依赖浏览器，直接 `node scripts/sse-selftest.mjs`）。
 *
 * 为什么要有这个脚本：
 * `api/sse.ts` 必须自己解析 SSE（EventSource 不支持 POST + Authorization 头），
 * 而「分帧解析」是最容易写错、又最难在浏览器里复现的一环。
 * 所以把纯解析函数 `parseSSEChunk` 抽到 `src/api/sse-parser.ts`，用 Node 直接喂分片数据验证。
 *
 * **本脚本 import 的是真正在浏览器里跑的同一份源码**（Node 24 原生类型擦除），
 * 不做任何复制粘贴，所以这里过了，浏览器里就是同一套逻辑。
 */

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

import { parseSSEChunk } from '../src/api/sse-parser.ts'

const here = dirname(fileURLToPath(import.meta.url))

let passed = 0
let failed = 0

function check(name, fn) {
  try {
    fn()
    passed += 1
    console.log(`  ✅ ${name}`)
  } catch (error) {
    failed += 1
    console.log(`  ❌ ${name}`)
    console.log(`     ${error instanceof Error ? error.message : String(error)}`)
  }
}

function group(title) {
  console.log(`\n${title}`)
}

/* ================================================================== 用例 1 */

group('① 帧被切成任意碎片（含在 data: 中间切断）')

check('第一片是半帧 → events 为空，rest 保留残余', () => {
  const first = parseSSEChunk('event: meta\ndata: {"conversation_id":7,"mo')
  assert.equal(first.events.length, 0, '半帧不应该被解析出来')
  assert.equal(first.rest, 'event: meta\ndata: {"conversation_id":7,"mo')
})

check('补上后半片 → 解析出完整 meta 帧，且 JSON 正确', () => {
  const part1 = 'event: meta\ndata: {"conversation_id":7,"mo'
  const part2 = 'de":"agent","trace_id":"abc","model":"deepseek-chat","offline":false,"embedding_mode":"local"}\n\n'
  const result = parseSSEChunk(part1 + part2)
  assert.equal(result.events.length, 1)
  assert.equal(result.rest, '')
  assert.equal(result.events[0].event, 'meta')
  assert.deepEqual(result.events[0].data, {
    conversation_id: 7,
    mode: 'agent',
    trace_id: 'abc',
    model: 'deepseek-chat',
    offline: false,
    embedding_mode: 'local',
  })
})

check('逐字节喂入（最极端的分片）最终结果与一次性喂入一致', () => {
  const raw =
    'event: token\ndata: {"text":"一线城市住宿标准为每晚 600 元 [1]。"}\n\n' +
    'event: end\ndata: {}\n\n'

  let buffer = ''
  const collected = []
  for (const char of raw) {
    buffer += char
    const { events, rest } = parseSSEChunk(buffer)
    collected.push(...events)
    buffer = rest
  }

  assert.equal(buffer, '', '全部字符喂完后不应有残余')
  assert.equal(collected.length, 2)
  assert.equal(collected[0].event, 'token')
  assert.deepEqual(collected[0].data, { text: '一线城市住宿标准为每晚 600 元 [1]。' })
  assert.equal(collected[1].event, 'end')
  assert.deepEqual(collected[1].data, {})
})

/* ================================================================== 用例 2 */

group('② 多行 data 必须用 \\n 连接（SSE 规范）')

check('跨 3 行的 data 只在行间断行，不累积 buffer 里的换行', () => {
  const raw = 'event: trace\ndata: {"node":"analyze",\ndata: "status":"ok",\ndata: "duration_ms":12}\n\n'
  const { events, rest } = parseSSEChunk(raw)
  assert.equal(rest, '')
  assert.equal(events.length, 1)
  assert.deepEqual(events[0].data, { node: 'analyze', status: 'ok', duration_ms: 12 })
})

check('多行 data 在碎片中间被切断，仍能正确 join', () => {
  const part1 = 'event: trace\ndata: {"node":"retrieve",\ndata: "st'
  const part2 = 'atus":"ok",\ndata: "duration_ms":88}\n\n'
  const a = parseSSEChunk(part1)
  assert.equal(a.events.length, 0)
  const b = parseSSEChunk(a.rest + part2)
  assert.equal(b.events.length, 1)
  assert.deepEqual(b.events[0].data, { node: 'retrieve', status: 'ok', duration_ms: 88 })
})

/* ================================================================== 用例 3 */

group('③ 一批数据含多帧 / CRLF / 注释行 / 未知字段')

check('一批数据含 3 帧，全部按顺序解析', () => {
  const raw =
    'event: sources\ndata: {"sources":[{"rank":1,"chunk_id":12,"doc_name":"员工报销制度.md"}]}\n\n' +
    'event: reflect\ndata: {"round":1,"passed":false,"unsupported":["这句话没有引用"],"action":"retry"}\n\n' +
    'event: done\ndata: {"conversation_id":7,"cost_usd":0.000231}\n\n'
  const { events, rest } = parseSSEChunk(raw)
  assert.equal(rest, '')
  assert.deepEqual(
    events.map((item) => item.event),
    ['sources', 'reflect', 'done'],
  )
  assert.equal(events[0].data.sources[0].doc_name, '员工报销制度.md')
  assert.equal(events[1].data.unsupported[0], '这句话没有引用')
  assert.equal(events[2].data.cost_usd, 0.000231)
})

check('CRLF 换行（sse-starlette 默认就是 \\r\\n）与 \\n 等价', () => {
  const raw = 'event: token\r\ndata: {"text":"你好"}\r\n\r\nevent: end\r\ndata: {}\r\n\r\n'
  const { events, rest } = parseSSEChunk(raw)
  assert.equal(rest, '')
  assert.deepEqual(
    events.map((item) => item.event),
    ['token', 'end'],
  )
  assert.deepEqual(events[0].data, { text: '你好' })
})

check('注释行（心跳）与 id/retry 字段被忽略', () => {
  const raw = ': keep-alive\nid: 42\nretry: 3000\nevent: token\ndata: {"text":"ok"}\n\n'
  const { events } = parseSSEChunk(raw)
  assert.equal(events.length, 1)
  assert.equal(events[0].event, 'token')
  assert.deepEqual(events[0].data, { text: 'ok' })
})

check('data 非合法 JSON 时退回原始字符串，不抛异常、不断流', () => {
  const raw = 'event: token\ndata: not-a-json\n\n'
  const { events } = parseSSEChunk(raw)
  assert.equal(events.length, 1)
  assert.equal(events[0].data, 'not-a-json')
})

check('data 冒号后的第一个空格按规范去掉，第二个空格保留', () => {
  // `data:  {"text":" x"}` → 去掉紧跟冒号的那一个空格 → `{"text":" x"}`
  // 注意：data 非 JSON 时 `parseDataField` 会先 trim 再退回原始字符串，
  // 所以这里必须用 JSON 载荷才能验证「第二个空格保留」。
  const { events } = parseSSEChunk('event: token\ndata:  {"text":" x"}\n\n')
  assert.equal(events.length, 1)
  assert.deepEqual(events[0].data, { text: ' x' })
})

/* ================================================================== 用例 4 */

group('④ 无 event 名的帧被丢弃；未闭合的尾部不丢数据')

check('只有 data 没有 event 的帧被丢弃（不污染业务逻辑）', () => {
  const { events } = parseSSEChunk('data: {"text":"孤儿"}\n\n')
  assert.equal(events.length, 0)
})

check('末尾未以空行结尾时，残余留在 rest 而不是被丢掉', () => {
  const { events, rest } = parseSSEChunk('event: token\ndata: {"text":"半句"') // 没有 \n\n
  assert.equal(events.length, 0)
  assert.equal(rest, 'event: token\ndata: {"text":"半句"')
})

/* ================================================================== 用例 5 */

group('⑤ 与浏览器同一份源码 + TextDecoder 增量解码')

check('自测脚本 import 的就是 src/api/sse-parser.ts 里的真实实现', () => {
  const source = readFileSync(resolve(here, '../src/api/sse-parser.ts'), 'utf8')
  const match = source.match(/export function parseSSEChunk\(buffer: string\): ParseSSEChunkResult \{[\s\S]*?\n\}/)
  assert.ok(match, '在源码里找不到 parseSSEChunk 函数体')
  assert.equal(typeof parseSSEChunk, 'function')
  // 纯函数：不可依赖网络/DOM
  assert.ok(!/fetch\(|document\.|window\./.test(match[0]), 'parseSSEChunk 不应触碰 DOM 或网络')
})

check('TextDecoder {stream:true}：UTF-8 汉字被在字节中间切断也不会乱码', () => {
  const raw = 'event: token\ndata: {"text":"一线城市住宿标准"}\n\nevent: end\ndata: {}\n\n'
  const bytes = new TextEncoder().encode(raw)
  // 故意在字节流中间切开（切点很可能落在一个汉字的 3 字节中间）
  const cut = 30
  const decoder = new TextDecoder('utf-8')

  let buffer = ''
  const collected = []
  for (const slice of [bytes.slice(0, cut), bytes.slice(cut)]) {
    buffer += decoder.decode(slice, { stream: true })
    const { events, rest } = parseSSEChunk(buffer)
    collected.push(...events)
    buffer = rest
  }
  buffer += decoder.decode()

  assert.equal(collected.length, 2)
  assert.deepEqual(collected[0].data, { text: '一线城市住宿标准' })
})

/* ================================================================== 汇总 */

console.log(`\n${'─'.repeat(56)}`)
console.log(`结果：通过 ${passed} 项，失败 ${failed} 项`)
if (failed > 0) {
  console.log('❌ SSE 解析自测失败')
  process.exit(1)
}
console.log('✅ SSE 增量解析自测全部通过（真实实现：src/api/sse-parser.ts）')
