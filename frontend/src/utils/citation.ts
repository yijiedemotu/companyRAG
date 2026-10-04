/**
 * 引用角标解析。
 *
 * 助手回答里的 `[1]` `[2]` 要渲染成可点击的小角标 —— **不能用 v-html 拼字符串**（XSS 风险），
 * 所以这里把纯文本切分成「文本片段」与「角标」两类节点，由模板循环渲染。
 *
 * 匹配规则：`/\[(\d+)\]/g`，即方括号里是纯数字。`[abc]`、`[]` 保持原样当文本。
 */

export interface TextNode {
  type: 'text'
  /** 稳定 key，模板 v-for 用 */
  key: string
  text: string
}

export interface CitationNode {
  type: 'citation'
  key: string
  /** 正文里的编号，从 1 开始，对应 sources[].rank */
  rank: number
  /** 原始片段，例如 `[1]`，便于无障碍朗读时保留 */
  raw: string
}

export type ContentNode = TextNode | CitationNode

const CITATION_PATTERN = /\[(\d+)\]/g

/**
 * 把回答文本切分成节点数组。
 *
 * @param content 原始文本（可能包含 `[n]`）
 * @param validRanks 可选的合法 rank 集合；只有落在集合里的 `[n]` 才会变成角标，
 *                   避免正文里 `[2024]` 这类年份被误判成引用
 */
export function splitCitations(content: string, validRanks?: ReadonlySet<number>): ContentNode[] {
  const nodes: ContentNode[] = []
  if (!content) return nodes

  let cursor = 0
  let index = 0

  // 每次调用重置 lastIndex，避免全局正则的状态在多次调用间串味
  CITATION_PATTERN.lastIndex = 0
  let match: RegExpExecArray | null = CITATION_PATTERN.exec(content)

  while (match !== null) {
    const rank = Number(match[1])
    const isCitation = Number.isInteger(rank) && rank > 0 && (!validRanks || validRanks.has(rank))

    if (isCitation) {
      // 先把角标之前的普通文本推入
      if (match.index > cursor) {
        nodes.push({ type: 'text', key: `t${index}`, text: content.slice(cursor, match.index) })
        index += 1
      }
      nodes.push({ type: 'citation', key: `c${index}`, rank, raw: match[0] })
      index += 1
      cursor = match.index + match[0].length
    }

    match = CITATION_PATTERN.exec(content)
  }

  if (cursor < content.length) {
    nodes.push({ type: 'text', key: `t${index}`, text: content.slice(cursor) })
  }

  // 相邻纯文本合并（上面对非引用匹配不做切分，所以不会产生碎片，这里只做兜底）
  return nodes.filter((node) => node.type !== 'text' || node.text !== '')
}

/** 从 sources 里提取全部合法 rank，用于过滤 `[2024]` 这类误判 */
export function collectRanks(sources: ReadonlyArray<{ rank: number }>): Set<number> {
  return new Set(sources.map((source) => source.rank))
}

/** 是否富文本里有可点击引用 */
export function hasCitations(content: string): boolean {
  CITATION_PATTERN.lastIndex = 0
  return CITATION_PATTERN.test(content)
}
