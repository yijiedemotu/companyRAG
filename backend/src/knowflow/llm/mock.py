"""离线模型：没有 API Key 时的全链路兜底。

**它存在的理由不是"凑合能用"，而是三条硬需求**：

1. **CI 零成本**：102 个测试、每次提交都跑，不能花钱、不能依赖外网；
2. **clone 下来 30 秒能看到完整效果**：面试官/评审者不会去申请 API Key；
3. **把"模型质量"与"工程质量"解耦**：链路出错时，能立刻排除"是不是模型抽风"。

**它怎么工作**：不训练、不推理，纯**抽取式**——
按查询词覆盖率在检索到的上下文里挑最相关的一段句子，附上引用编号返回。
所以它**只能验证链路连通性与引用格式，不能验证生成质量**，这一点必须诚实标注
（`OFFLINE_NOTICE` 会出现在每个答案里，`/health` 也会把 `offline=true` 报出来）。

**一个刻意的设计**：mock 会按 system prompt 扮演 6 种角色
（分析 / 评分 / 改写 / 反思 / 判官 / 问答），返回**格式合法的 JSON**。
这样离线模式下 LangGraph 的**每一条条件边都能被真实触发**，
测试才能覆盖"grade 判定不相关 → 走 rewrite 分支"这类路径。
用一个永远返回固定文本的 mock，图的分支就等于没测。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Sequence
from typing import Any

from knowflow.core.config import Settings, get_settings
from knowflow.llm.base import ChatMessage, ChatResult, ChatUsage, Chunk, messages_to_text
from knowflow.llm.prompts import OFFLINE_NOTICE, REFUSAL_ANSWER
from knowflow.retrieval.text import coverage

# 上下文块：[1] （出处：A > B）\n正文       或       [1] 正文
_BLOCK = re.compile(
    r"^\[(\d+)\]\s*(?:（出处：([^）]*)）)?[ \t]*\n?(.*?)(?=\n\[\d+\]\s|\Z)",
    re.DOTALL | re.MULTILINE,
)
_QUESTION = re.compile(r"(?:用户问题|问题)[：:]\s*(.+)", re.DOTALL)

# 句子切分：中文句末标点 + 换行
_SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]?")

# 抽取式回答的最低覆盖率门槛。低于它就拒答——保证离线模式也"不编"。
_MIN_BLOCK_COVERAGE = 0.25
_MIN_SENTENCE_SCORE = 1


def _detect_role(system_text: str) -> str:
    """按 system prompt 里的特征词识别这次调用要扮演什么角色。

    依赖 prompt 文本是有意的耦合：`llm/prompts.py` 与 `llm/mock.py` 必须一起改，
    而这个耦合会**在测试里立刻暴露**（改了 prompt 关键词，mock 的 JSON 就断言失败）。
    """
    if "问答路由器" in system_text:
        return "analyze"
    if "检索质量评审员" in system_text:
        return "grade"
    if "检索查询优化器" in system_text:
        return "rewrite"
    if "答案审核员" in system_text:
        return "reflect"
    if "评测判官" in system_text:
        return "judge"
    return "answer"


def _parse_contexts(user_text: str) -> list[tuple[int, str, str]]:
    """抠出 `(编号, 出处, 正文)`。出处可能不存在（grade 用的格式没有出处）。"""
    blocks: list[tuple[int, str, str]] = []
    for match in _BLOCK.finditer(user_text):
        rank = int(match.group(1))
        source = (match.group(2) or "").strip()
        body = (match.group(3) or "").strip()
        if body:
            blocks.append((rank, source, body))
    return blocks


def _parse_question(user_text: str) -> str:
    match = _QUESTION.search(user_text)
    if match:
        return match.group(1).strip()
    # 兜底：取最后一行非空文本
    lines = [line.strip() for line in user_text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _best_sentences(question: str, body: str, *, limit: int = 2) -> list[str]:
    """在正文里挑与问题最相关的句子（按查询词命中数排序，保持原文顺序）。"""
    scored: list[tuple[int, int, str]] = []
    for position, raw in enumerate(_SENTENCE.findall(body)):
        sentence = raw.strip()
        if len(sentence) < 6:
            continue
        score = coverage(question, sentence)
        if score >= _MIN_SENTENCE_SCORE / max(len(question), 1) or score > 0:
            scored.append((int(score * 1000), position, sentence))

    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked = sorted(scored[:limit], key=lambda item: item[1])
    return [sentence for _, _, sentence in picked]


class MockChatModel:
    """实现了 `ChatModel` 协议。`offline` 恒为 True。"""

    offline = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = "offline-mock"

    # ------------------------------------------------------------------ 内部
    def _handle(self, messages: Sequence[ChatMessage]) -> str:
        system_text = "\n".join(m.content for m in messages if m.role == "system")
        user_text = "\n".join(m.content for m in messages if m.role != "system")
        role = _detect_role(system_text)
        question = _parse_question(user_text)
        contexts = _parse_contexts(user_text)

        if role == "analyze":
            return json.dumps(
                {
                    "needs_retrieval": True,
                    "queries": [question] if question else [],
                    "question_type": "factual",
                    "reason": "离线模式：默认总是走检索，保证链路完整",
                },
                ensure_ascii=False,
            )
        if role == "rewrite":
            return json.dumps(
                {"queries": [question] if question else [], "reason": "离线模式：不做改写"},
                ensure_ascii=False,
            )
        if role == "grade":
            return self._grade(question, contexts, user_text)
        if role == "reflect":
            return self._reflect(user_text)
        if role == "judge":
            return self._judge(question, user_text, contexts)
        return self._answer(question, contexts)

    def _grade(self, question: str, contexts: list[tuple[int, str, str]], user_text: str) -> str:
        """按查询词覆盖率判相关性。阈值取得比真实闸门略宽，让 rewrite 分支也能被触发。"""
        if not contexts:
            # grade 的输入格式是 `[1] 正文`，与 answer 的格式略有差异，退化解析一次
            for match in re.finditer(r"^\[(\d+)\]\s*(.+)$", user_text, re.MULTILINE):
                contexts.append((int(match.group(1)), "", match.group(2).strip()))
        items: list[dict[str, Any]] = []
        for rank, _source, body in contexts:
            score = coverage(question, body)
            relevant = score >= 0.4
            items.append(
                {
                    "id": rank,
                    "relevant": relevant,
                    "reason": (
                        f"查询词覆盖率 {score:.2f}，高于阈值"
                        if relevant
                        else f"查询词覆盖率 {score:.2f}，低于阈值"
                    ),
                }
            )
        return json.dumps({"items": items}, ensure_ascii=False)

    def _reflect(self, user_text: str) -> str:
        """检查答案里的"事实性句子"是否都带引用编号。"""
        parts = user_text.split("待审核答案：")
        answer = parts[-1].strip() if len(parts) > 1 else user_text
        unsupported: list[dict[str, str]] = []
        for raw in _SENTENCE.findall(answer):
            sentence = raw.strip()
            # 只检查"有事实主张"的句子：够长、且含数字或中文实词
            if len(sentence) < 12:
                continue
            if re.search(r"\[\d+\]", sentence):
                continue
            unsupported.append({"sentence": sentence, "reason": "该句末尾没有引用编号"})
        return json.dumps(
            {
                "passed": not unsupported,
                "unsupported": unsupported,
                "overall": "离线模式：按引用编号是否存在做形式化检查",
            },
            ensure_ascii=False,
        )

    def _judge(self, question: str, user_text: str, contexts: list[tuple[int, str, str]]) -> str:
        answer = user_text.split("答案：")[-1].strip()
        if not answer or answer.startswith("知识库中没有"):
            return json.dumps({"score": 0.0, "reason": "答案为空或为拒答"}, ensure_ascii=False)
        # faithfulness：答案与上下文的覆盖率；answer_relevance：答案与问题的覆盖率
        if "参考资料" in user_text:
            context_text = user_text.split("参考资料：")[-1].split("答案：")[0]
            score = coverage(answer, context_text)
            reason = "离线模式：按答案与参考资料的关键词覆盖估算"
        else:
            score = coverage(question, answer)
            reason = "离线模式：按答案与问题的关键词覆盖估算"
        return json.dumps(
            {"score": round(min(1.0, score), 4), "reason": reason}, ensure_ascii=False
        )

    def _answer(self, question: str, contexts: list[tuple[int, str, str]]) -> str:
        """抽取式回答：挑覆盖率最高的上下文块，再挑块里最相关的句子。"""
        if not contexts:
            return REFUSAL_ANSWER

        ranked: list[tuple[float, int, str, str]] = []
        for rank, source, body in contexts:
            ranked.append((coverage(question, body), rank, source, body))
        ranked.sort(key=lambda item: -item[0])

        best_score, best_rank, best_source, best_body = ranked[0]
        if best_score < _MIN_BLOCK_COVERAGE:
            return REFUSAL_ANSWER

        sentences = _best_sentences(question, best_body)
        if not sentences:
            # 整块最相关但拆不出单句时，直接给块首 200 字，保证有内容可展示
            sentences = [best_body[:200].strip()]

        body_text = " ".join(sentences)
        citation = f"[{best_rank}]"
        source_hint = f"（出自 {best_source}）" if best_source else ""
        return f"{OFFLINE_NOTICE}\n\n{body_text} {citation}{source_hint}"

    # ------------------------------------------------------------------ 协议
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        text = self._handle(messages)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        prompt_tokens = len(messages_to_text(messages))
        return ChatResult(
            text=text,
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=len(text),
                model=self.model,
                llm_calls=1,
            ),
            finish_reason="stop",
            latency_ms=elapsed_ms,
            offline=True,
        )

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> Iterator[Chunk]:
        """按 8 个字符一块"假装"流式，并 yield 一个收尾块带上 usage。

        刻意加一点 `sleep`：不加的话前端会在同一帧里收到所有 token，
        看起来像非流式，"流式效果"就没法验证了。
        """
        result = self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        text = result.text
        step = 8
        for i in range(0, len(text), step):
            yield Chunk(text=text[i : i + step])
            time.sleep(0.012)
        yield Chunk(text="", usage=result.usage, finish_reason="stop")

    def supports_tools(self) -> bool:
        """离线模式不支持工具调用 → Agent 走"优雅降级"分支（见 `agent/graph.py`）。"""
        return False

    def health(self) -> dict[str, Any]:
        return {"offline": True, "model": self.model, "supports_tools": False}


__all__ = ["MockChatModel"]
