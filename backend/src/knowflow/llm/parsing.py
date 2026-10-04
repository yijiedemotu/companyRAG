"""从模型输出里稳妥地抠出 JSON。

**为什么需要单独一个模块**：所有"让模型输出 JSON"的地方都会遇到同一批脏数据：

1. 包了 ```json 代码块（最常见）；
2. 前后带一句解释文字（"好的，结果如下：{...}"）；
3. 中文全角标点（`“key”：`）、尾随逗号；
4. 被截断（`max_tokens` 用完，JSON 不闭合）；
5. 输出成了 JSON 数组而不是对象。

对策是**逐级降级**：先用最严格的方式，失败再放宽，最后才放弃。
绝不能因为模型多说了半句话就让整个问答 500 —— 拿不到结构化结果时，
调用方要能走"降级分支"（例如 grade 全部放行），而不是把错误抛给用户。
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_PATTERN = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
# 全角标点 → 半角（只在 JSON 结构位置，不碰字符串内容里的标点）
_FULLWIDTH_MAP = str.maketrans({"“": '"', "”": '"', "：": ":", "，": ",", "｛": "{", "｝": "}"})


def _strip_fences(text: str) -> str:
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _balanced_slice(text: str, open_ch: str, close_ch: str) -> str | None:
    """截出第一个配平的 `{...}` / `[...]`，能正确处理字符串里的括号与转义。

    不能用正则做这件事：`{"a": "}"}` 这种值里含括号的情况正则会切错，
    而模型恰好经常在理由字段里写括号。
    """
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_loads(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 放宽 1：去掉尾随逗号  ,}  ,]
    relaxed = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(relaxed)
    except json.JSONDecodeError:
        pass
    # 放宽 2：全角标点
    try:
        return json.loads(relaxed.translate(_FULLWIDTH_MAP))
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> Any | None:
    """尽最大努力从文本里取出一个 JSON 值（对象或数组）。取不到返回 None。"""
    if not text or not text.strip():
        return None

    for candidate in (_strip_fences(text), text.strip()):
        parsed = _try_loads(candidate)
        if parsed is not None:
            return parsed
        # 对象优先于数组：本项目的所有结构化输出都是对象
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            sliced = _balanced_slice(candidate, open_ch, close_ch)
            if sliced:
                parsed = _try_loads(sliced)
                if parsed is not None:
                    return parsed
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """只要对象。如果模型返回了数组，返回 None 让调用方走降级分支。"""
    parsed = extract_json(text)
    return parsed if isinstance(parsed, dict) else None


def extract_score(text: str) -> float | None:
    """从判官输出里取 0~1 的分数。

    优先级：JSON 的 `score` 字段 > 文本里第一个 0~1 的小数。
    返回值**夹到 [0,1]**：模型偶尔会输出 1.2 或 -0.1，直接入库会让指标不可比。
    """
    obj = extract_json_object(text)
    if obj is not None and "score" in obj:
        try:
            value = float(obj["score"])
            return min(1.0, max(0.0, value))
        except (TypeError, ValueError):
            pass
    match = re.search(r"(?<![\d.])(\d(?:\.\d+)?)(?![\d])", text or "")
    if not match:
        return None
    try:
        return min(1.0, max(0.0, float(match.group(1))))
    except ValueError:
        return None


def extract_queries(text: str, *, fallback: str) -> list[str]:
    """取改写后的查询列表；取不到就用原始问题兜底（永远不返回空列表）。"""
    obj = extract_json_object(text)
    if obj is not None:
        raw = obj.get("queries") or obj.get("query")
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        if isinstance(raw, list):
            cleaned = [str(q).strip() for q in raw if str(q).strip()]
            if cleaned:
                return cleaned[:3]
    return [fallback]


__all__ = [
    "extract_json",
    "extract_json_object",
    "extract_queries",
    "extract_score",
]
