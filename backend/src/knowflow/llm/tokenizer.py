"""token 估算。

**为什么需要它**：token 就是钱。但精确 tokenize 需要引入 tiktoken（还要按模型
下载词表），而本项目要支持任意 OpenAI 兼容供应商，词表不一定对得上。
所以这里做**估算**，用途是「成本量级判断 + 上下文预算控制」，
不是用来对账。真实用量从模型响应里的 `usage` 字段拿（那才是准的）。

估算规则（经验值，注释里给出依据）：
- 中文/日文/韩文汉字：约 1 token / 字（主流 BPE 词表对 CJK 通常 1~1.5 字/token，
  取 1.0 是**偏保守**（高估）的，用于预算控制更安全）；
- 其他可见字符：约 1 token / 4 字符（GPT 系列英文经验值）；
- 换行、空格也计入字符。
"""

from __future__ import annotations

import re
from typing import Final

# CJK 统一表意文字 + 中日韩标点 + 全角字符范围
_CJK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)

# 英文约 4 字符 / token
_CHARS_PER_TOKEN_LATIN: Final[float] = 4.0


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（空文本返回 0）。

    >>> estimate_tokens("")
    0
    >>> estimate_tokens("一线城市住宿标准")   # 8 个汉字
    8
    >>> estimate_tokens("hello world")      # 11 个字符 -> ceil(11/4) = 3
    3
    """
    if not text:
        return 0

    cjk_count = len(_CJK_PATTERN.findall(text))
    other_count = len(text) - cjk_count

    # ceil，保证非空文本至少算 1 个 token（否则短文本会算出 0，成本统计会漏）
    latin_tokens = -(-other_count // int(_CHARS_PER_TOKEN_LATIN))
    return int(cjk_count + latin_tokens)


def estimate_tokens_many(texts: list[str]) -> list[int]:
    return [estimate_tokens(t) for t in texts]


def fits_budget(texts: list[str], budget: int) -> bool:
    """判断一批文本的总 token 是否在预算内（组装上下文时用）。"""
    return sum(estimate_tokens(t) for t in texts) <= budget


__all__ = ["estimate_tokens", "estimate_tokens_many", "fits_budget"]
