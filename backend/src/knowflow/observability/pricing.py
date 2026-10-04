"""成本核算：token -> 美元 -> 人民币。

**一句话定位**：把「用了多少 token」翻译成「花了多少钱」，并且用 `Decimal` 保证
分位聚合不漂。

**在链路中的位置**：
``LLM 调用返回 usage -> pricing.estimate_cost() -> messages.cost_usd / traces.cost_usd``
再往上被 ``observability.metrics.collect_stats()`` 聚合成"今天花了多少钱"。

**关键设计取舍**：

1. **必须用 `Decimal`，且只能用 `Decimal(str(x))` 构造**。`Decimal(0.00014)` 会把
   二进制浮点误差原样带进来（``Decimal(0.00014) == Decimal('0.0001400000000000000106581...')``），
   再乘几百万 token 就是肉眼可见的偏差。配置项是 `float`（pydantic 侧），
   所以这里是**唯一**允许"转一手字符串"的地方。
2. **价格表按模型名前缀匹配**，不做精确匹配：供应商会在模型名后面挂版本号
   （``deepseek-chat-2026xx``、``gpt-4o-2024-11-20``），精确匹配会全部落到兜底价，
   而"所有模型都按兜底价算"是一种**静默错误**——数字看着有，其实全错。
3. **价格表只是量级估算**：真实账单还有缓存命中价、阶梯价、批处理折扣。
   这里的目标是"能回答这个功能一天花多少钱"，不是财务对账。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from knowflow.core.config import Settings

#: 成本保留 6 位小数，与 `messages.cost_usd DECIMAL(12,6)` / `traces.cost_usd` 完全一致。
#: 不一致的话，写库时会被数据库四舍五入，而日志里还是全精度，对账时对不上。
COST_QUANT: Final[Decimal] = Decimal("0.000001")

#: 展示用（`format_cost`）保留 6 位小数，人民币保留 4 位。
_CNY_QUANT: Final[Decimal] = Decimal("0.0001")

#: 每 1K token 的价格（美元）。
#:
#: ⚠ **以下均为参考价，仅用于量级估算，请以供应商账单为准**。
#: 价格会变、有缓存价/阶梯价，本项目不把这张表当财务依据，只用它回答
#: "这次请求大概是 1 分钱还是 1 块钱"。
#: 维护方式：**更长的前缀写前面**（`deepseek-reasoner` 必须排在 `deepseek` 前面），
#: 因为匹配是"第一个命中的前缀"。
_PRICE_TABLE: Final[tuple[tuple[str, str, str], ...]] = (
    # (模型名前缀, prompt 每 1K 美元, completion 每 1K 美元)
    ("deepseek-reasoner", "0.00055", "0.00219"),  # R1 类推理模型，输出贵 4 倍
    ("deepseek-chat", "0.00014", "0.00028"),  # V3 类，与项目默认配置一致
    ("deepseek", "0.00014", "0.00028"),  # 其它 deepseek-* 兜底
    ("gpt-4o-mini", "0.00015", "0.0006"),
    ("gpt-4o", "0.0025", "0.01"),
    ("gpt-4.1-mini", "0.0004", "0.0016"),
    ("gpt-4.1", "0.002", "0.008"),
    ("gpt-3.5", "0.0005", "0.0015"),
    ("o1-mini", "0.0011", "0.0044"),
    ("o1", "0.015", "0.06"),
    ("qwen-max", "0.0016", "0.0064"),
    ("qwen-plus", "0.0004", "0.0012"),
    ("qwen-turbo", "0.00005", "0.0002"),
    ("qwen", "0.0004", "0.0012"),  # 其它 qwen* 兜底
    ("glm-4", "0.00014", "0.00014"),
    ("claude-3-5-haiku", "0.0008", "0.004"),
    ("claude-3-5-sonnet", "0.003", "0.015"),
    ("claude", "0.003", "0.015"),
    ("moonshot", "0.0017", "0.0017"),
    ("ernie", "0.0008", "0.002"),
)


@dataclass(slots=True, frozen=True)
class Price:
    """一个模型的单价（美元 / 1K token）。

    `frozen=True`：价格是"值对象"，被下游缓存/复用时不希望被就地改写；
    `slots=True`：每秒可能构造几十次，省掉 `__dict__`。
    """

    prompt_per_1k: Decimal
    completion_per_1k: Decimal


def _dec(value: str | float | Decimal) -> Decimal:
    """统一把外部数值转成 `Decimal`。

    `float` 走 `str()` 中转是**刻意**的：`Decimal(0.00014)` 会把浮点误差冻进 Decimal，
    而 `Decimal(str(0.00014))` 得到干净的 `Decimal('0.00014')`。
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def resolve_price(model: str, settings: Settings) -> Price:
    """按模型名前缀查价，未命中则回落到配置里的默认单价。

    为什么用"前缀"而不是"相等"：见模块 docstring 第 2 条。
    为什么回落到配置：`get_settings()` 是配置的唯一来源，价格兜底值必须能被
    `.env` 调整（换供应商时不用改代码）。
    """
    name = (model or "").strip().lower()
    if name:
        for prefix, prompt, completion in _PRICE_TABLE:
            if name.startswith(prefix):
                return Price(prompt_per_1k=Decimal(prompt), completion_per_1k=Decimal(completion))
    return Price(
        prompt_per_1k=_dec(settings.pricing_prompt_per_1k),
        completion_per_1k=_dec(settings.pricing_completion_per_1k),
    )


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    settings: Settings,
) -> Decimal:
    """估算一次 LLM 调用的美元成本，定到 6 位小数。

    负数 token 按 0 处理：上游偶发会返回 ``-1`` 表示"未知"，
    直接参与运算会把费用**减少**，比忽略它危险得多。

    `ROUND_HALF_UP` 而不是默认的 `ROUND_HALF_EVEN`：财务口径四舍五入更符合直觉，
    也避免"0.0000005 舍成 0"时对不上账。
    """
    price = resolve_price(model, settings)
    p_tokens = max(int(prompt_tokens), 0)
    c_tokens = max(int(completion_tokens), 0)
    thousand = Decimal(1000)
    cost = (
        price.prompt_per_1k * Decimal(p_tokens) + price.completion_per_1k * Decimal(c_tokens)
    ) / thousand
    return cost.quantize(COST_QUANT, rounding=ROUND_HALF_UP)


def to_cny(usd: Decimal, settings: Settings) -> Decimal:
    """美元 -> 人民币，保留 4 位小数。

    汇率取配置 `usd_to_cny`（float），同样经过 `str()` 中转。
    保留 4 位而不是 6 位：汇率本身是约数，多留两位是**假精度**。
    """
    rate = _dec(settings.usd_to_cny)
    return (_dec(usd) * rate).quantize(_CNY_QUANT, rounding=ROUND_HALF_UP)


def format_cost(usd: Decimal, settings: Settings) -> str:
    """给日志用的可读成本，例：``$0.000231（≈¥0.0017）``。

    同时给两种货币是刻意的：开发者按美元想，老板按人民币看。
    """
    return f"${_dec(usd):.6f}（≈¥{to_cny(usd, settings):.4f}）"


__all__ = [
    "COST_QUANT",
    "Price",
    "estimate_cost",
    "format_cost",
    "resolve_price",
    "to_cny",
]
