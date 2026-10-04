"""统计显著性：bootstrap 置信区间与成对检验。

**一句话定位**：回答"这次提升是真的，还是在噪声上瞎调参？"

**在链路中的位置**：
``/eval/compare?run_a&run_b`` -> **本模块** -> ``{delta, ci95, p_value, significant}``。

**为什么要做显著性检验**（这段是回答"你做过什么工程判断"的关键）：
本项目的评测集只有 28 条用例。28 条里 recall@5 从 0.78 "涨"到 0.82，
看起来是 +4 个点，实际上只对应**一条用例**的翻转——很可能只是采样噪声。
不做检验的后果不是"少了个数字"，而是**优化目标错了**：
你会继续朝着随机波动的方向调参（今天 +4 点明天 -3 点），
最后得到一个在真实流量上并不更好的配置，还花掉了大量实验时间。
bootstrap 不需要假设正态分布（指标是有界的、非正态的，t 检验不适用），
成本只有几百次重采样，是这个规模下最合适的工具。

**关键设计取舍**：

1. **手写实现，不依赖 scipy**：项目没装 scipy（`requirements.txt` 里没有），
   而这里只用到"重采样 + 取分位点"两件事，几十行就够，引一个 30MB 的科学计算栈
   不划算。
2. **`random.Random(seed)` 显式实例化**，不用全局 `random`：
   全局函数的种子会被同一进程里其它代码影响，导致"同样的数据两次跑出不同 CI"，
   而这种不确定性会让 CI 门槛变成薛定谔的通过。
3. **成对检验（paired）而不是两组独立检验**：两次评测跑的是**同一批用例**，
   成对差分消掉了"题目难度"这个共同方差，检验功效高得多。
   代价是要求输入等长——不等长直接 `ValueError`，宁可报错也不悄悄对比错数据。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: 默认重采样次数。2000 次对应 p 值分辨率 ~0.0005，足够判定 0.05 门槛；
#: 再往上加（10000）只是让浏览器多等几秒，结论不会变。
DEFAULT_N_BOOT = 2000
DEFAULT_CONFIDENCE = 0.95


@dataclass(slots=True, frozen=True)
class ConfidenceInterval:
    """bootstrap 置信区间。`point` 是观测均值（不是重采样均值——后者只是它的估计）。"""

    point: float
    low: float
    high: float
    confidence: float


def _percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（与 `observability.metrics.percentile` 同口径）。

    这里刻意**再写一份私有实现**而不是互相 import：两个包处在不同层
    （观测层 vs 评测层），让评测层依赖观测层会把"纯函数、零依赖"这条
    （见 evaluation/metrics.py 的设计说明）破坏掉。
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    ratio = min(max(float(p), 0.0), 100.0) / 100.0
    position = ratio * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ConfidenceInterval:
    """对均值做有放回重采样的置信区间。

    返回 `ConfidenceInterval(point=观测均值, low, high, confidence)`；
    **空输入返回全 0 的 CI**（不抛异常，方便面板直接渲染）。

    `n_boot <= 0` 视为"不重采样"，此时 low = high = point
    （调用方显式关闭 bootstrap 时应该得到确定结果，而不是崩溃）。
    """
    if not values:
        return ConfidenceInterval(point=0.0, low=0.0, high=0.0, confidence=confidence)

    data = [float(v) for v in values]
    point = sum(data) / len(data)
    if n_boot <= 0:
        return ConfidenceInterval(point=point, low=point, high=point, confidence=confidence)

    rng = random.Random(seed)
    size = len(data)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(size):
            total += data[rng.randrange(size)]
        means.append(total / size)

    alpha = (1.0 - confidence) / 2.0
    low = _percentile(means, alpha * 100.0)
    high = _percentile(means, (1.0 - alpha) * 100.0)
    return ConfidenceInterval(point=point, low=low, high=high, confidence=confidence)


def paired_bootstrap_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
) -> dict[str, Any]:
    """成对 bootstrap 检验：A 与 B 是否**真的**有差异。

    约定 ``delta = mean_a - mean_b``（正数 = A 更好），返回::

        {"mean_a", "mean_b", "delta", "p_value", "significant",
         "ci95": {"low", "high"}, "n", "n_boot", "confidence"}

    算法（标准 paired bootstrap）：

    1. 先算成对差分 ``d_i = a_i - b_i``（成对差分把"题目难度"这个共同方差消掉）；
    2. 有放回重采样差分，得到 delta 的 bootstrap 分布；
    3. **p 值**：观测 delta 落在 0 的哪一侧，就数分布里落在**另一侧**的比例，
       再乘 2（双侧检验），最后夹到 ``[0, 1]``。
       夹取是必须的：比例可能略大于 0.5（分布整体偏向反侧），``2 * 0.51 = 1.02``
       会输出"p 值大于 1"这种非法数字，前端显示出来就是 bug。

    `significant = p_value < 0.05`（本项目约定的门槛，不做可配置——
    门槛可配置时大家总会调到自己满意的那个值，这就是 p-hacking）。

    两个输入**长度必须相等**：成对检验的前提是"同一批用例、同一顺序"，
    长度不等说明对比的是两次不同的数据集，结论没有意义 -> `ValueError`。
    """
    if len(a) != len(b):
        raise ValueError(
            f"成对检验要求两次评测的用例数相同（同一批用例、同一顺序），"
            f"实际得到 a={len(a)} 与 b={len(b)}。"
            "长度不等通常意味着对比了两个不同的数据集，这时任何 delta 都没有意义。"
        )
    if not a:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "delta": 0.0,
            "p_value": 1.0,
            "significant": False,
            "ci95": {"low": 0.0, "high": 0.0},
            "n": 0,
            "n_boot": n_boot,
            "confidence": DEFAULT_CONFIDENCE,
        }

    values_a = [float(v) for v in a]
    values_b = [float(v) for v in b]
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    delta = mean_a - mean_b
    diffs = [x - y for x, y in zip(values_a, values_b, strict=True)]
    size = len(diffs)

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    for _ in range(max(int(n_boot), 0)):
        total = 0.0
        for _ in range(size):
            total += diffs[rng.randrange(size)]
        boot_deltas.append(total / size)

    if not boot_deltas:
        # 不重采样：无分布可用，退化成"只看观测值符号"，且明确标记不显著
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "p_value": 1.0,
            "significant": False,
            "ci95": {"low": delta, "high": delta},
            "n": size,
            "n_boot": 0,
            "confidence": DEFAULT_CONFIDENCE,
        }

    # 双侧 p 值：观测 delta 偏正 -> 数落在 0 及以下的（含 0）比例，反之亦然
    if delta >= 0:
        opposite = sum(1 for value in boot_deltas if value <= 0.0)
    else:
        opposite = sum(1 for value in boot_deltas if value >= 0.0)
    p_value = min(max(2.0 * opposite / len(boot_deltas), 0.0), 1.0)

    alpha = (1.0 - DEFAULT_CONFIDENCE) / 2.0
    return {
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
        "delta": round(delta, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "ci95": {
            "low": round(_percentile(boot_deltas, alpha * 100.0), 6),
            "high": round(_percentile(boot_deltas, (1.0 - alpha) * 100.0), 6),
        },
        "n": size,
        "n_boot": len(boot_deltas),
        "confidence": DEFAULT_CONFIDENCE,
    }


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_N_BOOT",
    "ConfidenceInterval",
    "bootstrap_ci",
    "paired_bootstrap_test",
]
