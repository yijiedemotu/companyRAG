"""评测层：把「感觉更好了」变成可对比、可回归的数字。

**一句话定位**：第 ⑦ 层——标注集 + 指标 + 消融 + 显著性检验 + CI 门槛。

**在链路中的位置**：

    eval_datasets/eval_cases（题库）
      -> evaluation.harness 逐条跑检索/生成
      -> evaluation.metrics 算 recall/MRR/nDCG/...
      -> evaluation.stats 判断提升是否显著
      -> eval_runs.metrics_json -> /eval/* 接口与 CI 门槛

**关键设计取舍**：

1. **指标是纯函数、零依赖**（`metrics.py`）：指标算错是最隐蔽的 bug，
   所以不碰数据库、不碰模型，能用手算值逐条对照。
2. **显著性检验手写、不引 scipy**（`stats.py`）：28 条用例的评测集上，
   "提升了 4 个点"很可能只是噪声；bootstrap 是几行代码就能解决的事。
3. **切词只有一份实现**（`metrics.tokenize`）：闸门阈值是在 BM25 的切词口径下
   校准出来的，两处切词不一致会让阈值失去意义。

`harness.py`（编排与 LLM judge）由上层负责，本包只提供它需要的三块能力。
"""

from __future__ import annotations

from knowflow.evaluation.datasets import (
    CaseSpec,
    case_from_dict,
    get_or_create_dataset,
    load_jsonl,
    seed_dataset,
)
from knowflow.evaluation.metrics import (
    citation_precision,
    hit_rate,
    keyword_coverage,
    mrr,
    ndcg_at_k,
    recall_at_k,
    tokenize,
)
from knowflow.evaluation.stats import (
    ConfidenceInterval,
    bootstrap_ci,
    paired_bootstrap_test,
)

__all__ = [
    "CaseSpec",
    "ConfidenceInterval",
    "bootstrap_ci",
    "case_from_dict",
    "citation_precision",
    "get_or_create_dataset",
    "hit_rate",
    "keyword_coverage",
    "load_jsonl",
    "mrr",
    "ndcg_at_k",
    "paired_bootstrap_test",
    "recall_at_k",
    "seed_dataset",
    "tokenize",
]
