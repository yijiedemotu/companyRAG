"""检索质量评测与消融实验 CLI。

**这个脚本回答的问题是**："我刚才改的那个参数，到底让效果变好了还是变差了？"

它把"我感觉更好了"变成可比较的数字：

    消融（ablation）：同一份标注集，分别用
        vector        纯向量召回
        bm25          纯关键词召回
        hybrid        两路融合（RRF）
        hybrid_rerank 融合 + 重排
    跑同一套指标（recall@k / MRR / nDCG / 命中率 / 延迟），输出对比表。

**并且做显著性检验**：评测集只有 43 条时，recall 从 0.79 涨到 0.82
很可能只是噪声（一条题翻转就是 2.3 个百分点）。
所以除了均值，还会用 `paired_bootstrap_test` 给出 p 值与置信区间 ——
不做这一步，就会把随机波动当成"优化成果"写进简历。

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\run_eval.py
    .venv\\Scripts\\python.exe backend\\scripts\\run_eval.py --modes vector,hybrid --top-k 5
    .venv\\Scripts\\python.exe backend\\scripts\\run_eval.py --out data/eval_report.json
    .venv\\Scripts\\python.exe backend\\scripts\\run_eval.py --dataset knowflow-golden-v1 --llm-judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

from sqlalchemy import select  # noqa: E402

from knowflow.container import build_container  # noqa: E402
from knowflow.core.config import get_settings  # noqa: E402
from knowflow.core.logging import configure_logging, get_logger  # noqa: E402
from knowflow.db.models import EvalCaseResult, EvalDataset  # noqa: E402
from knowflow.evaluation.harness import run_evaluation  # noqa: E402
from knowflow.evaluation.stats import paired_bootstrap_test  # noqa: E402

logger = get_logger(__name__)

METRIC_COLUMNS = [
    ("hit_rate", "命中率", "{:.3f}"),
    ("recall_at_k", "recall@k", "{:.3f}"),
    ("mrr", "MRR", "{:.3f}"),
    ("ndcg", "nDCG", "{:.3f}"),
    ("avg_latency_ms", "平均延迟ms", "{:.0f}"),
]


def print_table(results: dict[str, dict[str, Any]], baseline: str) -> None:
    header = f"  {'模式':<16s}" + "".join(f"{label:>12s}" for _, label, _ in METRIC_COLUMNS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mode, metrics in results.items():
        row = f"  {mode:<16s}"
        for key, _label, fmt in METRIC_COLUMNS:
            value = metrics.get(key)
            row += f"{'—':>12s}" if value is None else f"{fmt.format(value):>12s}"
        if mode == baseline:
            row += "   ← 基线"
        print(row)


def print_significance(
    results: dict[str, dict[str, Any]], per_case: dict[str, dict[int, float]], baseline: str
) -> None:
    """对每个非基线模式，与基线做成对 bootstrap 检验。"""
    base_values = per_case.get(baseline, {})
    if not base_values:
        print("  （基线没有逐条结果，跳过显著性检验）")
        return

    print(f"\n  与基线「{baseline}」的成对 bootstrap 检验（recall@k，n_boot=2000, seed=42）：")
    print(f"  {'对比':<34s}{'Δ':>10s}{'p 值':>10s}{'95% CI':>22s}{'显著':>8s}")
    print("  " + "-" * 82)
    for mode, metrics in results.items():
        if mode == baseline:
            continue
        current = per_case.get(mode, {})
        shared = sorted(set(base_values) & set(current))
        if not shared:
            continue
        a = [current[case_id] for case_id in shared]
        b = [base_values[case_id] for case_id in shared]
        try:
            report = paired_bootstrap_test(a, b, n_boot=2000, seed=42)
        except ValueError as exc:
            print(f"  {mode:<34s} 无法检验：{exc}")
            continue
        ci = report["ci95"]
        delta = report["delta"]
        low, high = float(ci["low"]), float(ci["high"])
        mark = "是" if report["significant"] else "否"
        note = ""
        if not report["significant"] and abs(delta) > 0:
            # 这是本脚本最想传达的一句话
            note = "  ← 差异不显著，不要当成改进"
        contrast = f"{mode} vs {baseline}"
        interval = f"[{low:+.4f}, {high:+.4f}]"
        print(
            f"  {contrast:<34s}{delta:>+10.4f}{report['p_value']:>10.3f}"
            f"{interval:>22s}{mark:>8s}{note}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="KnowFlow 检索质量评测与消融实验")
    parser.add_argument("--dataset", default="knowflow-golden-v1", help="数据集名称")
    parser.add_argument(
        "--modes",
        default="vector,bm25,hybrid,hybrid_rerank",
        help="要对比的检索模式（逗号分隔）",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--baseline", default="hybrid_rerank", help="显著性检验的基线模式")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="额外用 LLM 判官算 faithfulness / answer_relevance（慢且花钱）",
    )
    parser.add_argument("--out", default="", help="把报告写成 JSON（相对仓库根）")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(
        level="WARNING" if not settings.debug else settings.log_level, json_output=settings.log_json
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print("=" * 78)
    print("KnowFlow 检索质量评测（消融实验）")
    print("=" * 78)
    print(f"  数据集   : {args.dataset}")
    print(f"  模式     : {', '.join(modes)}")
    print(f"  top_k    : {args.top_k}")
    print(
        f"  融合     : {settings.fusion}   rerank={settings.rerank_enabled}   "
        f"autocut={settings.autocut_enabled}"
    )
    print(f"  闸门阈值 : 向量 {settings.vector_min_score} / 覆盖率 {settings.keyword_min_coverage}")
    print(f"  LLM 判官 : {args.llm_judge}")

    container = build_container(settings)

    results: dict[str, dict[str, Any]] = {}
    per_case: dict[str, dict[int, float]] = {}
    started_all = time.perf_counter()

    try:
        with container.session_factory() as session:
            dataset = session.execute(
                select(EvalDataset).where(EvalDataset.name == args.dataset)
            ).scalar_one_or_none()
            if dataset is None:
                print(f"\n  ✗ 数据集 {args.dataset!r} 不存在。")
                print("    先跑： .venv\\Scripts\\python.exe backend\\scripts\\seed_demo.py")
                return 2
            dataset_id = dataset.id
            print(f"\n  数据集 id={dataset_id}，用例 {dataset.case_count} 条")

        for mode in modes:
            print(f"\n  ▶ 运行 mode={mode} ...")
            t0 = time.perf_counter()
            with container.session_factory() as session:
                services = container.build_request_services(session)
                run_info = run_evaluation(
                    session=session,
                    dataset_id=dataset_id,
                    mode=mode,
                    top_k=args.top_k,
                    settings=settings,
                    retriever=services.retriever,
                    use_llm_judge=args.llm_judge,
                )
                session.commit()
            elapsed = time.perf_counter() - t0
            metrics = run_info.get("metrics", run_info)
            results[mode] = metrics
            run_id = run_info.get("run_id")

            if run_id is not None:
                with container.session_factory() as session:
                    rows = list(
                        session.execute(
                            select(EvalCaseResult.case_id, EvalCaseResult.recall_at_k).where(
                                EvalCaseResult.run_id == run_id
                            )
                        ).all()
                    )
                per_case[mode] = {int(case_id): float(recall or 0.0) for case_id, recall in rows}

            print(
                f"    完成，耗时 {elapsed:.1f}s  "
                f"hit_rate={metrics.get('hit_rate')}  recall@{args.top_k}="
                f"{metrics.get('recall_at_k')}  mrr={metrics.get('mrr')}"
            )
    finally:
        container.close()

    print("\n" + "=" * 78)
    print("对比结果")
    print("=" * 78)
    print_table(results, args.baseline)
    print_significance(results, per_case, args.baseline)

    print(f"\n  总耗时 {time.perf_counter() - started_all:.1f}s")
    print("\n  怎么读这张表：")
    print("    · recall@k  —— 正确答案所在的文档有没有被召回（检索的底线指标）")
    print("    · MRR       —— 第一个正确结果的排名倒数（越靠前越高）")
    print("    · nDCG      —— 排序质量（考虑位置折扣）")
    print("    · 延迟      —— 重排是最贵的一步，通常能换来 recall 提升")
    print("    · 显著性    —— 不显著就别写进简历，那是噪声")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = settings.data_dir.parent / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "top_k": args.top_k,
                    "baseline": args.baseline,
                    "config": {
                        "fusion": settings.fusion,
                        "rerank_enabled": settings.rerank_enabled,
                        "vector_min_score": settings.vector_min_score,
                        "keyword_min_coverage": settings.keyword_min_coverage,
                    },
                    "results": results,
                    "per_case_recall": {
                        mode: {str(k): v for k, v in values.items()}
                        for mode, values in per_case.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  报告已写入 {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
