"""相关性闸门阈值标定（不连数据库、不调大模型、零成本）。

**为什么这个脚本是必要的**：`VECTOR_MIN_SCORE` / `KEYWORD_MIN_COVERAGE`
这两个阈值是"防幻觉闸门"的开关。拍脑袋设的阈值一定不对，原因是：

**BGE-M3 的余弦相似度基线很高。** 实测（一次运行的原始数据）：

    相关问题 vs 正确片段   0.72
    无关问题 vs 任意片段   0.45 ~ 0.52      ← 基线，不是 0

也就是说"相似度 0.5"在 BGE-M3 上**不代表相关**，它可能只是"两段都是中文"。
如果用直觉设成 0.35，那么任何问题都能通过闸门，防幻觉机制完全失效
（这正是本脚本第一次跑出来的结论）。

本脚本的做法：
  1. 把 `data/samples/` 里的文档用**真实的解析器 + 切片器**切成 chunk；
  2. 用**真实的 embedding 模型**编码；
  3. 对 `data/eval_cases.jsonl` 里每条问句，分别计算
     - 正样本分数：与"期望命中文档"的 chunk 的最高相似度
     - 负样本分数：与"其它文档"chunk 的最高相似度
  4. 扫阈值，输出通过率 / 误放行率，取 F1 最优点。

跑法：
    .venv\\Scripts\\python.exe backend\\scripts\\calibrate_threshold.py
    .venv\\Scripts\\python.exe backend\\scripts\\calibrate_threshold.py --out data/threshold_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

import numpy as np  # noqa: E402

from knowflow.core.config import get_settings  # noqa: E402
from knowflow.embeddings.factory import build_embedder, get_embedder_status  # noqa: E402
from knowflow.ingest.chunkers import split_document  # noqa: E402
from knowflow.ingest.loaders import parse_bytes  # noqa: E402
from knowflow.retrieval.text import coverage  # noqa: E402


def load_corpus(settings, embedder):  # noqa: ANN001
    """解析 samples 目录 → 切片 → 向量化。返回 (chunk_texts, chunk_docs, matrix)。"""
    texts: list[str] = []
    docs: list[str] = []
    for path in sorted((settings.data_dir / "samples").glob("*")):
        if path.suffix.lower().lstrip(".") not in settings.allowed_extensions:
            continue
        parsed = parse_bytes(
            path.read_bytes(), path.name, allowed_extensions=settings.allowed_extensions
        )
        chunks = split_document(
            parsed,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chars=settings.chunk_min_chars,
            parent_chunk_size=settings.parent_chunk_size,
        )
        for chunk in chunks:
            # 检索命中的是子块，所以用子块正文评测（与线上一致）
            texts.append(chunk.content)
            docs.append(f"{path.name}")
    print(f"  语料：{len(texts)} 个切片，来自 {len(set(docs))} 个文档")
    vectors = embedder.embed_documents(texts)
    return texts, docs, np.asarray(vectors, dtype=np.float64)


def load_cases(settings) -> list[dict]:
    path = settings.data_dir / "eval_cases.jsonl"
    cases: list[dict] = []
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def evaluate(settings, embedder, texts, docs, matrix, cases, show: int = 6) -> dict:
    positives: list[float] = []
    negatives: list[float] = []
    pos_cov: list[float] = []
    neg_cov: list[float] = []

    sample_rows: list[tuple[str, str, float, float, float, float]] = []

    for case in cases:
        question = case["question"]
        expected = case.get("expected_doc")
        query = np.asarray(embedder.embed_query(question), dtype=np.float64)
        sims = matrix @ query  # 已归一化，点积即余弦

        pos_mask = np.array([d == expected for d in docs])
        neg_mask = ~pos_mask
        if not pos_mask.any():
            continue

        best_pos = float(sims[pos_mask].max())
        best_neg = float(sims[neg_mask].max()) if neg_mask.any() else 0.0
        positives.append(best_pos)
        negatives.append(best_neg)

        # 关键词覆盖率同理（用的是线上那份唯一实现）
        pos_texts = [t for t, d in zip(texts, docs, strict=True) if d == expected]
        neg_texts = [t for t, d in zip(texts, docs, strict=True) if d != expected]
        cov_pos = max((coverage(question, t) for t in pos_texts), default=0.0)
        cov_neg = max((coverage(question, t) for t in neg_texts), default=0.0)
        pos_cov.append(cov_pos)
        neg_cov.append(cov_neg)

        sample_rows.append((question, expected or "?", best_pos, best_neg, cov_pos, cov_neg))

    print(f"\n  参与评测的用例：{len(positives)} 条")
    print(
        f"  正样本相似度： min={min(positives):.4f} 均值={np.mean(positives):.4f} "
        f"中位={np.median(positives):.4f} max={max(positives):.4f}"
    )
    print(
        f"  负样本相似度： min={min(negatives):.4f} 均值={np.mean(negatives):.4f} "
        f"中位={np.median(negatives):.4f} max={max(negatives):.4f}"
    )
    print(f"  正样本覆盖率： min={min(pos_cov):.3f} 均值={np.mean(pos_cov):.3f}")
    print(
        f"  负样本覆盖率： min={min(neg_cov):.3f} 均值={np.mean(neg_cov):.3f} max={max(neg_cov):.3f}"
    )

    print(f"\n  抽样 {show} 条（问题 / 期望文档 / 正样本分 / 负样本分 / 正覆盖 / 负覆盖）：")
    for question, expected, bp, bn, cp, cn in sample_rows[:show]:
        print(
            f"    {question[:26]:28s} {expected:24s} vec+={bp:.3f} vec-={bn:.3f} "
            f"cov+={cp:.2f} cov-={cn:.2f}"
        )

    return {
        "positives": positives,
        "negatives": negatives,
        "pos_coverage": pos_cov,
        "neg_coverage": neg_cov,
        "samples": sample_rows,
    }


def sweep_vector(data: dict) -> tuple[float, list[dict]]:
    """扫向量阈值，取 F1 最优。"""
    positives, negatives = data["positives"], data["negatives"]
    rows: list[dict] = []
    best: tuple[float, float] | None = None

    for step in range(30, 96):
        threshold = step / 100
        tp = sum(1 for v in positives if v >= threshold)
        fn = len(positives) - tp
        fp = sum(1 for v in negatives if v >= threshold)
        tn = len(negatives) - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "recall": round(recall, 4),
                "false_accept": round(fp / len(negatives), 4) if negatives else 0.0,
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
        if best is None or f1 > best[0]:
            best = (f1, threshold)

    assert best is not None
    return best[1], rows


def sweep_keyword(data: dict) -> tuple[float, list[dict]]:
    pos, neg = data["pos_coverage"], data["neg_coverage"]
    rows: list[dict] = []
    best: tuple[float, float] | None = None
    for step in range(0, 61):
        threshold = step / 100
        tp = sum(1 for v in pos if v >= threshold)
        fn = len(pos) - tp
        fp = sum(1 for v in neg if v >= threshold)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "recall": round(recall, 4),
                "false_accept": round(fp / len(neg), 4) if neg else 0.0,
                "f1": round(f1, 4),
            }
        )
        if best is None or f1 > best[0]:
            best = (f1, threshold)
    assert best is not None
    return best[1], rows


def main() -> int:
    parser = argparse.ArgumentParser(description="相关性闸门阈值标定")
    parser.add_argument("--out", default="", help="把结果写到 JSON 文件（相对仓库根）")
    args = parser.parse_args()

    settings = get_settings()
    print("=" * 78)
    print("相关性闸门阈值标定")
    print("=" * 78)
    print(
        f"  当前配置： VECTOR_MIN_SCORE={settings.vector_min_score}  "
        f"KEYWORD_MIN_COVERAGE={settings.keyword_min_coverage}"
    )

    embedder = build_embedder(settings)
    if settings.embedding_provider == "local":
        from knowflow.embeddings.base import warmup_embedder

        warmup_embedder(embedder)
    status = get_embedder_status(embedder)
    print(f"  向量化： {status.mode}  dim={status.dim}")

    texts, docs, matrix = load_corpus(settings, embedder)
    cases = load_cases(settings)
    print(f"  用例： {len(cases)} 条")

    data = evaluate(settings, embedder, texts, docs, matrix, cases)

    vec_threshold, vec_rows = sweep_vector(data)
    kw_threshold, kw_rows = sweep_keyword(data)

    print("\n" + "-" * 78)
    print("向量阈值扫描（部分）：")
    print(f"  {'阈值':>6s} {'recall':>8s} {'误放行':>8s} {'precision':>10s} {'F1':>7s}")
    for row in vec_rows:
        if int(row["threshold"] * 100) % 5 == 0:
            mark = " ←最优" if row["threshold"] == vec_threshold else ""
            print(
                f"  {row['threshold']:6.2f} {row['recall']:8.3f} {row['false_accept']:8.3f} "
                f"{row['precision']:10.3f} {row['f1']:7.3f}{mark}"
            )

    print("\n覆盖率阈值扫描（部分）：")
    print(f"  {'阈值':>6s} {'recall':>8s} {'误放行':>8s} {'F1':>7s}")
    for row in kw_rows:
        if int(row["threshold"] * 100) % 10 == 0:
            mark = " ←最优" if row["threshold"] == kw_threshold else ""
            print(
                f"  {row['threshold']:6.2f} {row['recall']:8.3f} {row['false_accept']:8.3f} "
                f"{row['f1']:7.3f}{mark}"
            )

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(
        f"  建议 VECTOR_MIN_SCORE      = {vec_threshold:.2f}   （当前 {settings.vector_min_score}）"
    )
    print(
        f"  建议 KEYWORD_MIN_COVERAGE  = {kw_threshold:.2f}   "
        f"（当前 {settings.keyword_min_coverage}）"
    )
    print("\n  注意：闸门是「向量 OR 覆盖率」通过，所以两个阈值不是独立的——")
    print("  实际误放行率低于各自单独扫描的结果。要拿到联合效果，请跑")
    print("  scripts/run_eval.py（会真正过一遍 gate 并统计拒答率）。")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = settings.data_dir.parent / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "embedding_model": status.model,
                    "embedding_mode": status.mode,
                    "case_count": len(data["positives"]),
                    "vector": {
                        "recommended": vec_threshold,
                        "current": settings.vector_min_score,
                        "positives": data["positives"],
                        "negatives": data["negatives"],
                        "sweep": vec_rows,
                    },
                    "keyword": {
                        "recommended": kw_threshold,
                        "current": settings.keyword_min_coverage,
                        "positives": data["pos_coverage"],
                        "negatives": data["neg_coverage"],
                        "sweep": kw_rows,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  结果已写入 {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
