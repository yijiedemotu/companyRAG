"""BGE-M3 本地加载与向量化自检。

跑法：
    .venv\\Scripts\\python.exe backend\\scripts\\_check_embedding.py <模型目录>

验证三件事：
  1. 模型能否加载（含离线路径）；
  2. 输出维度是否为 1024；
  3. 语义是否真的有效——「住宿标准」和「房费报销上限」的相似度
     必须明显高于「住宿标准」和「年假天数」。如果这一条不成立，
     说明向量化根本没起作用，检索一定会烂。
"""

from __future__ import annotations

import sys
import time

import numpy as np

DEFAULT_PATH = (
    r"D:\Projects\pythonProjects\agentProject1\projects\01-minimal-rag"
    r"\models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    from sentence_transformers import SentenceTransformer

    print(f"[1/4] 加载模型: {path}")
    t0 = time.perf_counter()
    model = SentenceTransformer(path, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"      加载成功，耗时 {load_s:.1f}s")
    dim = model.get_sentence_embedding_dimension()
    print(f"[2/4] 向量维度 = {dim}")
    assert dim == 1024, f"期望 1024 维，实际 {dim}"

    texts = [
        "一线城市住宿标准为每晚 600 元",
        "出差住房费用最多能报多少",
        "员工每年可以享受几天年假",
    ]
    t0 = time.perf_counter()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=3)
    enc_s = time.perf_counter() - t0
    print(
        f"[3/4] 编码 {len(texts)} 条，耗时 {enc_s:.2f}s（{enc_s / len(texts) * 1000:.0f} ms/条，CPU）"
    )

    sim_synonym = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    print("[4/4] 语义有效性检查：")
    print(f"      住宿标准  vs 房费报销上限（同义）  = {sim_synonym:.4f}")
    print(f"      住宿标准  vs 年假天数（无关）      = {sim_unrelated:.4f}")
    gap = sim_synonym - sim_unrelated
    print(f"      差值 = {gap:+.4f}")
    if gap <= 0.05:
        print("      结果：FAIL —— 语义区分度不足，向量化不可用")
        return 1
    print("      结果：OK —— 语义区分度充足，向量化可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
