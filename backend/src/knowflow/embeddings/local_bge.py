"""本地 sentence-transformers 向量化（默认 BAAI/bge-m3）。

**一句话定位**：用本地模型把文本编码成 1024 维归一化向量，零 API 成本、
离线可复现，是 `EMBEDDING_PROVIDER=local`（默认）的实现。

**在整条链路中的位置**：`ingest` 写向量库前调 `embed_documents`，
`retrieve` 用 `embed_query` 编查询；`factory.build_embedder` 会给它套一层
降级包装，所以本类抛出的 `EmbeddingError` 不一定冒到 HTTP 层，
而是可能变成 `/health` 里的 `embedding_mode=hash(fallback:...)`。

**关键设计取舍**：

1. **惰性加载**：构造函数**不**碰模型。进程启动就加载 2GB 权重会让 `/health`、
   文档列表等所有接口一起变慢，而且离线 CI 根本不该加载模型。
   真正加载发生在第一次 `embed_*` / `warmup()`，用 `threading.Lock` 保证只加载一次。
2. **模型来源三级回退**（见 `resolve_model_source`）：显式本地目录 →
   `DATA_DIR/model_cache/<模型名>` → 交给 sentence-transformers 走 HF 下载
   （会读 `HF_ENDPOINT` 镜像）。这样「生产离线」「开发用缓存」「没模型先跑通」
   三种场景各得其所。
3. **维度交叉校验**：加载后模型实际维度必须等于 `EMBEDDING_DIM`，不等直接抛错。
   换向量模型会同时改变**维度**和**语义空间**，旧向量全部作废；
   放过去的表现是「检索结果全错但不报错」，属于最贵的 bug。
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from knowflow.core.config import Settings, get_settings
from knowflow.embeddings.base import (
    Embedder,
    EmbeddingError,
    build_text_for_embedding,
    zero_vector,
)

if TYPE_CHECKING:  # 只为类型检查导入，运行期按需 import（模块导入必须保持轻量）
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

# HuggingFace 缓存目录的命名约定：`BAAI/bge-m3` -> `BAAI--bge-m3`。
_SLASHES = ("/", "\\")


def sanitize_model_name(model_name: str) -> str:
    """把模型名转成可以当目录名用的形式：`BAAI/bge-m3` -> `BAAI--bge-m3`。

    用 `--` 是 HuggingFace 自己的约定（`models--BAAI--bge-m3`），
    保持一致可以让「手抄一份模型目录进 DATA_DIR」这件事不容易写错。
    """
    name = model_name.strip()
    for ch in _SLASHES:
        name = name.replace(ch, "--")
    return name


class LocalBgeEmbedder(Embedder):
    """sentence-transformers 本地实现（默认 `BAAI/bge-m3`）。

    显式继承 `Embedder` 协议（而不是「结构上碰巧满足」）：mypy 会在类定义处校验
    实现完整性，将来协议加成员、这里少实现一个方法会**立刻**报错。
    """

    provider = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self.model = self._settings.embedding_model
        self.dim = self._settings.embedding_dim
        # 加载后会被改成真实设备（auto 时只有加载完才知道是 cpu 还是 cuda）
        self.display_name = f"{self.model} (local, {self._settings.embedding_device})"
        self._model: SentenceTransformer | None = None
        self._device: str | None = None
        # 惰性加载的互斥锁：并发请求同时首次触发加载时，只让一个人真正加载
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ 状态
    @property
    def is_loaded(self) -> bool:
        """模型是否已经在内存里（`/health` 与启动日志读它，不触发加载）。"""
        return self._model is not None

    @property
    def device(self) -> str | None:
        """实际推理设备；未加载时为 None。"""
        return self._device

    # ------------------------------------------------------------ 模型来源
    def resolve_model_source(self) -> str:
        """按三级优先级决定模型来源，返回给 `SentenceTransformer` 的入参。

        1. `EMBEDDING_MODEL_PATH` 非空且目录存在 → 直接用该目录（完全离线，
           连 HF 的缓存/索引都不查）；
        2. 否则 `DATA_DIR/model_cache/<sanitized_model_name>` 存在 → 用该缓存目录；
        3. 否则用模型名（如 `BAAI/bge-m3`）交给 sentence-transformers 下载。

        注意第 2 条与 HuggingFace 原生缓存布局**不同**：原生是
        `models--BAAI--bge-m3/snapshots/<sha>/`，本项目的约定是
        `model_cache/BAAI--bge-m3/`（即把 snapshot 目录内容直接放进去）。
        如果你手上是原生缓存的 snapshot，请用 `EMBEDDING_MODEL_PATH`
        显式指向它（第 1 优先级），那条路必然命中。
        """
        configured = self._settings.embedding_model_path.strip()
        if configured:
            path = Path(configured).expanduser()
            if path.is_dir():
                return str(path)
            # 配了却不存在：继续往下回退，但必须留下 warning，
            # 否则「以为在跑本地模型、其实在下模型/用兜底」这种事没人知道。
            logger.warning(
                "EMBEDDING_MODEL_PATH 指向的目录不存在，继续回退到缓存或模型下载",
                embedding_model_path=configured,
            )

        cached = self._settings.embedding_cache_dir / sanitize_model_name(self.model)
        if cached.is_dir():
            return str(cached)

        return self.model

    # -------------------------------------------------------------- 设备选择
    def _resolve_device(self) -> str:
        """把 `EMBEDDING_DEVICE=auto` 解析成真实的 `cpu` / `cuda`。

        torch 用 try 包住：这个模块在只跑 hash 兜底的环境里也必须能 import，
        那时 torch / sentence-transformers 可能压根没装。
        """
        requested = self._settings.embedding_device
        if requested == "cpu":
            return "cpu"
        try:
            # 故意在方法内 import：模块级 import torch 要 2~5 秒，
            # 而只跑 hash 兜底的环境根本不需要它。
            import torch
        except Exception as exc:  # noqa: BLE001
            logger.warning("导入 torch 失败，回退 cpu", error=f"{type(exc).__name__}: {exc}")
            return "cpu"

        cuda_available = False
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 - 驱动异常时不要让进程起不来
            logger.warning("torch.cuda.is_available() 失败，回退 cpu", error=str(exc))

        if not cuda_available:
            if requested == "cuda":
                logger.warning("EMBEDDING_DEVICE=cuda 但当前机器不可用，回退 cpu")
            return "cpu"

        try:
            logger.info("检测到可用 GPU", cuda_device=torch.cuda.get_device_name(0))
        except Exception as exc:  # noqa: BLE001 - 拿不到名字不影响推理
            logger.debug("读取 GPU 名称失败，忽略", error=str(exc))
        return "cuda"

    # -------------------------------------------------------------- 模型加载
    def _ensure_model(self) -> SentenceTransformer:
        """返回已加载的模型；需要时加载（双检锁，保证只加载一次）。"""
        if self._model is not None:
            return self._model
        with self._load_lock:
            # 双检：等锁期间可能已经有别的线程加载完了，直接复用
            if self._model is not None:
                return self._model
            source = self.resolve_model_source()
            device = self._resolve_device()
            started = time.perf_counter()
            try:
                # 延迟到真正要用时才 import：没装 sentence-transformers 的环境
                # （只跑 hash 兜底）依然能 import 本模块。
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(source, device=device)
            except Exception as exc:
                raise EmbeddingError(
                    f"加载本地向量模型失败（source={source}, device={device}）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            actual_dim = _read_embedding_dim(model)
            expected_dim = self._settings.embedding_dim
            if actual_dim != expected_dim:
                raise EmbeddingError(
                    f"向量维度不匹配：{source} 实际输出 {actual_dim} 维，"
                    f"而 EMBEDDING_DIM={expected_dim}。"
                    "换 embedding 模型必须重建向量库：新旧向量的维度和语义空间都不同，"
                    "混在同一个集合里会让检索结果完全错乱（且不会报错）。"
                    "请清空向量库目录后重新入库，或先把 EMBEDDING_DIM 改成模型真实维度。"
                )

            self._model = model
            self._device = device
            self.display_name = f"{self.model} (local, {device})"
            logger.info(
                "本地向量模型加载完成",
                model=self.model,
                source=source,
                device=device,
                dim=actual_dim,
                load_ms=int((time.perf_counter() - started) * 1000),
            )
            return model

    def warmup(self) -> None:
        """显式预热：启动流程调它可把「首次请求慢 2 秒」提前到启动阶段。"""
        self._ensure_model()

    def unload(self) -> None:
        """释放模型（测试用；CUDA 下顺带归还显存）。"""
        with self._load_lock:
            self._model = None
            self._device = None
            self.display_name = f"{self.model} (local, {self._settings.embedding_device})"
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - 没有 torch/没有 GPU 时无需清理
            logger.debug("释放 CUDA 缓存失败，忽略", error=str(exc))
        logger.info("本地向量模型已卸载", model=self.model)

    # ---------------------------------------------------------------- 编码
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """分批编码文档侧文本，返回 L2 归一化后的 `list[list[float]]`。

        分批大小取 `EMBEDDING_BATCH_SIZE`：一次把所有块塞进去会在长文档上
        把内存打爆（尤其 GPU），而 batch=1 又浪费算力。
        """
        if not texts:
            return []
        model = self._ensure_model()
        prepared = [build_text_for_embedding(t) for t in texts]
        batch_size = self._settings.embedding_batch_size
        vectors: list[list[float]] = []

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            # 空白块单独短路成零向量：交给模型编码会拿到全 0（还带 warning），
            # 不如在这里显式处理，保证「空输入 -> 零向量」是契约而不是副作用。
            filled = [i for i, text in enumerate(batch) if text]
            batch_vectors = [zero_vector(self.dim) for _ in batch]
            if filled:
                started = time.perf_counter()
                try:
                    encoded = model.encode(
                        [batch[i] for i in filled],
                        batch_size=len(filled),
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                except Exception as exc:
                    raise EmbeddingError(
                        f"本地向量模型推理失败（model={self.model}, "
                        f"batch_size={len(filled)}）：{type(exc).__name__}: {exc}"
                    ) from exc
                # `.tolist()` 而不是继续用 numpy：本项目对外的向量一律是纯 python
                # list，numpy 只允许出现在向量库检索内部，避免类型到处渗。
                rows: list[list[float]] = encoded.tolist()
                for slot, row in zip(filled, rows, strict=True):
                    batch_vectors[slot] = row
                logger.debug(
                    "本地向量化批次完成",
                    batch_size=len(filled),
                    ms=int((time.perf_counter() - started) * 1000),
                )
            vectors.extend(batch_vectors)

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """编码查询。bge-m3 不需要查询指令前缀，所以查文档与查查询同路径。"""
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else zero_vector(self.dim)


def _read_embedding_dim(model: SentenceTransformer) -> int:
    """读取模型输出维度，兼容 sentence-transformers 6.x 的方法改名。

    6.1.0 起 `get_sentence_embedding_dimension()` 改名为 `get_embedding_dimension()`
    （旧名仍可用但会打 FutureWarning）。这里优先用新名，避免启动日志被警告刷屏。
    """
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = getattr(model, "get_sentence_embedding_dimension", None)
    if getter is None:
        raise EmbeddingError("无法从 sentence-transformers 模型读取输出维度")
    try:
        return int(getter())
    except Exception as exc:
        raise EmbeddingError(f"读取模型输出维度失败：{type(exc).__name__}: {exc}") from exc


__all__ = ["LocalBgeEmbedder", "sanitize_model_name"]
