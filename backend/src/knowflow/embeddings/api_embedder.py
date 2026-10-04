"""OpenAI 兼容 `/embeddings` 接口的向量化实现。

**一句话定位**：把向量化从本地模型换成远程 HTTP 服务，用于「没有 GPU / 想统一
走网关 / 想用云厂商向量模型」的场景（DeepSeek、Qwen、自建 TEI 网关都兼容）。

**在整条链路中的位置**：与 `LocalBgeEmbedder` 完全同构 —— 同协议、同维度校验、
同「空文本 → 零向量」约定，所以上层（ingest / retrieve）不需要知道自己用的是谁。
`EMBEDDING_PROVIDER=api` 时由 `factory.build_embedder` 选中它。

**关键设计取舍：直接用 httpx，不用 `langchain_openai.OpenAIEmbeddings`**：

1. 错误可读性：本项目要求失败必须抛 `EmbeddingError` 并带上**状态码 + 响应体
   前 200 字符**（方便一眼看出是 Key 错、模型名错还是限流）。
   langchain 会把底层异常包成自己的层级，这些信息要翻两层源码才能捞出来。
2. 依赖更少：本模块只需要 httpx（已在 requirements 里，LangChain 内部也在用），
   少一层抽象就少一处版本兼容问题。
3. 协议本身很小：`POST /embeddings` + 重试 + 分批，30 行就写完了，
   引库的调试成本大于收益（和项目里「BM25 自研」是同一条原则）。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx
import structlog

from knowflow.core.config import Settings, get_settings
from knowflow.embeddings.base import (
    Embedder,
    EmbeddingError,
    build_text_for_embedding,
    zero_vector,
)

logger = structlog.get_logger(__name__)

# OpenAI 兼容协议里单次请求的条数上限是各家自己定的，64 是实测都能过的保守值。
# 它不是一个「业务可调参数」，所以放模块常量；真正的批大小仍由
# EMBEDDING_BATCH_SIZE 控制，取两者更小者。
_MAX_PROVIDER_BATCH: Final[int] = 64
_RETRY_BACKOFF_BASE_S: Final[float] = 0.5
_RETRY_BACKOFF_MAX_S: Final[float] = 4.0
_ERROR_BODY_PREVIEW_CHARS: Final[int] = 200
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# 预热用的探针文本：只要接口通了就说明 Key / 模型名 / 网络都没问题。
_WARMUP_TEXT: Final[str] = "ping"


class ApiEmbedder(Embedder):
    """OpenAI 兼容接口实现（`EMBEDDING_PROVIDER=api`）。"""

    provider = "api"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()
        self.model = self._settings.embedding_model
        self.dim = self._settings.embedding_dim
        base_url = self._settings.embedding_base_url.strip() or self._settings.openai_base_url
        self._base_url = base_url.rstrip("/")
        self._api_key = (
            self._settings.embedding_api_key.strip() or self._settings.openai_api_key.strip()
        )
        # 没有 Key 就直接在构造期报错：让 factory 立刻降级，
        # 而不是等到用户第一次上传文档时才 401（那时用户已经等了半天解析）。
        if not self._api_key:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=api 但没有任何可用的 Key："
                "请设置 EMBEDDING_API_KEY 或 OPENAI_API_KEY"
            )
        self.display_name = f"{self.model} (api, {self._base_url})"
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------ 资源管理
    @property
    def is_loaded(self) -> bool:
        """HTTP 实现是无状态的，随时可用。"""
        return True

    def _http(self) -> httpx.Client:
        """惰性创建并复用 `httpx.Client`（连接池复用比每次新建快很多）。"""
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=self._settings.llm_timeout,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
            return self._client

    def close(self) -> None:
        """关闭连接池（测试与优雅退出用）。"""
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def warmup(self) -> None:
        """发一条最小请求做探针：Key 错 / 模型名错 / 网关不通都会在这里暴露。

        代价是一次几乎不花钱的调用，换来「启动期就知道向量化能不能用」，
        比让第一个用户踩到 401 划算。
        """
        self.embed_documents([_WARMUP_TEXT])

    # ---------------------------------------------------------------- 编码
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """分批编码；空白文本本地短路成零向量，不浪费 token。"""
        if not texts:
            return []
        prepared = [build_text_for_embedding(t) for t in texts]
        batch_size = max(1, min(self._settings.embedding_batch_size, _MAX_PROVIDER_BATCH))
        vectors: list[list[float]] = []

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            filled = [i for i, text in enumerate(batch) if text]
            batch_vectors = [zero_vector(self.dim) for _ in batch]
            if filled:
                rows = self._post_embeddings([batch[i] for i in filled])
                for slot, row in zip(filled, rows, strict=True):
                    batch_vectors[slot] = row
            vectors.extend(batch_vectors)

        return vectors

    def embed_query(self, text: str) -> list[float]:
        """编码查询；接口侧一般不需要指令前缀，与文档侧同路径。"""
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else zero_vector(self.dim)

    # ------------------------------------------------------------ HTTP 细节
    def _post_embeddings(self, batch: list[str]) -> list[list[float]]:
        """发一批请求，带指数退避重试；失败抛 `EmbeddingError`。"""
        url = f"{self._base_url}/embeddings"
        payload: dict[str, Any] = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        attempts = self._settings.llm_max_retries + 1
        last_error = ""

        for attempt in range(attempts):
            try:
                response = self._http().post(url, json=payload)
            except Exception as exc:  # noqa: BLE001 - 超时/连接失败都归网络层
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 >= attempts:
                    break
                self._sleep_before_retry(attempt, reason=last_error)
                continue

            if response.status_code == httpx.codes.OK:
                return self._parse_response(response.json(), expected=len(batch))

            body = response.text[:_ERROR_BODY_PREVIEW_CHARS]
            if response.status_code in _RETRYABLE_STATUS and attempt + 1 < attempts:
                last_error = f"HTTP {response.status_code}: {body}"
                self._sleep_before_retry(attempt, reason=last_error)
                continue
            # 4xx（Key 错、模型名错、请求体错）重试没有意义，直接失败并带上原文
            raise EmbeddingError(
                f"embedding 接口返回 {response.status_code}"
                f"（url={url}, model={self.model}）：{body}"
            )

        raise EmbeddingError(
            f"embedding 接口调用失败（url={url}, model={self.model}, "
            f"已重试 {attempts} 次）：{last_error}"
        )

    def _sleep_before_retry(self, attempt: int, *, reason: str) -> None:
        delay = min(_RETRY_BACKOFF_BASE_S * (2**attempt), _RETRY_BACKOFF_MAX_S)
        logger.warning(
            "embedding 请求失败，准备重试",
            attempt=attempt + 1,
            delay_s=delay,
            model=self.model,
            reason=reason,
        )
        time.sleep(delay)

    def _parse_response(self, raw: object, *, expected: int) -> list[list[float]]:
        """解析响应体，并把维度不对的向量挡在入库之前。"""
        if not isinstance(raw, Mapping):
            raise EmbeddingError(f"embedding 接口返回的不是 JSON 对象：{raw!r:.200}")
        data = raw.get("data")
        if not isinstance(data, list) or not data:
            raise EmbeddingError(f"embedding 接口返回缺少 data 字段：{raw!r:.200}")

        entries = [item for item in data if isinstance(item, Mapping)]
        # 有些网关会乱序返回，按 index 排回去，否则向量和文本会错位对应
        entries.sort(key=lambda item: int(item.get("index", 0)))

        vectors: list[list[float]] = []
        for entry in entries:
            embedding = entry.get("embedding")
            if not isinstance(embedding, list):
                raise EmbeddingError(f"embedding 接口返回的 embedding 字段不是数组：{entry!r:.200}")
            vector = [float(x) for x in embedding]
            if len(vector) != self.dim:
                raise EmbeddingError(
                    f"接口返回的向量维度 {len(vector)} 与 EMBEDDING_DIM={self.dim} 不一致。"
                    "换 embedding 模型必须重建向量库：新旧向量的维度和语义空间都不同。"
                )
            vectors.append(vector)

        if len(vectors) != expected:
            raise EmbeddingError(
                f"embedding 接口返回条数不对：请求 {expected} 条，返回 {len(vectors)} 条"
            )
        return vectors


__all__ = ["ApiEmbedder"]
