"""全项目配置的唯一来源。

设计原则（三句话）：

1. **一个字段只有一处定义**：所有可调参数都在这里，业务代码禁止出现魔法值
   （`chunk_size=600` 这种散落在五个文件里的写法，改一个参数要改五处，必然漏）。
2. **错误在启动期暴露**：用 pydantic 的校验器做交叉校验（如 `overlap < size`），
   配置写错时进程直接起不来，而不是第一次请求才 500。
3. **派生属性不制造第二个真相来源**：`is_offline`、`upload_dir` 这类都由已声明的
   字段算出来，不单独配。

读取顺序：显式入参 > 环境变量 > `backend/.env` > `.env`（仓库根） > 默认值。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# --------------------------------------------------------------------------------------
# 路径常量：全部由本文件位置推导，不依赖"当前工作目录"
#   config.py -> core -> knowflow -> src -> backend -> <repo root>
# --------------------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]

EmbeddingProvider = Literal["local", "hash", "api"]
VectorBackend = Literal["chroma", "memory"]
FusionStrategy = Literal["rrf", "weighted"]
RetrievalMode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]
ChatMode = Literal["rag", "agent"]
CheckpointBackend = Literal["memory", "sqlite"]


class Settings(BaseSettings):
    """KnowFlow 运行期配置。字段顺序 = `.env.example` 的顺序，方便对照。"""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env", REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------- 应用 ----------------
    app_name: str = "KnowFlow"
    app_version: str = "1.0.0"
    env: Literal["dev", "prod", "test"] = "dev"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api/v1"

    # CORS 同时支持两种写法：
    #   CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173   （逗号，更符合直觉）
    #   CORS_ORIGINS=["http://localhost:5173"]                     （JSON，pydantic 默认）
    # NoDecode 关掉 pydantic-settings 对复杂类型的 JSON 预解析，交给下面的校验器统一处理。
    # 不加 NoDecode，逗号写法会在**启动时**抛 JSONDecodeError —— 真实踩过的坑。
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # ---------------- 数据与数据库 ----------------
    data_dir: Path = REPO_ROOT / "data"
    database_url: str = "mysql+pymysql://root:1234@127.0.0.1:3306/knowflow?charset=utf8mb4"
    db_echo: bool = False
    db_pool_size: int = Field(default=10, ge=1, le=200)
    db_max_overflow: int = Field(default=20, ge=0, le=200)
    db_pool_recycle: int = Field(default=1800, ge=-1)

    # ---------------- 认证 ----------------
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = Field(default=1440, ge=1)

    # ---------------- 大模型（OpenAI 兼容协议） ----------------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com/v1"
    openai_model: str = "deepseek-chat"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_judge_model: str = ""  # 留空 = 复用 openai_model（评测/判官用小模型可省钱）

    # ---------------- 向量化 ----------------
    embedding_provider: EmbeddingProvider = "local"
    embedding_model: str = "BAAI/bge-m3"
    embedding_model_path: str = ""  # 指向本地模型目录则可完全离线
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_batch_size: int = Field(default=16, ge=1, le=512)
    embedding_dim: int = Field(default=1024, ge=8, le=8192)
    embedding_base_url: str = ""
    embedding_api_key: str = ""

    # ---------------- 向量库 ----------------
    vector_backend: VectorBackend = "chroma"
    chroma_dir: Path | None = None  # None -> data_dir / "chroma"

    # ⚠ 这两个阈值是**标定出来的，不是拍脑袋来的**。
    # 实测（backend/scripts/calibrate_threshold.py，BGE-M3 + 43 条标注用例）：
    #     正样本（问句 vs 正确片段）相似度  min=0.465 均值=0.676 中位=0.694
    #     负样本（问句 vs 其它片段）相似度  min=0.414 均值=0.570 中位=0.570
    # 关键结论：**BGE-M3 上"无关文本"的相似度基线约 0.57，不是 0**。
    # 所以直觉上的 0.35 会让闸门形同虚设（误放行率 100%，防幻觉机制完全失效）。
    # 扫描 F1 最优点：向量 0.63 / 覆盖率 0.44；这里取略宽松的 0.60 / 0.40 以避免误杀。
    # 换 embedding 模型必须重新标定 —— 余弦相似度的尺度完全变了。
    vector_min_score: float = Field(default=0.60, ge=-1.0, le=1.0)
    keyword_min_coverage: float = Field(default=0.40, ge=0.0, le=1.0)

    # ---------------- 切分 ----------------
    chunk_size: int = Field(default=600, ge=50, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=4000)
    chunk_min_chars: int = Field(default=80, ge=0)
    parent_chunk_size: int = Field(default=1800, ge=100)

    # ---------------- 检索 ----------------
    top_k: int = Field(default=5, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=500)
    fusion: FusionStrategy = "rrf"
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    rerank_enabled: bool = True
    rerank_top_n: int = Field(default=10, ge=1, le=50)
    autocut_enabled: bool = True
    autocut_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    max_context_chars: int = Field(default=8000, ge=500)

    # ---------------- Agent（LangGraph） ----------------
    agent_max_rewrite: int = Field(default=2, ge=0, le=5)
    agent_max_reflect: int = Field(default=1, ge=0, le=5)
    agent_grading_enabled: bool = True
    agent_checkpoint_backend: CheckpointBackend = "sqlite"

    # ---------------- 会话记忆 ----------------
    memory_max_turns: int = Field(default=6, ge=1, le=50)
    memory_max_chars: int = Field(default=4000, ge=200)

    # ---------------- 上传 ----------------
    max_upload_mb: int = Field(default=20, ge=1, le=500)
    allowed_extensions: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["md", "markdown", "txt", "pdf", "csv", "json"]
    )

    # ---------------- 日志 ----------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # ---------------- 限流 ----------------
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = Field(default=60, ge=1)

    # ---------------- 成本 ----------------
    pricing_prompt_per_1k: float = Field(default=0.00014, ge=0)
    pricing_completion_per_1k: float = Field(default=0.00028, ge=0)
    usd_to_cny: float = Field(default=7.2, gt=0)

    # ---------------- 模型下载镜像 ----------------
    hf_endpoint: str = "https://hf-mirror.com"

    # ==================================================================================
    # 校验器
    # ==================================================================================
    @field_validator("cors_origins", "allowed_extensions", mode="before")
    @classmethod
    def _split_comma_or_json(cls, v: Any) -> Any:
        """让列表型配置同时接受 `a,b,c` 与 `["a","b"]` 两种写法。"""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return []
            if raw.startswith("["):
                return json.loads(raw)
            return [item.strip() for item in raw.split(",") if item.strip()]
        return v

    @field_validator("allowed_extensions", mode="after")
    @classmethod
    def _normalize_ext(cls, v: list[str]) -> list[str]:
        return sorted({e.strip().lower().lstrip(".") for e in v if e.strip()})

    @field_validator("api_prefix", mode="after")
    @classmethod
    def _normalize_prefix(cls, v: str) -> str:
        prefix = "/" + v.strip().strip("/")
        return "" if prefix == "/" else prefix

    @model_validator(mode="after")
    def _cross_check(self) -> Settings:
        """交叉校验：这些错误一旦放过，会在运行期变成难查的怪问题。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP({self.chunk_overlap}) 必须小于 CHUNK_SIZE({self.chunk_size})，"
                "否则递归切分器无法收敛（会无限切分）"
            )
        if self.parent_chunk_size < self.chunk_size:
            raise ValueError(
                f"PARENT_CHUNK_SIZE({self.parent_chunk_size}) 不能小于 "
                f"CHUNK_SIZE({self.chunk_size})，父块必须能容纳子块"
            )
        if self.rerank_top_n > self.fetch_k:
            raise ValueError(
                f"RERANK_TOP_N({self.rerank_top_n}) 不应大于 FETCH_K({self.fetch_k})，"
                "重排只能作用于已召回的候选"
            )
        if self.env == "prod":
            if self.jwt_secret == "change-me-in-production":
                raise ValueError("生产环境必须设置 JWT_SECRET")
            if self.debug:
                raise ValueError("生产环境必须关闭 DEBUG")
        return self

    # ==================================================================================
    # 派生属性：只算不配，避免第二个真相来源
    # ==================================================================================
    @property
    def is_offline(self) -> bool:
        """没有可用 API Key = 离线模式。

        注意 `.strip()`：`OPENAI_API_KEY=" "`（一个空格）在 shell 里很常见，
        不做 strip 会误判成"在线"，然后在第一次提问时 401 —— 这类问题极难排查。
        """
        return not self.openai_api_key.strip()

    @property
    def llm_mode(self) -> str:
        return "offline" if self.is_offline else "online"

    @property
    def judge_model(self) -> str:
        return self.llm_judge_model.strip() or self.openai_model

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_path(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "agent_checkpoints.sqlite"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_dir / "model_cache"

    def ensure_dirs(self) -> list[Path]:
        """创建所有运行期目录，返回创建/确认过的目录列表（启动日志会打出来）。"""
        created: list[Path] = []
        for path in (self.data_dir, self.upload_dir, self.chroma_path):
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        return created

    def apply_runtime_env(self) -> None:
        """把需要影响第三方库行为的配置注入 `os.environ`。

        `HF_ENDPOINT` 必须在 import huggingface_hub 之前生效，所以由启动代码
        （container/main）在加载模型前调用一次。
        """
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        if self.embedding_model_path:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def public_snapshot(self) -> dict[str, Any]:
        """给 `/health` 与启动日志用的脱敏快照（绝不输出任何 Key）。"""
        return {
            "app": self.app_name,
            "version": self.app_version,
            "env": self.env,
            "api_prefix": self.api_prefix,
            "offline": self.is_offline,
            "llm_mode": self.llm_mode,
            "llm_model": self.openai_model,
            "llm_base_url": self.openai_base_url,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "vector_backend": self.vector_backend,
            "fusion": self.fusion,
            "top_k": self.top_k,
            "fetch_k": self.fetch_k,
            "rerank_enabled": self.rerank_enabled,
            "agent_max_rewrite": self.agent_max_rewrite,
            "agent_max_reflect": self.agent_max_reflect,
            "data_dir": str(self.data_dir),
            "chroma_dir": str(self.chroma_path),
            "database_url": _mask_dsn(self.database_url),
        }


def _mask_dsn(dsn: str) -> str:
    """隐藏 DSN 里的密码，日志与 /health 里只留 `user:***@host`。"""
    if "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    if "@" not in rest:
        return dsn
    creds, host = rest.rsplit("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        return f"{scheme}://{user}:***@{host}"
    return dsn


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。

    用 `lru_cache` 而不是模块级全局变量，是为了测试能 `get_settings.cache_clear()`
    换一套配置重建容器（否则测试之间会互相污染）。
    """
    return Settings()


__all__ = [
    "BACKEND_DIR",
    "REPO_ROOT",
    "ChatMode",
    "CheckpointBackend",
    "EmbeddingProvider",
    "FusionStrategy",
    "RetrievalMode",
    "Settings",
    "VectorBackend",
    "get_settings",
]
