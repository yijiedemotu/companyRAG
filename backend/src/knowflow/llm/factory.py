"""按配置构造对话模型。

**一个刻意的非对称设计**：

- **embedding 会自动降级**（`embeddings/factory.py`：本地模型加载失败 → 哈希向量），
  因为向量化失败会让整条链路完全不可用，降级至少能跑通、且 `/health` 会标记 degraded；
- **对话模型不自动降级**：配了 Key 但调用失败时**必须报错**，
  绝不偷偷换成离线抽取式回答 —— 那会让用户拿到质量极差却看起来正常的答案，
  是最糟糕的一种失败。宁可 502。

这条不对称是刻意的，面试时是个好问题："降级该不该静默？"
答案：**降级可以，静默不行**。凡降级必向上暴露（`/health` + 响应里的 `offline` 字段）。
"""

from __future__ import annotations

from knowflow.core.config import Settings, get_settings
from knowflow.core.logging import get_logger
from knowflow.llm.base import ChatModel
from knowflow.llm.mock import MockChatModel
from knowflow.llm.openai_compat import OpenAIChatModel

logger = get_logger(__name__)


def build_chat_model(settings: Settings | None = None) -> ChatModel:
    """主对话模型：写答案的那一个。"""
    cfg = settings or get_settings()
    if cfg.is_offline:
        logger.info("llm.mode", mode="offline", reason="未配置 OPENAI_API_KEY")
        return MockChatModel(cfg)
    logger.info(
        "llm.mode",
        mode="online",
        model=cfg.openai_model,
        base_url=cfg.openai_base_url,
    )
    return OpenAIChatModel(cfg)


def build_judge_model(settings: Settings | None = None) -> ChatModel:
    """判官模型：评测的 faithfulness / relevance 与 agent 的 grade / reflect 用它。

    单独一个入口是为了**能配便宜的小模型**：判官调用次数远多于主生成
    （一次问答可能 grade 一次 + reflect 一次），用主模型评主模型既贵又有自我偏好。
    `LLM_JUDGE_MODEL` 留空则复用主模型。
    """
    cfg = settings or get_settings()
    if cfg.is_offline:
        # 离线时 judge 也用 mock：它会返回格式合法的 score，让评测指标有值可算
        return MockChatModel(cfg)
    return OpenAIChatModel(cfg, model=cfg.judge_model)


__all__ = ["build_chat_model", "build_judge_model"]
