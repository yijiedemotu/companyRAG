"""KnowFlow —— 企业知识库智能问答服务。

分层（依赖方向永远向下，下层不反向 import 上层）：

    api/            HTTP 进出、参数校验、错误映射
    services/       业务编排（一次上传、一次问答的全流程）
    agent/ rag/     能力层：LangGraph 状态图、混合检索、提示词
    ingest/ retrieval/ vectorstore/ llm/   组件层
    db/ core/ observability/               基础设施

面试时可以这样一句话概括：**"控制流被画成图，模型只在节点内部做判断"**，
所以每一步都可观测、可兜底、可测试。
"""

__version__ = "1.0.0"
