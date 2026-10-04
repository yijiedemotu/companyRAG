"""服务层：一次业务流程的事务边界与编排。

分层约定（依赖方向永远向下）：

    api/        只做 HTTP：参数校验、鉴权、把 service 结果转成 schema
    services/   业务编排：一次上传 / 一次问答 / 一次评测的完整流程
    agent/ rag/ 能力层
    组件层      ingest / retrieval / vectorstore / llm / db

**api/ 里不允许出现 SQLAlchemy 查询**，**services/ 里不允许出现 HTTP 概念**
（HTTPException、Request、status_code）。跨过这条线，测试就没法脱离 HTTP 跑。
"""

from knowflow.services.auth import AuthService
from knowflow.services.chat import ChatAnswer, ChatService
from knowflow.services.chunks import SqlChunkEnricher, fetch_chunk_infos
from knowflow.services.document import DocumentService
from knowflow.services.knowledge_base import KBService

__all__ = [
    "AuthService",
    "ChatAnswer",
    "ChatService",
    "DocumentService",
    "KBService",
    "SqlChunkEnricher",
    "fetch_chunk_infos",
]
