"""文档与切片契约。

字段来源：契约文档 5.3 节。

两个字段值得单独说明：

- `DocumentOut.error_code` / `error_message`：文档入库是异步多阶段的
  （UPLOADED → PARSING → CHUNKING → EMBEDDING → READY，任一步可 FAILED）。
  前端靠这两个字段显示"为什么失败"，没有它们用户只会看到状态永远停在 FAILED，
  只能翻服务器日志。契约里明确要求保留，**不要因为"看着像内部信息"删掉**。
- `ChunkOut` 故意**不返回 `parent_content`**：它是喂给模型的父块，
  比子块大 3 倍。文档详情页展示切片列表时返回它，一页 20 条就能有几百 KB，
  而前端一个字段都用不上（需要父块的是问答链路，走服务端内部）。
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from knowflow.db.models import (
    DOC_STATUS_CHUNKING,
    DOC_STATUS_EMBEDDING,
    DOC_STATUS_FAILED,
    DOC_STATUS_PARSING,
    DOC_STATUS_READY,
    DOC_STATUS_UPLOADED,
)
from knowflow.schemas.common import UtcDatetime

__all__ = [
    "ChunkOut",
    "DocumentOut",
    "DocumentStatus",
]

# 文档状态字面量集合，顺序即生命周期顺序。
# `Literal[...]` 只接受字面量（mypy 硬限制），所以下面这条断言是"两处取值
# 不许漂移"的守卫：谁改了 ORM 常量没同步这里，import 就失败。
DocumentStatus = Literal[
    "UPLOADED",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "READY",
    "FAILED",
]

assert get_args(DocumentStatus) == (
    DOC_STATUS_UPLOADED,
    DOC_STATUS_PARSING,
    DOC_STATUS_CHUNKING,
    DOC_STATUS_EMBEDDING,
    DOC_STATUS_READY,
    DOC_STATUS_FAILED,
), "DocumentStatus 与 db/models/knowledge.py 的状态常量不一致"


class DocumentOut(BaseModel):
    """文档输出（契约 5.3 节的 `DocumentOut`，字段顺序照抄）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="文档 ID")
    kb_id: int = Field(description="所属知识库 ID")
    filename: str = Field(description="原始文件名（含扩展名，展示用）")
    ext: str = Field(description="小写扩展名，不含点，如 pdf")
    size_bytes: int = Field(description="原始文件字节数")
    status: DocumentStatus = Field(
        description="入库状态：UPLOADED/PARSING/CHUNKING/EMBEDDING/READY/FAILED"
    )
    parser: str | None = Field(
        default=None, description="实际使用的解析器：markdown/text/pdf/csv/json"
    )
    page_count: int | None = Field(default=None, description="页数，仅 PDF 有值")
    char_count: int = Field(default=0, description="解析后的字符总数")
    chunk_count: int = Field(default=0, description="切出的切片数")
    token_count: int = Field(default=0, description="切片 token 数合计")
    ingest_ms: int | None = Field(default=None, description="入库总耗时（毫秒），未完成时为 null")
    error_code: str | None = Field(
        default=None,
        description="失败错误码，如 DOCUMENT_PARSE_ERROR；前端据此显示失败原因",
    )
    error_message: str | None = Field(
        default=None, description="失败的人类可读原因，最长 1024 字符"
    )
    created_at: UtcDatetime = Field(default=None, description="上传时间，UTC ISO8601 带 Z")


class ChunkOut(BaseModel):
    """切片输出（契约 5.3 节的 `ChunkOut`）。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="切片 ID")
    doc_id: int = Field(description="所属文档 ID")
    chunk_index: int = Field(description="文档内序号，从 0 开始递增")
    content: str = Field(description="切片正文（检索单位）")
    char_count: int = Field(default=0, description="正文字符数")
    token_count: int = Field(default=0, description="正文 token 数")
    page_no: int | None = Field(default=None, description="页码，仅 PDF 有值")
    section_path: str | None = Field(
        default=None, description="所属小节路径，如「员工报销制度 > 差旅报销标准」"
    )
    vector_id: str = Field(
        description="向量库锚点，固定为 `{doc_id}:{chunk_index}`，一致性校验靠它比对"
    )
