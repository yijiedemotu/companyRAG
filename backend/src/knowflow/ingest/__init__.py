"""入库（ingest）子包：把用户上传的原始文件变成可检索、可喂给模型的文本块。

**一句话定位**：``bytes -> ParsedDocument -> list[Chunk]``，是"上传"到"向量化"之间的全部逻辑。

**在链路中的位置**：
``API 上传 -> loaders.parse_bytes -> chunkers.split_document -> embedding -> Chroma + MySQL``

**关键设计取舍**：加载层负责"把各种格式读成同一种东西"（归一化文本 + 页码 + 编码 + 警告），
切分层负责"在正确的边界上切开"（页 -> 小节 -> 标点），两层都不碰数据库、不碰模型，
因此可以脱离整个服务单独跑（离线评测与消融实验都直接调用这两个函数）。

**约定**：本包不读全局配置里"不该读的东西"——``loaders`` 只用
``allowed_extensions``；``split_document`` 的四个切分参数全部由调用方传入，
默认值由 API 层从 ``get_settings()`` 取，这样一次进程内可以扫多组参数。
"""

from __future__ import annotations

from knowflow.ingest.chunkers import Chunk, split_document
from knowflow.ingest.loaders import (
    ParsedDocument,
    ParsedPage,
    detect_ext,
    parse_bytes,
    parse_path,
)

__all__ = [
    "Chunk",
    "ParsedDocument",
    "ParsedPage",
    "detect_ext",
    "parse_bytes",
    "parse_path",
    "split_document",
]
