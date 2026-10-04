"""异常体系与统一错误响应。

设计原则：

1. **业务代码只抛语义化异常**（`KBNameConflictError`），不关心 HTTP 状态码；
   映射表集中在这里，改一个错误码只改一处。
2. **错误响应必须带 `request_id`**：用户截图一个 request_id，运维就能在日志里
   定位到那一次请求的全部信息。没有它，线上排障只能靠时间戳猜。
3. **`code` 是给机器看的稳定标识**（前端按它做分支），`message` 是给人看的中文。
   前端绝不应该去匹配 `message` 文本。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class KnowFlowError(Exception):
    """所有业务异常的基类。

    子类通过类属性声明 `code` / `http_status` / `default_message`，
    调用方只需 `raise KBNameConflictError(f"知识库 {name!r} 已存在")`。
    """

    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    default_message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: Any = None,
        code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.detail = detail
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        super().__init__(self.message)

    def to_dict(self, *, request_id: str | None = None) -> dict[str, Any]:
        """转成对外统一的错误信封（见 `docs/01-数据库与接口契约.md` 第一节）。"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "detail": self.detail,
                "request_id": request_id,
                "timestamp": _utc_now_iso(),
            }
        }

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------------------
# 400 客户端数据问题
# --------------------------------------------------------------------------------------
class ValidationError(KnowFlowError):
    code = "VALIDATION_ERROR"
    http_status = 400
    default_message = "请求参数不合法"


class UnsupportedFileTypeError(KnowFlowError):
    code = "UNSUPPORTED_FILE_TYPE"
    http_status = 400
    default_message = "不支持的文件类型"


class FileTooLargeError(KnowFlowError):
    code = "FILE_TOO_LARGE"
    http_status = 400
    default_message = "文件超过大小上限"


class DocumentParseError(KnowFlowError):
    code = "DOCUMENT_PARSE_ERROR"
    http_status = 400
    default_message = "文档解析失败"


class DocumentEmptyError(KnowFlowError):
    code = "DOCUMENT_EMPTY"
    http_status = 400
    default_message = "文档解析后没有有效文本"


# --------------------------------------------------------------------------------------
# 401 / 403
# --------------------------------------------------------------------------------------
class UnauthorizedError(KnowFlowError):
    code = "UNAUTHORIZED"
    http_status = 401
    default_message = "未认证或登录已过期"


class BadCredentialsError(KnowFlowError):
    code = "BAD_CREDENTIALS"
    http_status = 401
    default_message = "用户名或密码错误"


class ForbiddenError(KnowFlowError):
    code = "FORBIDDEN"
    http_status = 403
    default_message = "没有权限执行该操作"


# --------------------------------------------------------------------------------------
# 404
# --------------------------------------------------------------------------------------
class NotFoundError(KnowFlowError):
    code = "NOT_FOUND"
    http_status = 404
    default_message = "资源不存在"


class KBNotFoundError(NotFoundError):
    code = "KB_NOT_FOUND"
    default_message = "知识库不存在"


class DocumentNotFoundError(NotFoundError):
    code = "DOCUMENT_NOT_FOUND"
    default_message = "文档不存在"


class ConversationNotFoundError(NotFoundError):
    code = "CONVERSATION_NOT_FOUND"
    default_message = "会话不存在"


class RunNotFoundError(NotFoundError):
    code = "RUN_NOT_FOUND"
    default_message = "评测运行不存在"


class TraceNotFoundError(NotFoundError):
    code = "TRACE_NOT_FOUND"
    default_message = "Trace 不存在"


# --------------------------------------------------------------------------------------
# 409 冲突
# --------------------------------------------------------------------------------------
class ConflictError(KnowFlowError):
    code = "CONFLICT"
    http_status = 409
    default_message = "资源冲突"


class KBNameConflictError(ConflictError):
    code = "KB_NAME_CONFLICT"
    default_message = "同名知识库已存在"


class UsernameConflictError(ConflictError):
    code = "USERNAME_CONFLICT"
    default_message = "用户名已被占用"


class DocumentDuplicateError(ConflictError):
    code = "DOCUMENT_DUPLICATE"
    default_message = "该文档已存在于这个知识库中"


# --------------------------------------------------------------------------------------
# 429 / 5xx
# --------------------------------------------------------------------------------------
class RateLimitedError(KnowFlowError):
    code = "RATE_LIMITED"
    http_status = 429
    default_message = "请求过于频繁，请稍后再试"


class UpstreamError(KnowFlowError):
    code = "UPSTREAM_ERROR"
    http_status = 502
    default_message = "上游模型服务调用失败"


class NotReadyError(KnowFlowError):
    code = "NOT_READY"
    http_status = 503
    default_message = "服务尚未就绪"


class EmbeddingError(KnowFlowError):
    """向量化失败。放在 core 而不是 embeddings 包里，避免 core 反向依赖业务包。"""

    code = "EMBEDDING_ERROR"
    http_status = 502
    default_message = "文本向量化失败"


class VectorStoreError(KnowFlowError):
    code = "VECTOR_STORE_ERROR"
    http_status = 500
    default_message = "向量库操作失败"


__all__ = [
    "BadCredentialsError",
    "ConflictError",
    "ConversationNotFoundError",
    "DocumentDuplicateError",
    "DocumentEmptyError",
    "DocumentNotFoundError",
    "DocumentParseError",
    "EmbeddingError",
    "FileTooLargeError",
    "ForbiddenError",
    "KBNotFoundError",
    "KBNameConflictError",
    "KnowFlowError",
    "NotFoundError",
    "NotReadyError",
    "RateLimitedError",
    "RunNotFoundError",
    "TraceNotFoundError",
    "UnauthorizedError",
    "UnsupportedFileTypeError",
    "UpstreamError",
    "UsernameConflictError",
    "ValidationError",
    "VectorStoreError",
]
