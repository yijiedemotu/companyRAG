"""认证契约：注册 / 登录 / 当前用户。

字段来源：契约文档 5.1 节。

两条容易漏的规则（都已经在这里用校验器钉住，service 层不用重复写）：

- `username` 3–32 位 `[A-Za-z0-9_]`。这个正则**同时是数据库约束的应用层表达**：
  库里只有 `UNIQUE`，没有字符集约束，所以这里是唯一的拦截点。
- `password` ≥ 6 位。上限 128 是给 bcrypt 的：bcrypt 只取前 72 字节，
  超长口令在语义上会被静默截断，这里用 128 的上限明确挡住"用户以为设了长密码"
  的误解（`Field(max_length=128)` 在 4000 字符的请求上也能省掉一次哈希计算）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from knowflow.db.models import ROLE_ADMIN, ROLE_USER
from knowflow.schemas.common import UtcDatetime

__all__ = [
    "LoginRequest",
    "RegisterRequest",
    "TokenOut",
    "UserOut",
]

# 角色取值。定义为 tuple 而不是在这里重写 `Literal["admin", "user"]`，
# 因为 role 的"真相来源"是 ORM 模块里的常量（`ROLE_ADMIN` / `ROLE_USER`）。
ACCOUNT_ROLES: tuple[str, ...] = (ROLE_USER, ROLE_ADMIN)


class RegisterRequest(BaseModel):
    """注册请求体。"""

    model_config = ConfigDict(from_attributes=True)

    username: str = Field(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9_]{3,32}$",
        description="用户名，3–32 位，只允许字母、数字与下划线",
    )
    password: str = Field(
        min_length=6,
        max_length=128,
        description="密码，至少 6 位",
    )
    display_name: str | None = Field(
        default=None,
        max_length=64,
        description="昵称（可选），留空则前端展示用户名",
    )


class LoginRequest(BaseModel):
    """登录请求体。这里不加格式校验：登录失败必须统一返回 401 BAD_CREDENTIALS，
    如果在 schema 层因格式不符直接 422，就等于告诉攻击者"这个用户名格式不存在"。
    """

    model_config = ConfigDict(from_attributes=True)

    username: str = Field(min_length=1, max_length=64, description="用户名")
    password: str = Field(min_length=1, max_length=128, description="密码")


class UserOut(BaseModel):
    """用户信息（契约 5.1 节的 `UserOut`）。

    **绝不包含 `password_hash`**：响应模型就是一道"防泄漏闸门"，
    只要它不在字段列表里，任何粗心的 `model_validate(user)` 都泄漏不出去。
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="用户 ID")
    username: str = Field(description="用户名")
    display_name: str | None = Field(default=None, description="昵称，可能为空")
    role: str = Field(description="角色：admin 或 user")
    is_active: bool = Field(description="账号是否启用")
    created_at: UtcDatetime = Field(default=None, description="注册时间，UTC ISO8601 带 Z")


class TokenOut(BaseModel):
    """登录/注册成功的返回。契约 5.1 节：`{access_token, token_type, expires_in, user}`。"""

    model_config = ConfigDict(from_attributes=True)

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="令牌类型，固定 bearer")
    expires_in: int = Field(
        description="有效期，单位**秒**（注意不是分钟；由 ACCESS_TOKEN_EXPIRE_MINUTES*60 得到）",
        ge=1,
    )
    user: UserOut = Field(description="当前登录用户")
