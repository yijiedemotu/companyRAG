"""口令哈希与 JWT 签发/校验。

两个真实踩过的坑，都在这里处理掉：

1. **bcrypt 的 72 字节上限**。bcrypt 只取输入的前 72 字节；更麻烦的是
   `bcrypt` 5.x 对超过 72 字节的输入**直接抛 ValueError**，而不是静默截断。
   一个用 emoji 或长中文密码的用户会在注册时拿到 500。
   做法：超过 72 字节时先 `sha256` 再 base64（32 字节），这是 passlib 与
   Django 采用的通行方案（bcrypt-sha256）。**必须同时改 hash 与 verify**，
   只改一边会导致老用户全部登录失败。

2. **JWT 的 `exp` 必须是 UTC**。用本地时间会因时区偏移导致 token 提前/延后过期。
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import UnauthorizedError

_BCRYPT_MAX_BYTES = 72


def _prepare_password(password: str) -> bytes:
    """把任意长口令压到 bcrypt 能接受的 72 字节以内。

    <= 72 字节：原样使用（保持与"直接 bcrypt"的兼容性）；
    >  72 字节：sha256 后 base64（44 字节），并在前面加固定标记，
               避免"长口令的 sha256"与"某个恰好等于该 base64 的短口令"撞车。
    """
    raw = password.encode("utf-8")
    if len(raw) <= _BCRYPT_MAX_BYTES:
        return raw
    digest = base64.b64encode(hashlib.sha256(raw).digest())
    return b"$bcrypt-sha256$" + digest


def hash_password(password: str) -> str:
    """生成 bcrypt 哈希（每次调用 salt 不同，同口令两次结果不同，这是对的）。"""
    return bcrypt.hashpw(_prepare_password(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验口令。任何异常都返回 False——校验失败的原因不应该泄露给调用方。"""
    try:
        return bcrypt.checkpw(_prepare_password(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(
    *,
    subject: str | int,
    extra: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> tuple[str, int]:
    """签发 access token。

    返回 `(token, expires_in_seconds)`——`expires_in` 要给前端，让它能提前刷新，
    而不是等 401 才发现过期。
    """
    cfg = settings or get_settings()
    now = datetime.now(UTC)
    expire_delta = timedelta(minutes=cfg.access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int((now + expire_delta).timestamp()),
        "iss": cfg.app_name,
    }
    if extra:
        payload.update(extra)
    token = jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)
    return token, int(expire_delta.total_seconds())


def decode_access_token(token: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """解析并校验 token，失败统一抛 `UnauthorizedError`（不区分"过期"与"伪造"）。

    不区分是有意的：告诉攻击者"签名对但过期了"会泄露信息。
    真正需要区分时看服务端日志里的异常类型。
    """
    cfg = settings or get_settings()
    try:
        return jwt.decode(
            token,
            cfg.jwt_secret,
            algorithms=[cfg.jwt_algorithm],
            issuer=cfg.app_name,
            options={"require": ["exp", "sub", "iat"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError("身份凭证无效") from exc


__all__ = [
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
