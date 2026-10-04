"""用户与认证服务。

**"第一个注册的用户自动成为 admin"** 是刻意的：本地演示与 CI 都需要一个
管理员，如果要求"先手动改数据库"，那就没人跑得起来。
生产环境应该关掉这个行为（`ENV=prod` 时只允许第一个用户注册，之后需要邀请）。

安全细节：
- 登录失败时**不区分**"用户不存在"与"密码错误"（都返回 `BAD_CREDENTIALS`），
  否则接口变成用户名枚举器；
- 被禁用的账号返回同样的错误，不透露"这个账号存在但被禁用"。
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from knowflow.core.config import Settings, get_settings
from knowflow.core.exceptions import (
    BadCredentialsError,
    ForbiddenError,
    UnauthorizedError,
    UsernameConflictError,
    ValidationError,
)
from knowflow.core.logging import get_logger
from knowflow.core.security import create_access_token, hash_password, verify_password
from knowflow.db.models.identity import ROLE_ADMIN, ROLE_USER, User

logger = get_logger(__name__)


class AuthService:
    def __init__(self, *, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    def register(
        self, *, username: str, password: str, display_name: str | None = None
    ) -> tuple[User, str, int]:
        clean = username.strip()
        if not clean:
            raise ValidationError("用户名不能为空")
        if len(password) < 6:
            raise ValidationError("密码至少 6 位")

        existing = self.session.execute(select(User.id).where(User.username == clean)).scalar()
        if existing is not None:
            raise UsernameConflictError(f"用户名 {clean!r} 已被占用")

        user_count = int(self.session.execute(select(func.count(User.id))).scalar() or 0)
        role = ROLE_ADMIN if user_count == 0 else ROLE_USER

        user = User(
            username=clean,
            password_hash=hash_password(password),
            display_name=display_name,
            role=role,
        )
        self.session.add(user)
        try:
            self.session.flush()
        except IntegrityError as exc:
            self.session.rollback()
            raise UsernameConflictError(f"用户名 {clean!r} 已被占用") from exc

        token, expires_in = create_access_token(
            subject=user.id,
            extra={"username": user.username, "role": user.role},
            settings=self.settings,
        )
        logger.info("auth.registered", user_id=user.id, username=user.username, role=user.role)
        return user, token, expires_in

    def login(self, *, username: str, password: str) -> tuple[User, str, int]:
        user = self.session.execute(
            select(User).where(User.username == username.strip())
        ).scalar_one_or_none()

        # 先算一次哈希校验，再判 user 是否为 None —— 保持两条路径耗时接近，
        # 避免"用户名不存在"明显更快而被用来做用户名枚举（时序侧信道）。
        password_hash = user.password_hash if user is not None else "$2b$12$" + "x" * 53
        ok = verify_password(password, password_hash)

        if user is None or not ok:
            logger.info("auth.login_failed", username=username[:32])
            raise BadCredentialsError("用户名或密码错误")
        if not user.is_active:
            logger.info("auth.login_disabled", user_id=user.id)
            raise BadCredentialsError("用户名或密码错误")

        token, expires_in = create_access_token(
            subject=user.id,
            extra={"username": user.username, "role": user.role},
            settings=self.settings,
        )
        logger.info("auth.login_ok", user_id=user.id, username=user.username)
        return user, token, expires_in

    def get_active_user(self, user_id: int) -> User:
        user = self.session.get(User, user_id)
        if user is None:
            raise UnauthorizedError("用户不存在")
        if not user.is_active:
            raise ForbiddenError("账号已被禁用")
        return user

    def ensure_admin(self, user: User) -> None:
        if not user.is_admin:
            raise ForbiddenError("需要管理员权限")


__all__ = ["AuthService"]
