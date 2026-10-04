"""认证路由（契约 5.1）。

**一句话定位**：注册 / 登录 / 当前用户三个接口，是其余所有接口的前置条件。

**在链路中的位置**：前端登录页 -> **本模块** -> `AuthService` -> `users` 表 + JWT。

**关键设计取舍**：

1. **注册返回 201 且直接带 token**。契约写的是 `201 TokenOut` —— 注册完不用再登录一次，
   少一次往返，也少一次"注册成功但登录失败"的中间态（密码已经在请求体里了）。
2. **登录失败不区分"用户不存在"与"密码错误"**（`AuthService.login` 统一抛
   `BAD_CREDENTIALS`）。在路由层不要为了"更友好的提示"把它拆开，
   那会把登录接口变成一个用户名枚举器。
3. **`/auth/me` 只回 `UserOut`**：响应模型里没有 `password_hash` 字段，
   所以任何粗心的 `model_validate(user)` 都泄漏不出去。
"""

from __future__ import annotations

from fastapi import APIRouter, status

from knowflow.api.deps import CurrentUser, Services, SettingsDep
from knowflow.schemas.auth import LoginRequest, RegisterRequest, TokenOut, UserOut
from knowflow.services.auth import AuthService

__all__ = ["router"]

router = APIRouter(prefix="/auth", tags=["认证"])


def _token_out(*, user: object, token: str, expires_in: int) -> TokenOut:
    """把 `(User, token, expires_in)` 组装成契约 5.1 的 `TokenOut`。

    用 `model_validate` + dict 而不是逐字段构造：新增用户字段时这里不用改，
    而且 `UserOut` 会把多余的字段（如 `password_hash`）直接丢掉——
    **响应模型就是那道防泄漏闸门**。
    """
    payload = {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "user": UserOut.model_validate(user),
    }
    return TokenOut.model_validate(payload)


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    summary="注册并直接登录",
)
def register(payload: RegisterRequest, services: Services, settings: SettingsDep) -> TokenOut:
    """注册。**第一个注册的用户自动成为 admin**（方便本地演示与 CI，见 `AuthService`）。"""
    user, token, expires_in = AuthService(session=services.session, settings=settings).register(
        username=payload.username,
        password=payload.password,
        display_name=payload.display_name,
    )
    services.session.commit()
    return _token_out(user=user, token=token, expires_in=expires_in)


@router.post("/login", response_model=TokenOut, summary="登录")
def login(payload: LoginRequest, services: Services, settings: SettingsDep) -> TokenOut:
    user, token, expires_in = AuthService(session=services.session, settings=settings).login(
        username=payload.username, password=payload.password
    )
    return _token_out(user=user, token=token, expires_in=expires_in)


@router.get("/me", response_model=UserOut, summary="当前登录用户")
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
