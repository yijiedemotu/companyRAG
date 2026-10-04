"""口令哈希与 JWT。

保护两条承诺：

1. **口令的 72 字节上限**：bcrypt 5.x 对超过 72 字节的输入**直接抛 ValueError**，
   一个用 emoji 或长中文口令的用户会在注册时拿到 500。这里必须走 `bcrypt-sha256` 兜底，
   而且 **hash 与 verify 两侧都要走**（只改一边会让老用户全部登录失败）。
2. **JWT 的失败必须收敛成 ``UnauthorizedError``**：过期 / 篡改 / 伪造 issuer / 缺必需声明
   都统一成一个错误码，不区分"签名对但过期了"这种会泄露信息的细节。
"""

from __future__ import annotations

import time

import jwt
import pytest

from knowflow.core.config import Settings
from knowflow.core.exceptions import UnauthorizedError
from knowflow.core.security import (
    _prepare_password,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

#: 32 字节以上，避免 pyjwt 的 InsecureKeyLengthWarning 混进测试输出
SECRET = "unit-test-secret-0123456789abcdef-0123456789"


@pytest.fixture
def security_settings() -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        jwt_secret=SECRET,
        app_name="KnowFlow",
        access_token_expire_minutes=30,
    )


def _forged(settings: Settings, **payload: object) -> str:
    base: dict[str, object] = {
        "sub": "1",
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "iss": settings.app_name,
    }
    base.update(payload)
    return jwt.encode(base, settings.jwt_secret, algorithm=settings.jwt_algorithm)


# --------------------------------------------------------------------------------------
# 口令
# --------------------------------------------------------------------------------------
def test_同一口令两次哈希不同且都能校验通过() -> None:
    """加盐生效：两次哈希不同是**正确行为**，相同才说明没加盐。"""
    first = hash_password("passw0rd")
    second = hash_password("passw0rd")
    assert first != second
    assert verify_password("passw0rd", first) is True
    assert verify_password("passw0rd", second) is True


def test_错误口令校验失败() -> None:
    assert verify_password("passw0rd", hash_password("passw0rd!")) is False


def test_非法哈希串返回False而不是抛异常() -> None:
    """校验失败的原因不应该泄露给调用方，更不该把 500 冒到注册/登录接口。"""
    assert verify_password("passw0rd", "not-a-bcrypt-hash") is False
    assert verify_password("passw0rd", "") is False


@pytest.mark.parametrize(
    ("label", "password"),
    [
        ("50 个汉字", "汉" * 50),  # 150 字节
        ("30 个 emoji", "🔒" * 30),  # 120 字节
        ("刚好 72 字节", "a" * 72),
        ("73 字节（越界一个）", "a" * 73),
        ("很长的英文口令", "x" * 400),
    ],
)
def test_超过72字节的口令也能哈希与校验(label: str, password: str) -> None:
    """bcrypt 5.x 对 >72 字节直接抛 ValueError，必须由 ``_prepare_password`` 兜住。"""
    digest = hash_password(password)
    assert digest.startswith("$2")
    assert verify_password(password, digest) is True


def test_超长口令的哈希与前缀标记一起工作() -> None:
    """长口令走 sha256+base64，并带固定前缀，避免与"恰好等于该 base64 的短口令"撞车。"""
    prepared = _prepare_password("汉" * 50)
    assert prepared.startswith(b"$bcrypt-sha256$")
    assert len(prepared) <= 72


def test_两个不同的超长口令不会互相通过() -> None:
    """只改 hash 不改 verify（或反之）最容易出的就是这种"谁的密码都能登"的错。"""
    digest = hash_password("汉" * 50)
    assert verify_password("汉" * 49, digest) is False
    assert verify_password("汉" * 51, digest) is False


def test_72字节边界内外的两个口令互不相同() -> None:
    """边界处不能因为截断而让两种口令等价。"""
    inside = hash_password("a" * 72)
    assert verify_password("a" * 71, inside) is False


# --------------------------------------------------------------------------------------
# JWT
# --------------------------------------------------------------------------------------
def test_能签发也能解析出同样的主体(security_settings: Settings) -> None:
    token, expires_in = create_access_token(
        subject=42, extra={"role": "admin"}, settings=security_settings
    )
    payload = decode_access_token(token, settings=security_settings)
    assert payload["sub"] == "42"
    assert payload["role"] == "admin"
    assert payload["iss"] == security_settings.app_name
    assert expires_in == 30 * 60


def test_过期token抛UnauthorizedError(security_settings: Settings) -> None:
    expired = _forged(security_settings, iat=int(time.time()) - 100, exp=int(time.time()) - 10)
    with pytest.raises(UnauthorizedError) as excinfo:
        decode_access_token(expired, settings=security_settings)
    assert excinfo.value.code == "UNAUTHORIZED"
    assert "过期" in excinfo.value.message


def test_篡改一个字符的token抛UnauthorizedError(security_settings: Settings) -> None:
    token, _ = create_access_token(subject=1, settings=security_settings)
    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
    with pytest.raises(UnauthorizedError):
        decode_access_token(tampered, settings=security_settings)


def test_伪造issuer的token被拒绝(security_settings: Settings) -> None:
    with pytest.raises(UnauthorizedError):
        decode_access_token(
            _forged(security_settings, iss="evil-issuer"), settings=security_settings
        )


def test_换密钥签发的token被拒绝(security_settings: Settings) -> None:
    other = security_settings.model_copy(update={"jwt_secret": "another-secret-" * 2})
    token, _ = create_access_token(subject=1, settings=other)
    with pytest.raises(UnauthorizedError):
        decode_access_token(token, settings=security_settings)


@pytest.mark.parametrize("missing", ["sub", "exp", "iat"])
def test_缺少必需声明的token被拒绝(security_settings: Settings, missing: str) -> None:
    """``options={"require": [...]}`` 保证残缺 token 不会当成合法身份。"""
    payload: dict[str, object] = {
        "sub": "1",
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "iss": security_settings.app_name,
    }
    payload.pop(missing)
    token = jwt.encode(payload, SECRET, algorithm=security_settings.jwt_algorithm)
    with pytest.raises(UnauthorizedError):
        decode_access_token(token, settings=security_settings)


def test_垃圾字符串被拒绝(security_settings: Settings) -> None:
    with pytest.raises(UnauthorizedError):
        decode_access_token("not.a.jwt", settings=security_settings)
