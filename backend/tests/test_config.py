"""配置层测试。

保护三类承诺（每一条都是真实踩过的坑）：

1. **配置错在启动期就炸**，而不是第一次请求 500：``chunk_overlap >= chunk_size``、
   ``parent_chunk_size < chunk_size``、``rerank_top_n > fetch_k`` 都必须在构造期 raise。
2. **两种列表写法都要能解析**：``CORS_ORIGINS=a,b`` 与 ``CORS_ORIGINS=["a","b"]``。
   不加 ``NoDecode`` 时逗号写法会在启动时抛 ``JSONDecodeError``。
3. **不泄露密钥**：``OPENAI_API_KEY=" "``（一个空格）必须算离线；
   ``_mask_dsn`` / ``public_snapshot`` 里不能出现口令。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from knowflow.core.config import Settings, _mask_dsn, get_settings


def make_settings(**overrides: object) -> Settings:
    """构造配置：**不读 .env**，结论只由显式入参决定。"""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# 离线判定
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("api_key", "expected_offline"),
    [
        ("", True),  # 没配 -> 离线
        (" ", True),  # 一个空格（shell 里最常见的写法）-> 必须算离线
        ("\t\n", True),  # 只有空白字符
        ("sk-abc123", False),
        ("  sk-abc123  ", False),  # 两侧空格不代表没配
    ],
)
def test_只有空白字符的APIKey都算离线模式(api_key: str, expected_offline: bool) -> None:
    """``OPENAI_API_KEY=" "`` 不做 strip 会被误判成"在线"，然后在第一次提问时 401。"""
    config = make_settings(openai_api_key=api_key)
    assert config.is_offline is expected_offline
    assert config.llm_mode == ("offline" if expected_offline else "online")


def test_判官模型留空时复用主模型() -> None:
    config = make_settings(openai_model="deepseek-chat", llm_judge_model="  ")
    assert config.judge_model == "deepseek-chat"


# --------------------------------------------------------------------------------------
# 列表型配置：逗号 / JSON 两种写法
# --------------------------------------------------------------------------------------
def test_逗号写法的CORS种源能被环境变量解析(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    assert make_settings().cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_JSON写法的CORS种源能被环境变量解析(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["http://a.example"]')
    assert make_settings().cors_origins == ["http://a.example"]


def test_逗号写法与JSON写法解析结果一致(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a,http://b")
    comma = make_settings().cors_origins
    monkeypatch.setenv("CORS_ORIGINS", '["http://a", "http://b"]')
    as_json = make_settings().cors_origins
    assert comma == as_json == ["http://a", "http://b"]


def test_空的CORS配置解析成空列表而不是报错() -> None:
    assert make_settings(cors_origins="").cors_origins == []
    assert make_settings(cors_origins="   ").cors_origins == []


def test_扩展名白名单会去点去空格并小写排序(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "md, .TXT ,pdf,MD")
    assert make_settings().allowed_extensions == ["md", "pdf", "txt"]


# --------------------------------------------------------------------------------------
# 启动期交叉校验
# --------------------------------------------------------------------------------------
def test_chunk_overlap不小于chunk_size时启动期报错() -> None:
    """否则递归切分器无法收敛（会无限切分）。"""
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_parent_chunk_size小于chunk_size时启动期报错() -> None:
    """父块必须能容纳子块。"""
    with pytest.raises(ValidationError, match="PARENT_CHUNK_SIZE"):
        make_settings(chunk_size=800, parent_chunk_size=700)


def test_rerank_top_n大于fetch_k时启动期报错() -> None:
    """重排只能作用于已召回的候选。"""
    with pytest.raises(ValidationError, match="RERANK_TOP_N"):
        make_settings(fetch_k=10, rerank_top_n=20)


@pytest.mark.parametrize(
    "overrides",
    [
        {"env": "prod", "debug": False},  # 没改 JWT_SECRET
        {"env": "prod", "jwt_secret": "a-real-secret-value", "debug": True},  # 生产开 debug
    ],
)
def test_生产环境的危险配置在启动期报错(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_测试环境的默认密钥不会被拦下() -> None:
    """只有 prod 才拦：本地/CI 不该因为没配密钥就跑不起来。"""
    assert make_settings(env="test").jwt_secret == "change-me-in-production"


# --------------------------------------------------------------------------------------
# 派生属性与脱敏
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api/v1/", "/api/v1"),
        ("/api/v1", "/api/v1"),
        ("api/v1", "/api/v1"),
        ("/api/v1/", "/api/v1"),
        ("/", ""),
        ("", ""),
    ],
)
def test_api_prefix会被归一化(raw: str, expected: str) -> None:
    """路由前缀拼接不做归一化就会出现 ``//api/v1`` 这种 404 路由。"""
    assert make_settings(api_prefix=raw).api_prefix == expected


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        (
            "mysql+pymysql://root:1234@127.0.0.1:3306/knowflow?charset=utf8mb4",
            "mysql+pymysql://root:***@127.0.0.1:3306/knowflow?charset=utf8mb4",
        ),
        (
            "mysql+pymysql://root:1234@127.0.0.1:3306/knowflow",
            "mysql+pymysql://root:***@127.0.0.1:3306/knowflow",
        ),
        ("postgresql://host/db", "postgresql://host/db"),  # 没有口令 -> 原样返回
        ("not-a-dsn", "not-a-dsn"),
    ],
)
def test_掩码后的DSN不泄露口令(dsn: str, expected: str) -> None:
    masked = _mask_dsn(dsn)
    assert masked == expected
    assert "1234" not in masked


def test_健康快照里既没有APIKey也没有数据库口令() -> None:
    config = make_settings(
        openai_api_key="sk-super-secret-key",
        database_url="mysql+pymysql://root:1234@127.0.0.1:3306/knowflow",
    )
    snapshot = config.public_snapshot()
    dumped = repr(snapshot)
    assert "sk-super-secret-key" not in dumped
    assert "1234" not in dumped
    # 但离线/在线状态必须如实上报（不能因为脱敏把事实也藏掉）
    assert snapshot["offline"] is False
    assert snapshot["llm_mode"] == "online"


def test_派生路径都以data_dir为根(tmp_path: Path) -> None:
    config = make_settings(data_dir=tmp_path)
    assert config.upload_dir == tmp_path / "uploads"
    assert config.chroma_path == tmp_path / "chroma"
    assert config.checkpoint_path == tmp_path / "agent_checkpoints.sqlite"
    assert config.embedding_cache_dir == tmp_path / "model_cache"
    assert config.max_upload_bytes == config.max_upload_mb * 1024 * 1024


def test_ensure_dirs会创建运行期目录(tmp_path: Path) -> None:
    config = make_settings(data_dir=tmp_path / "data")
    created = config.ensure_dirs()
    assert set(created) == {config.data_dir, config.upload_dir, config.chroma_path}
    assert all(path.is_dir() for path in created)


# --------------------------------------------------------------------------------------
# 单例
# --------------------------------------------------------------------------------------
def test_get_settings是进程内单例且能通过cache_clear换配置() -> None:
    """``lru_cache`` 而不是模块级全局变量，就是为了测试能换一套配置重建容器。"""
    get_settings.cache_clear()
    first = get_settings()
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings() is not first
    get_settings.cache_clear()
