"""前后端接口契约对齐检查。

**为什么需要这个脚本**：前端是照着 `docs/01-数据库与接口契约.md` 写的，
后端是照着同一份契约实现的 —— 但**两边都"按文档做"依然会不一致**，
因为文档没有约束到"路径里参数叫什么名字"这种细节：

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    前端  GET /api/v1/conversations/${id}/messages
    后端  GET /api/v1/conversations/{conversation_id}/messages
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这在双方各自自测时**都是 200**（前端自测时后端还没写，后端自测时用的是 pytest 造的路径），
只有真正联调才会 404。所以把"前端调用的每个路径"与"后端 OpenAPI 里的每个路径"
做一次集合比对，是**联调前最便宜的一道保险**。

做三件事：
  1. 从 `frontend/src/**/*.ts` 里抠出所有 `/api/...`、`/health`、`/ready`、`/metrics` 形式的路径；
  2. 从后端真实 OpenAPI 里取出全部路径；
  3. 两边**把路径参数名归一**（`${id}` / `{conversation_id}` → `{}`）后比对，报告：
     - ❌ 前端调了但后端没有 → **真 bug，联调必 404**
     - ⚠ 后端有但前端没调 → 只是提示（可能是有意留的接口 / 前端还没做）

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\check_api_contract.py
退出码：0 = 前端调用的路径后端全都有；1 = 存在对不上的路径
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND / "src"))

# 抠字符串字面量：'...' / "..." / `...`
_STRING_LITERAL = re.compile(r"""(?P<q>['"`])(?P<path>/[^'"`\s]*)(?P=q)""")
# 模板参数 `${...}` 与占位 `{...}` 统一成 `{}`
_TEMPLATE_PARAM = re.compile(r"\$\{[^}]*\}")
_PATH_PARAM = re.compile(r"\{[^}/]*\}")

# 前端用相对路径（`baseURL = '/api/v1'`），所以第一段必然是这些之一
_RELATIVE_ROOTS = (
    "auth",
    "kbs",
    "documents",
    "chat",
    "conversations",
    "messages",
    "eval",
    "obs",
)
# 运维三件套在根路径，前端用的是绝对路径
_ABSOLUTE_ROOTS = ("/health", "/ready", "/metrics")

#: 前端 axios 实例的 baseURL（见 frontend/src/api/client.ts）。
#: 从源码里读而不是写死，是为了"改前缀时这个脚本不会悄悄失效"。
_BASE_PATTERN = re.compile(r"VITE_API_BASE_URL\s*\?\?\s*['\"]([^'\"]+)['\"]")

_FRONTEND_DIRS = ("src",)
_SCAN_SUFFIXES = (".ts", ".vue")


def read_frontend_base() -> str:
    client = REPO_ROOT / "frontend" / "src" / "api" / "client.ts"
    if client.exists():
        match = _BASE_PATTERN.search(client.read_text(encoding="utf-8"))
        if match:
            return match.group(1).rstrip("/")
    return "/api/v1"


def normalize(path: str) -> str:
    """把路径里的参数名抹掉，只比结构。

    `/{conversation_id}/messages` 与 `/1/messages` 都会归一成 `/{}/messages`，
    这样"参数叫什么名字"就不会造成假报警 —— 那本来就不影响路由匹配。
    """
    path = path.split("?")[0]  # 去掉 query
    path = path.rstrip("/") or "/"
    path = _TEMPLATE_PARAM.sub("{}", path)
    return _PATH_PARAM.sub("{}", path)


def collect_frontend_paths(base: str) -> dict[str, set[str]]:
    """`{归一化后的完整路径: {出现位置}}`。

    三处细节决定了这个脚本有没有用，每一处都是第一版踩出来的：

    1. **前端写的是相对路径**（`get('/kbs')` + `baseURL='/api/v1'`），
       所以要**自己把 base 拼上去**，否则永远匹配不上后端的 `/api/v1/kbs`。
       第一版漏了这一步，只找到 4 个路径、还报了两个假警。
    2. **只扫 API 层（`src/api/**`）**。全仓库扫会把**前端路由**当成接口路径：
       `router/index.ts` 里的 `/kbs/:id`、`/obs/traces/:traceId` 用的是
       Vue Router 的 `:param` 语法，和 HTTP 路径根本不是一个东西 ——
       全扫会多报 6 个假警，把真问题淹掉。
       （本项目所有接口 URL 都集中在 `src/api/` 下，这正是这样分层的好处。）
    3. **运维三件套在根路径**，前端用的是绝对 URL，且可能出现在任意组件里
       （例如顶栏显示 `/health` 的模式），所以对它们单独在整个 `src/` 里扫。
    """
    found: dict[str, set[str]] = {}
    frontend = REPO_ROOT / "frontend" / "src"

    # ---- (a) API 层：相对路径 ----
    for path in sorted((frontend / "api").rglob("*.ts")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _STRING_LITERAL.finditer(text):
            raw = match.group("path")
            first = raw.lstrip("/").split("/")[0]
            if first not in _RELATIVE_ROOTS:
                continue
            full = normalize(f"{base}{raw}")
            if full == normalize(base):
                continue
            found.setdefault(full, set()).add(str(path.relative_to(frontend.parent)))

    # ---- (b) 全 src：运维绝对路径 ----
    for path in sorted(frontend.rglob("*")):
        if path.suffix not in _SCAN_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _STRING_LITERAL.finditer(text):
            raw = match.group("path")
            if not raw.startswith(_ABSOLUTE_ROOTS):
                continue
            full = normalize(raw)
            found.setdefault(full, set()).add(str(path.relative_to(frontend.parent)))

    return found


def collect_backend_paths() -> dict[str, set[str]]:
    from knowflow.main import create_app

    app = create_app()
    spec = app.openapi()
    found: dict[str, set[str]] = {}
    for raw, operations in (spec.get("paths") or {}).items():
        methods = {
            m.upper() for m in operations if m.lower() in {"get", "post", "put", "patch", "delete"}
        }
        found[normalize(raw)] = methods
    return found


def main() -> int:
    base = read_frontend_base()
    frontend = collect_frontend_paths(base)
    backend = collect_backend_paths()

    print("=" * 78)
    print("前后端接口契约对齐检查")
    print("=" * 78)
    print(f"  前端 baseURL   : {base}")
    print(f"  前端调用的路径 : {len(frontend)} 个")
    print(f"  后端暴露的路径 : {len(backend)} 个")

    missing = sorted(set(frontend) - set(backend))
    unused = sorted(set(backend) - set(frontend))

    print("\n--- 前端调用 & 后端存在 ---")
    for path in sorted(set(frontend) & set(backend)):
        print(f"  ✓ {path:52s} {sorted(backend[path])}")

    if missing:
        print(f"\n--- ❌ 前端调了但后端没有（{len(missing)} 个，联调必 404）---")
        for path in missing:
            where = ", ".join(sorted(frontend[path]))
            print(f"  ✗ {path:52s} ← {where}")
    else:
        print("\n--- ✅ 前端调用的每个路径后端都有 ---")

    if unused:
        print(f"\n--- ⚠ 后端有但前端没调（{len(unused)} 个，仅提示）---")
        for path in unused:
            print(f"  · {path:52s} {sorted(backend[path])}")

    print("\nRESULT:", "OK" if not missing else f"{len(missing)} 个路径对不上")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
