"""鉴权审计：确认每个业务接口都要求登录。

**为什么需要这个脚本**：这个漏洞是靠它发现的 —— 最初 39 个端点里有 **21 个漏了鉴权**
（`GET /kbs`、`GET /documents/{id}`、`GET /obs/traces` 等只读但含业务数据的接口）。
它们的行为是**返回 200 + 正常数据**，所以任何功能测试、任何断言"字段对不对"的测试
都发现不了；只有"**不带 token 打一次**"才会暴露。

**两个实现细节值得记下来**：

1. **必须从 `original_router` 递归**，不能直接遍历 `app.routes`。
   FastAPI 0.142 把 `include_router` 做成了**惰性**的：`app.routes` 里放的是
   `_IncludedRouter` 占位对象（`path` 和 `methods` 都是 None），
   真正的 `APIRoute` 藏在 `original_router.routes` 里，前缀在 `include_context.prefix`。
   第一版脚本直接遍历 `app.routes`，只找到 1 个端点，**差点得出"没有漏洞"的错误结论**。
2. **必须继承路由级依赖**。`include_router(..., dependencies=[Depends(get_current_user)])`
   这种"统一加鉴权"的写法不会出现在单个端点的 `dependant` 里，
   而是挂在 `include_context.dependencies` 上。只查端点自身的依赖会**误报**。

所以：既查端点自身依赖，也查从上层继承下来的依赖。

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe backend\\scripts\\audit_auth.py
退出码：0 = 全部合规；1 = 有未鉴权接口（可直接当 CI 门禁）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "src"))

# 刻意公开的运维探针：服务"半死不活"时也要能回答，拿不到 token 时正是需要看它的时候
PUBLIC_PATHS = {"/health", "/ready", "/metrics"}
# 刻意公开的业务接口：未登录时必须可用
PUBLIC_ROUTES = {("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/login")}

AUTH_CALLABLES = {"get_current_user", "get_admin_user"}


def _call_names(dependant: Any, seen: set[int] | None = None) -> set[str]:
    """递归收集一个 Dependant 子树里用到的所有依赖函数名。"""
    if dependant is None:
        return set()
    if seen is None:
        seen = set()
    if id(dependant) in seen:
        return set()
    seen.add(id(dependant))

    names: set[str] = set()
    call = getattr(dependant, "call", None)
    if call is not None:
        names.add(getattr(call, "__name__", str(call)))
    for sub in getattr(dependant, "dependencies", []) or []:
        names |= _call_names(sub, seen)
    return names


def _deps_from_include_context(context: Any) -> set[str]:
    """取出 `include_router(..., dependencies=[...])` 里的依赖函数名。"""
    names: set[str] = set()
    for dep in getattr(context, "dependencies", None) or []:
        target = getattr(dep, "dependency", None)
        if target is not None:
            names.add(getattr(target, "__name__", str(target)))
    return names


def walk_routes(routes: Any, prefix: str = "", inherited: frozenset[str] = frozenset()):
    """递归产出 `(完整路径, methods, 端点自带的依赖名, 继承的依赖名)`。"""
    for route in routes:
        name = type(route).__name__
        if name == "APIRoute":
            yield (
                prefix + route.path,
                sorted(getattr(route, "methods", None) or []),
                _call_names(getattr(route, "dependant", None)),
                inherited,
            )
        elif name == "_IncludedRouter":
            context = getattr(route, "include_context", None)
            sub_prefix = prefix + (getattr(context, "prefix", "") or "")
            sub_inherited = inherited | _deps_from_include_context(context)
            yield from walk_routes(
                route.original_router.routes, sub_prefix, frozenset(sub_inherited)
            )
        elif hasattr(route, "routes"):
            # 兜底：其它带子路由的容器（如 Mount）
            yield from walk_routes(route.routes, prefix, inherited)


def main() -> int:
    from knowflow.main import create_app

    app = create_app()

    protected: list[str] = []
    public: list[str] = []
    problems: list[str] = []

    print("=" * 78)
    print("接口鉴权审计")
    print("=" * 78)

    for path, methods, own_deps, inherited_deps in walk_routes(app.routes):
        if path in PUBLIC_PATHS:
            continue
        has_auth = bool(AUTH_CALLABLES & (own_deps | inherited_deps))
        for method in methods:
            if method in ("HEAD", "OPTIONS"):
                continue
            label = f"{method:6s} {path:44s}"
            if (method, path) in PUBLIC_ROUTES:
                public.append(label)
                print(f"  ○ {label} 刻意公开（登录/注册）")
            elif has_auth:
                protected.append(label)
                print(f"  ✓ {label} 已鉴权")
            else:
                problems.append(label)
                print(f"  ✗ {label} **缺少鉴权**")

    print()
    print(f"已鉴权端点 = {len(protected)}   刻意公开 = {len(public)}")
    if problems:
        print(f"\n未鉴权且不该公开的端点（{len(problems)} 个）：")
        for item in problems:
            print(f"  - {item}")
    print("\nRESULT:", "OK" if not problems else f"{len(problems)} 个端点缺少鉴权")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
