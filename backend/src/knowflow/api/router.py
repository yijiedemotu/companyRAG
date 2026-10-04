"""路由汇总：把各分组的子路由挂到 `settings.api_prefix` 下。

**一句话定位**：唯一的"接口清单"落点 —— 想确认某个路径存不存在，只看这一个文件。

**在链路中的位置**：`main.create_app()` -> **本模块** -> `api/routes/*.py`。

**关键设计取舍**：

1. **子路由自己带 `prefix` + `tags`**（如 `/kbs`、`tags=["知识库"]`），
   本文件只做"挂载"而不再拼一遍路径。
   两边都拼一次的话，改路径要改两个地方，而且很容易出现
   `/kbs/kbs` 这种拼重复的低级错误（尤其在赶时间的时候）。

2. **鉴权走"默认需要、例外公开"，而不是"默认公开、逐个声明"**。
   这一条是**修出来的**，不是一开始就这么写的 —— 值得记下来：

   最初的设计是"逐个路由用 `CurrentUser` 声明"（理由是"读起来直观"）。
   后来写了个审计脚本扫全部端点，结果 **39 个端点里有 21 个漏了鉴权**，
   包括 `GET /kbs`、`GET /documents/{id}`、`GET /obs/traces` 这些**只读但含业务数据**的接口。
   它们的表现是**返回 200 + 正常数据**，所以任何功能测试都发现不了；
   只有"不带 token 打一次"才暴露。

   **教训**：安全属性不能靠"每个人记得写"，必须靠"默认安全"。
   所以现在改成：`AUTH_REQUIRED` 作为**路由级依赖**统一挂在业务路由上，
   任何**将来新增**的端点都会自动受保护；确实需要公开的（注册/登录、运维探针）
   才显式排除。默认值的方向决定了犯错时的后果。

3. **运维三件套（`/health` `/ready` `/metrics`）在根路径且不鉴权**，
   由 `include_ops()` 单独挂载 —— 它们的挂载位置与业务路由**故意不同**，
   放一起很容易在下次重构时"顺手"挪进 `/api/v1`，那会让所有探针地址失效。

4. 返回的是没有前缀的 `api_router`，前缀由 `main.create_app()` 用
   `settings.api_prefix` 挂载 —— 这样改 `API_PREFIX` 环境变量就能整体挪动，
   不需要动任何路由代码。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI

from knowflow.api.deps import get_current_user
from knowflow.api.routes import (
    auth,
    chat,
    conversations,
    documents,
    evaluation,
    knowledge_bases,
    observability,
    search,
)

__all__ = ["AUTH_REQUIRED", "PUBLIC_ROUTERS", "api_router", "include_ops", "router_auth_map"]

#: 挂在业务路由上的**路由级鉴权依赖**。
#: 单个端点再声明 `user: CurrentUser` 也不会重复执行 —— FastAPI 对同一依赖
#: 在同一请求内会缓存结果。
AUTH_REQUIRED = [Depends(get_current_user)]

#: 刻意公开的 `auth` 子路由：`/auth/register` 与 `/auth/login` 必须在未登录时可用，
#: 而 `/auth/me` 自己声明了 `CurrentUser`。所以整个 auth 路由**不能**加全局鉴权。
PUBLIC_ROUTERS = ("auth",)

#: 每个业务子路由是否要求鉴权。审计脚本（`scripts/_audit_auth.py`）读这张表，
#: 所以它既是配置也是"安全文档"。
router_auth_map: dict[str, bool] = {
    "auth": False,
    "knowledge_bases": True,
    "search": True,
    "documents": True,
    "chat": True,
    "conversations": True,
    "evaluation": True,
    "observability": True,
}

#: 业务路由集合（全部在 `settings.api_prefix` 之下）。
api_router = APIRouter()

api_router.include_router(auth.router)  # 公开：登录/注册；/auth/me 自行声明鉴权

# 其余业务路由统一要求登录（secure by default）。
api_router.include_router(knowledge_bases.router, dependencies=AUTH_REQUIRED)
# 检索调试必须排在文档路由**之前**：`/kbs/{kb_id}/search` 与 `/kbs/{kb_id}/documents`
# 同级，先注册的优先匹配。这里没有路径冲突（字面量段不同），
# 但顺序保持一致能让将来加 `/kbs/{kb_id}/{something}` 时不至于静默被吃掉。
api_router.include_router(search.router, dependencies=AUTH_REQUIRED)
api_router.include_router(documents.router, dependencies=AUTH_REQUIRED)
api_router.include_router(chat.router, dependencies=AUTH_REQUIRED)
api_router.include_router(conversations.router, dependencies=AUTH_REQUIRED)
api_router.include_router(evaluation.router, dependencies=AUTH_REQUIRED)
api_router.include_router(observability.router, dependencies=AUTH_REQUIRED)

# 断言：router_auth_map 必须覆盖上面挂载的每一个子路由，
# 否则"加了新路由但忘了在 map 里登记"就会让审计脚本给出假的安全结论。
_REGISTERED = (
    "auth",
    "knowledge_bases",
    "search",
    "documents",
    "chat",
    "conversations",
    "evaluation",
    "observability",
)
assert set(router_auth_map) == set(_REGISTERED), (
    f"router_auth_map 与已挂载子路由不一致："
    f"缺少 {set(_REGISTERED) - set(router_auth_map)}，多余 {set(router_auth_map) - set(_REGISTERED)}"
)


def include_ops(app: FastAPI) -> None:
    """把运维三件套挂到**根路径**（不带 `api_prefix`、不鉴权）。

    它们必须永远能回答 —— 探针在服务"半死不活"时也要拿到状态，
    所以不能要求 token（拿不到 token 的时候往往正是需要看 `/health` 的时候）。

    参数类型是 `FastAPI` 而不是 `APIRouter`：调用方传的是 app 实例。
    写错类型注解不会在运行期报错（Python 不检查），但会让 `mypy` 直接失败 ——
    这也说明类型检查确实抓到了"注解与用法不一致"这种真问题。
    """
    from knowflow.api.routes import ops

    app.include_router(ops.router)
