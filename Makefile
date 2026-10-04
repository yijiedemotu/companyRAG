# =====================================================================================
# KnowFlow — 仓库级任务入口（Makefile）
#
#   make help          看全部任务
#   make check         一条命令跑完 ruff + mypy + pytest（本地和 CI 都用它）
#
# ⚠ Windows 上**默认没有 make**。本文件是给 Linux/macOS/WSL/Git-Bash 和 CI 用的；
#   Windows 原生 PowerShell 请用等价入口：
#       .\tasks.ps1 help
#       .\tasks.ps1 check
#   两个入口的目标名与行为**一一对应**（见 README「常用命令」表）。
#
# ⚠ 不要用 `make install` 装到系统 Python：所有任务都优先用仓库内的 .venv。
# =====================================================================================

# --------------------------------------------------------------------------------------
# 变量
# --------------------------------------------------------------------------------------
BACKEND   := backend
FRONTEND  := frontend
SRC       := $(BACKEND)/src
UVICORN   := uvicorn knowflow.main:app --host 127.0.0.1 --port 8000 --app-dir $(SRC)
PIP_INDEX := https://pypi.tuna.tsinghua.edu.cn/simple

# Python 解释器：Windows 在 .venv/Scripts/，POSIX 在 .venv/bin/。
# 用 $(OS) 判断：Windows 上 make（Git-Bash/MSYS）会设置 OS=Windows_NT。
ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python.exe
else
  PY := .venv/bin/python
endif

# 解释器不存在时给出人话提示，而不是一堆 "No such file or directory"
.PHONY: _pycheck
_pycheck:
	@test -x "$(PY)" || { \
	  echo "✗ 没找到虚拟环境解释器：$(PY)"; \
	  echo "  先跑：python -m venv .venv  然后：make install-dev"; \
	  exit 1; \
	}

# --------------------------------------------------------------------------------------
# 依赖安装
# --------------------------------------------------------------------------------------
.PHONY: install
install: ## 装后端运行期依赖（清华镜像）
	$(PY) -m pip install -r $(BACKEND)/requirements.txt -i $(PIP_INDEX)

.PHONY: install-dev
install-dev: ## 装后端开发依赖（pytest / ruff / mypy）
	$(PY) -m pip install -r $(BACKEND)/requirements-dev.txt -i $(PIP_INDEX)

.PHONY: frontend-install
frontend-install: ## 装前端依赖（pnpm）
	cd $(FRONTEND) && pnpm install

# --------------------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------------------
.PHONY: run
run: _pycheck ## 起后端（127.0.0.1:8000，不带 --reload）
	$(PY) -m $(UVICORN)

.PHONY: dev
dev: _pycheck ## 起后端开发模式（带 --reload，改代码自动重启）
	$(PY) -m $(UVICORN) --reload

.PHONY: frontend-dev
frontend-dev: ## 起前端开发服务器（127.0.0.1:5173，proxy /api → 8000）
	cd $(FRONTEND) && pnpm dev

.PHONY: frontend-build
frontend-build: ## 前端生产构建（含 vue-tsc 类型检查）
	cd $(FRONTEND) && pnpm build

# --------------------------------------------------------------------------------------
# 数据库迁移（Alembic）
# --------------------------------------------------------------------------------------
.PHONY: migrate
migrate: _pycheck ## 升级数据库到最新版本（alembic upgrade head）
	cd $(BACKEND) && ../$(PY) -m alembic upgrade head

.PHONY: revision
revision: _pycheck ## 生成新迁移（用法：make revision M="加 xxx 字段"）
	@test -n "$(M)" || { echo '用法：make revision M="迁移说明"'; exit 1; }
	cd $(BACKEND) && ../$(PY) -m alembic revision --autogenerate -m "$(M)"

.PHONY: downgrade
downgrade: _pycheck ## 回退一个版本（alembic downgrade -1）
	cd $(BACKEND) && ../$(PY) -m alembic downgrade -1

.PHONY: seed
seed: _pycheck ## 灌演示数据（admin/admin123 + 演示 KB + 6 份文档 + 43 条评测集）
	$(PY) $(BACKEND)/scripts/seed_demo.py

# --------------------------------------------------------------------------------------
# 质量
# --------------------------------------------------------------------------------------
.PHONY: test
test: _pycheck ## 只跑 pytest
	cd $(BACKEND) && ../$(PY) -m pytest

.PHONY: test-all
test-all: test smoke lint typecheck ## pytest + 冒烟 + ruff + mypy（全量）
	@echo "✓ test-all 全部通过"

.PHONY: cov
cov: _pycheck ## 覆盖率（需要 pytest-cov，没装会提示）
	cd $(BACKEND) && ../$(PY) -m pytest --cov=knowflow --cov-report=term-missing --cov-report=xml

.PHONY: lint
lint: _pycheck ## ruff check + ruff format --check（含 src / scripts / tests）
	$(PY) -m ruff check $(SRC) $(BACKEND)/scripts $(BACKEND)/tests
	$(PY) -m ruff format --check $(SRC) $(BACKEND)/scripts $(BACKEND)/tests

.PHONY: format
format: _pycheck ## ruff 自动修复 + 格式化（会改文件）
	$(PY) -m ruff check --fix $(SRC) $(BACKEND)/scripts $(BACKEND)/tests
	$(PY) -m ruff format $(SRC) $(BACKEND)/scripts $(BACKEND)/tests

.PHONY: typecheck
typecheck: _pycheck ## mypy strict（读 backend/pyproject.toml 的配置）
	$(PY) -m mypy $(SRC)/knowflow --ignore-missing-imports

.PHONY: check
check: lint typecheck test ## ★ ruff + mypy + pytest，一条命令跑完（CI 用）

# --------------------------------------------------------------------------------------
# 自检 / 评测 / 标定
# --------------------------------------------------------------------------------------
.PHONY: smoke
smoke: _pycheck ## 端到端链路自检（跑在 knowflow_test 库 + data/_smoke，不碰开发数据）
	$(PY) $(BACKEND)/scripts/smoke_pipeline.py

.PHONY: smoke-http
smoke-http: _pycheck ## HTTP 冒烟：真起 uvicorn 打全部端点 + 校验 SSE 帧格式与顺序
	$(PY) $(BACKEND)/scripts/smoke_http.py

.PHONY: audit
audit: _pycheck ## 鉴权审计：确认没有漏加鉴权的端点（本项目曾漏过 21 个）
	$(PY) $(BACKEND)/scripts/audit_auth.py

.PHONY: yaml
yaml: _pycheck ## 校验 docker-compose.yml 与 CI 配置的 YAML 语法
	$(PY) $(BACKEND)/scripts/check_yaml.py

.PHONY: contract
contract: _pycheck ## 前后端接口契约对齐：前端调的每个路径后端都得有（联调前必跑）
	$(PY) $(BACKEND)/scripts/check_api_contract.py

.PHONY: docs
docs: _pycheck ## 文档质量门：教程结构 / HTML 自包含与锚点 / 数字可追溯 / 路径存在
	$(PY) $(BACKEND)/scripts/check_docs.py

.PHONY: eval
eval: _pycheck ## 检索质量评测与消融实验（vector/bm25/hybrid/hybrid_rerank 四组对比）
	$(PY) $(BACKEND)/scripts/run_eval.py --modes vector,bm25,hybrid,hybrid_rerank

.PHONY: calibrate
calibrate: _pycheck ## 标定相关性闸门阈值（零成本：不连库、不调大模型）
	$(PY) $(BACKEND)/scripts/calibrate_threshold.py --out data/threshold_calibration.json

# --------------------------------------------------------------------------------------
# 清理
# --------------------------------------------------------------------------------------
.PHONY: clean
clean: ## 清掉缓存与构建产物（不动 data/ 与 .venv）
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf $(BACKEND)/*.egg-info $(FRONTEND)/dist
	@echo "✓ 已清理（data/ 与 .venv/ 未动）"

# --------------------------------------------------------------------------------------
# help：从本文件的 `##` 注释里自动提取，永不与实现脱节
# --------------------------------------------------------------------------------------
.PHONY: help
help: ## 显示本帮助
	@echo "KnowFlow 任务（Windows 请用 .\\tasks.ps1 <task>）"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "例：make check    # ruff + mypy + pytest"

.DEFAULT_GOAL := help
