<#
=======================================================================================
 KnowFlow — 仓库级任务入口（Windows / PowerShell 版）

   与根目录 Makefile 的**目标名与行为一一对应**，因为 Windows 上默认没有 make。

       .\tasks.ps1 help        列出全部任务
       .\tasks.ps1 check       一条命令跑完 ruff + mypy + pytest
       .\tasks.ps1 test        只跑 pytest
       .\tasks.ps1 typecheck   只跑 mypy strict

   为什么不用 `make`：Windows 原生环境没有 make；即便装了 Git-Bash 的 make，
   它对 .venv/Scripts/*.exe 与中文输出路径的处理也常出岔子。
   两个入口都保留，是为了让 Linux/CI（make）和本机 Windows（tasks.ps1）各用顺手的那个。

   退出码约定：子任务失败 → 脚本以非 0 退出，方便接进 CI 或 pre-commit。
=======================================================================================
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task = 'help',

    # `revision` 用：迁移说明（对应 Makefile 的 M="..."）
    [Parameter(Position = 1)]
    [string]$Message = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------------------
# 路径与解释器：脚本位置决定仓库根，不依赖"当前工作目录"
# ---------------------------------------------------------------------------------------
$RepoRoot = $PSScriptRoot
$Backend  = Join-Path $RepoRoot 'backend'
$Frontend = Join-Path $RepoRoot 'frontend'
$Src      = Join-Path $Backend  'src'

# 解释器：优先 .venv（Windows 用 Scripts\python.exe），退回到系统 python。
$VenvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) { $VenvPy = Join-Path $RepoRoot '.venv/bin/python' }
if (Test-Path $VenvPy) {
    $Py = $VenvPy
} else {
    $Py = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $Py) {
    Write-Host 'x 没找到 Python。先建虚拟环境：python -m venv .venv' -ForegroundColor Red
    exit 1
}

$PipIndex = 'https://pypi.tuna.tsinghua.edu.cn/simple'

# ---------------------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------------------
function Write-Step([string]$Text) {
    Write-Host "`n>>> $Text" -ForegroundColor Cyan
}

function Write-Ok([string]$Text) {
    Write-Host "OK  $Text" -ForegroundColor Green
}

# 运行一条命令并按需断言退出码（PowerShell 不会因为原生命令失败而抛异常）
function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(ValueFromRemainingArguments)][string[]]$Args,
        [string]$WorkDir = $RepoRoot
    )
    Write-Host "    $Exe $($Args -join ' ')" -ForegroundColor DarkGray
    Push-Location $WorkDir
    try {
        & $Exe @Args
        if ($LASTEXITCODE -ne 0) {
            throw "命令失败（退出码 $LASTEXITCODE）：$Exe $($Args -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

# 在 backend/ 目录里跑 python -m <mod> ...
function Invoke-BackendPy {
    param(
        [Parameter(Mandatory)][string[]]$PyArgs,
        [string]$WorkDir = $Backend
    )
    Invoke-Step -Exe $Py -Args $PyArgs -WorkDir $WorkDir
}

function Get-HelpText {
    $lines = Get-Content -LiteralPath $PSCommandPath -Encoding UTF8
    $rows  = foreach ($line in $lines) {
        if ($line -match "^\s{4}'([a-z][a-z0-9-]*)'\s*\{\s*##\s*(.+?)\s*$") {
            [pscustomobject]@{ Task = $matches[1]; Desc = $matches[2] }
        }
    }
    Write-Host ''
    Write-Host 'KnowFlow 任务（等价入口：make <task>）' -ForegroundColor White
    Write-Host ''
    foreach ($row in $rows) {
        Write-Host ('  {0,-18}' -f $row.Task) -ForegroundColor Cyan -NoNewline
        Write-Host $row.Desc
    }
    Write-Host ''
    Write-Host '例：.\tasks.ps1 check    # ruff + mypy + pytest' -ForegroundColor DarkGray
    Write-Host ''
}

Write-Verbose "repo=$RepoRoot"
Write-Verbose "python=$Py"

# ---------------------------------------------------------------------------------------
# 任务分发（目标名与 Makefile 完全一致）
# ---------------------------------------------------------------------------------------
switch ($Task.ToLowerInvariant()) {

    'help' {  ## 显示本帮助
        Get-HelpText
    }

    # ---------------- 依赖安装 ----------------
    'install' {  ## 装后端运行期依赖（清华镜像）
        Invoke-Step -Exe $Py -Args @('-m', 'pip', 'install', '-r', (Join-Path $Backend 'requirements.txt'), '-i', $PipIndex)
        Write-Ok '运行期依赖已安装'
    }

    'install-dev' {  ## 装后端开发依赖（pytest / ruff / mypy）
        Invoke-Step -Exe $Py -Args @('-m', 'pip', 'install', '-r', (Join-Path $Backend 'requirements-dev.txt'), '-i', $PipIndex)
        Write-Ok '开发依赖已安装'
    }

    'frontend-install' {  ## 装前端依赖（pnpm）
        Invoke-Step -Exe 'pnpm' -Args @('install') -WorkDir $Frontend
        Write-Ok '前端依赖已安装'
    }

    # ---------------- 运行 ----------------
    'run' {  ## 起后端（127.0.0.1:8000，不带 --reload）
        Invoke-Step -Exe $Py -Args @('-m', 'uvicorn', 'knowflow.main:app',
            '--host', '127.0.0.1', '--port', '8000', '--app-dir', $Src)
    }

    'dev' {  ## 起后端开发模式（带 --reload）
        Invoke-Step -Exe $Py -Args @('-m', 'uvicorn', 'knowflow.main:app',
            '--host', '127.0.0.1', '--port', '8000', '--app-dir', $Src, '--reload')
    }

    'frontend-dev' {  ## 起前端开发服务器（127.0.0.1:5173）
        Invoke-Step -Exe 'pnpm' -Args @('dev') -WorkDir $Frontend
    }

    'frontend-build' {  ## 前端生产构建（含 vue-tsc 类型检查）
        Invoke-Step -Exe 'pnpm' -Args @('build') -WorkDir $Frontend
        Write-Ok '前端构建完成（frontend/dist）'
    }

    # ---------------- Alembic 迁移 ----------------
    'migrate' {  ## 升级数据库到最新版本（alembic upgrade head）
        Invoke-BackendPy -PyArgs @('-m', 'alembic', 'upgrade', 'head')
        Write-Ok '数据库已升级到 head'
    }

    'revision' {  ## 生成新迁移（.\tasks.ps1 revision "加 xxx 字段"）
        if ([string]::IsNullOrWhiteSpace($Message)) {
            Write-Host '用法：.\tasks.ps1 revision "迁移说明"' -ForegroundColor Yellow
            exit 1
        }
        Invoke-BackendPy -PyArgs @('-m', 'alembic', 'revision', '--autogenerate', '-m', $Message)
        Write-Host '提示：生成后请打开文件补全 downgrade()，并确认 alembic.ini 仍是纯 ASCII' -ForegroundColor Yellow
    }

    'downgrade' {  ## 回退一个版本（alembic downgrade -1）
        Invoke-BackendPy -PyArgs @('-m', 'alembic', 'downgrade', '-1')
        Write-Ok '已回退一个版本'
    }

    'seed' {  ## 灌演示数据（admin/admin123 + 演示 KB + 6 份文档 + 43 条评测集）
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\seed_demo.py'))
        Write-Ok '演示数据就绪'
    }

    # ---------------- 质量 ----------------
    'test' {  ## 只跑 pytest
        Invoke-BackendPy -PyArgs @('-m', 'pytest')
    }

    'test-all' {  ## pytest + 冒烟 + ruff + mypy（全量）
        Invoke-BackendPy -PyArgs @('-m', 'pytest')
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\smoke_pipeline.py'))
        & $PSCommandPath lint
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PSCommandPath typecheck
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Ok 'test-all 全部通过'
    }

    'cov' {  ## 覆盖率（需要 pytest-cov）
        Invoke-BackendPy -PyArgs @('-m', 'pytest', '--cov=knowflow',
            '--cov-report=term-missing', '--cov-report=xml')
    }

    'lint' {  ## ruff check + ruff format --check（含 src / scripts / tests）
        Invoke-Step -Exe $Py -Args @('-m', 'ruff', 'check', $Src, (Join-Path $Backend 'scripts'), (Join-Path $Backend 'tests'))
        Invoke-Step -Exe $Py -Args @('-m', 'ruff', 'format', '--check', $Src, (Join-Path $Backend 'scripts'), (Join-Path $Backend 'tests'))
        Write-Ok 'ruff 通过'
    }

    'format' {  ## ruff 自动修复 + 格式化（会改文件）
        Invoke-Step -Exe $Py -Args @('-m', 'ruff', 'check', '--fix', $Src, (Join-Path $Backend 'scripts'), (Join-Path $Backend 'tests'))
        Invoke-Step -Exe $Py -Args @('-m', 'ruff', 'format', $Src, (Join-Path $Backend 'scripts'), (Join-Path $Backend 'tests'))
        Write-Ok '已格式化'
    }

    'typecheck' {  ## mypy strict（读 backend/pyproject.toml 的配置）
        Invoke-Step -Exe $Py -Args @('-m', 'mypy', (Join-Path $Src 'knowflow'), '--ignore-missing-imports')
        Write-Ok 'mypy 通过'
    }

    'check' {  ## ★ ruff + mypy + pytest，一条命令跑完（CI 用）
        Write-Step 'check = lint + typecheck + test'
        & $PSCommandPath lint
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PSCommandPath typecheck
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $PSCommandPath test
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Ok 'check 全部通过（ruff + mypy + pytest）'
    }

    # ---------------- 自检 / 评测 / 标定 ----------------
    'smoke' {  ## 端到端链路自检（跑在 knowflow_test 库 + data/_smoke，不碰开发数据）
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\smoke_pipeline.py'))
    }

    'smoke-http' {  ## HTTP 冒烟：真起 uvicorn 打全部端点 + 校验 SSE 帧格式与顺序
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\smoke_http.py'))
    }

    'audit' {  ## 鉴权审计：确认没有漏加鉴权的端点（本项目曾漏过 21 个）
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\audit_auth.py'))
    }

    'yaml' {  ## 校验 docker-compose.yml 与 CI 配置的 YAML 语法
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\check_yaml.py'))
    }

    'contract' {  ## 前后端接口契约对齐：前端调的每个路径后端都得有（联调前必跑）
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\check_api_contract.py'))
    }

    'docs' {  ## 文档质量门：教程结构 / HTML 自包含与锚点 / 数字可追溯 / 路径存在
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\check_docs.py'))
    }

    'eval' {  ## 检索质量评测与消融实验（四种检索模式对比）
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\run_eval.py'),
            '--modes', 'vector,bm25,hybrid,hybrid_rerank')
    }

    'calibrate' {  ## 标定相关性闸门阈值（零成本：不连库、不调大模型）
        Invoke-Step -Exe $Py -Args @((Join-Path $Backend 'scripts\calibrate_threshold.py'),
            '--out', 'data/threshold_calibration.json')
    }

    # ---------------- 清理 ----------------
    'clean' {  ## 清掉缓存与构建产物（不动 data/ 与 .venv）
        $targets = @('.pytest_cache', '.mypy_cache', '.ruff_cache', '.coverage', 'htmlcov',
                     (Join-Path $Backend 'knowflow.egg-info'), (Join-Path $Frontend 'dist'))
        foreach ($t in $targets) {
            $full = Join-Path $RepoRoot $t
            if (Test-Path $full) { Remove-Item -Recurse -Force $full -ErrorAction SilentlyContinue }
        }
        Get-ChildItem -Path $RepoRoot -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '\\\.venv\\' -and $_.FullName -notmatch '\\node_modules\\' } |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Ok '已清理（data/ 与 .venv/ 未动）'
    }

    default {
        Write-Host "x 未知任务：$Task" -ForegroundColor Red
        Get-HelpText
        exit 1
    }
}
