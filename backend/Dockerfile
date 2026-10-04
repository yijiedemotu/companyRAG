# =====================================================================================
# KnowFlow 后端镜像（多阶段：builder 装依赖 → runtime 只拷结果）
#
# 构建（**构建上下文必须是仓库根**，因为要拷 backend/ 与 data/samples/）：
#     docker build -f backend/Dockerfile -t knowflow-backend .
#   或
#     docker compose build backend
#
# ⚠ 本文件在本机**未实测**（开发机没装 Docker）。已做的只有：人工核对 + compose 的 YAML 语法解析。
#    第一次在有 Docker 的机器上跑时，最可能出问题的是 pip 依赖解析与模型路径，请留意日志。
#
# --------------------------------------------------------------------------------------
# 关于模型权重（2GB）——镜像里**不包含** BGE-M3，两个选择：
#
#   A) 构建时下载（镜像大但开箱即用）
#        取消下面 Dockerfile 里 build 阶段的注释，或构建时加：
#            docker build -f backend/Dockerfile --build-arg PRELOAD_MODEL=1 -t knowflow-backend .
#        代价：镜像 +2GB，构建慢且依赖网络。
#
#   B) 【推荐】运行时挂载卷 + EMBEDDING_MODEL_PATH
#        把宿主机的 HF 缓存或模型目录挂进容器，完全离线、镜像保持小：
#            volumes:
#              - ${HOME}/.cache/huggingface:/models/hf:ro
#            environment:
#              EMBEDDING_MODEL_PATH: /models/hf/models--BAAI--bge-m3/snapshots/<sha>
#        见 docker-compose.yml 里 backend 服务的注释。
#
#   都不做也可以：设 EMBEDDING_PROVIDER=hash，用哈希向量兜底把链路跑通（零依赖）。
# =====================================================================================

# --------------------------------------------------------------------------------------
# 阶段 1：builder —— 只负责把依赖装进 /opt/venv，编译工具链留在这一层
# --------------------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# 国内构建加速（如不需要可删掉这一行，走官方源）
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 装依赖需要编译器：chromadb/torch 的部分依赖没有 manylinux 预编译包时要从源码构建
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# --- 1) 【关键】先装 CPU-only 的 torch --------------------------------------------------
# 不加 --index-url 的话，pip 默认从 PyPI 装 torch，会拉 **2.5GB 的 CUDA 版**
# （连带 nvidia-* 一堆包），而容器里根本没有 GPU —— 白白多 2GB 且启动更慢。
# download.pytorch.org/whl/cpu 上的是同一版本号的 CPU 构建，约 200MB。
# 必须先装：让 pip 在前面的层里就确定 torch，后面 requirements.txt 里的
# sentence-transformers 只会复用它，不会再把 CUDA 版拖回来。
ARG TORCH_VERSION=2.9.1
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

# --- 2) 再装其余依赖（requirements.txt 是版本的唯一权威） -------------------------------
# 只拷 requirements 先进来：源码改动不会让这一层缓存失效。
COPY backend/requirements.txt /build/requirements.txt
RUN pip install --index-url "${PIP_INDEX_URL}" -r /build/requirements.txt

# --- 3) 可选：构建时预下载 BGE-M3 权重 -------------------------------------------------
# 默认关闭。开启方式：--build-arg PRELOAD_MODEL=1
ARG PRELOAD_MODEL=0
ARG HF_ENDPOINT=https://hf-mirror.com
ENV HF_HOME=/models/hf
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
        HF_ENDPOINT="${HF_ENDPOINT}" pip install --index-url "${PIP_INDEX_URL}" "huggingface-hub[cli]" && \
        HF_ENDPOINT="${HF_ENDPOINT}" python -m huggingface_hub.commands.huggingface_cli download BAAI/bge-m3; \
    else \
        echo "跳过模型预下载（运行时用 EMBEDDING_MODEL_PATH 挂载，或 EMBEDDING_PROVIDER=hash）"; \
    fi

# --------------------------------------------------------------------------------------
# 阶段 2：runtime —— 只拷 venv + 源码，带编译器的那一层丢掉
# --------------------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="knowflow-backend" \
      org.opencontainers.image.description="企业知识库 RAG + LangGraph Agent 问答服务" \
      org.opencontainers.image.version="1.0.0"

# - PYTHONUNBUFFERED: 日志实时刷出来，否则 docker logs 看不到
# - PYTHONDONTWRITEBYTECODE: 容器里不留 __pycache__
# - PYTHONPATH: 让 `knowflow` 可导入（src 布局）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:${PATH}" \
    TZ=UTC

# 运行期只要这两个系统库：
#   libgomp1 —— torch / numpy 的 OpenMP 运行库（CPU 推理必需，缺了 import torch 就失败）
#   curl     —— HEALTHCHECK 用
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# --- 非 root 用户 ----------------------------------------------------------------
# UID/GID 固定成 10001，避免和宿主机用户冲突；named volume 的属主问题也更好推理。
RUN groupadd --gid 10001 knowflow \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin knowflow

# --- 从 builder 拷 venv（只拷结果，不带 apt 缓存与编译器） ------------------------
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# --- 源码 ------------------------------------------------------------------------
COPY --chown=knowflow:knowflow backend/src/       /app/src/
COPY --chown=knowflow:knowflow backend/alembic/   /app/alembic/
COPY --chown=knowflow:knowflow backend/alembic.ini /app/alembic.ini
COPY --chown=knowflow:knowflow backend/scripts/   /app/scripts/
COPY --chown=knowflow:knowflow backend/pyproject.toml /app/pyproject.toml

# --- 样例文档与评测集：seed_demo / run_eval / calibrate 都要用 --------------------
# 这两个是仓库资产（不是运行期数据），必须进镜像，否则容器里跑不了评测与标定。
COPY --chown=knowflow:knowflow data/samples/          /app/data/samples/
COPY --chown=knowflow:knowflow data/eval_cases.jsonl  /app/data/eval_cases.jsonl

# --- 运行期数据目录（上传原文件 / 向量库 / checkpointer） -------------------------
# 用 volume 覆盖：data/chroma、data/uploads 都必须持久化，否则重建容器即丢索引。
# ⚠ DATA_DIR 与 DATABASE_URL 是**一一对应**的：挂错卷会让向量库和 MySQL 串库
#   （见 docs/00 第 8 节）。compose 里两边都指向同一个命名卷就是为了防这个。
RUN mkdir -p /app/data/chroma /app/data/uploads \
 && chown -R knowflow:knowflow /app/data
ENV DATA_DIR=/app/data

# 模型权重目录的挂载点（选择 B 用；不挂也无害）
RUN mkdir -p /models && chown -R knowflow:knowflow /models
ENV HF_HOME=/models/hf

USER knowflow

EXPOSE 8000

# --------------------------------------------------------------------------------------
# 健康检查：打 /health（无需鉴权），它返回的是**真实运行模式**
#   --start-period 给足 60s：首次启动要连库、建 Chroma 客户端、重建 BM25 索引，
#   如果配了 local embedding 还要加载模型（CPU 上可能十几秒）。
#   这段时间内的失败不算数，避免容器被误判为不健康而反复重启。
# --------------------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 注意：不要用 `--reload`（那是开发用的，会在容器里起额外的 watcher 进程）
CMD ["uvicorn", "knowflow.main:app", "--host", "0.0.0.0", "--port", "8000"]
