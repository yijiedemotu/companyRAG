# =====================================================================================
# KnowFlow 前端镜像（多阶段：pnpm 构建 → nginx 托管静态文件 + 反代 API）
#
# 构建（构建上下文是 frontend/，因为只需要它自己的源码与 lockfile）：
#     docker build -f frontend/Dockerfile -t knowflow-frontend ./frontend
#   或
#     docker compose build frontend
#
# ⚠ 本文件**未实测**（开发机没装 Docker）。已做的只有：人工核对 + compose 的 YAML 语法解析。
#
# 为什么用 nginx 而不是 `vite preview`：preview 是开发服务器，
# 不支持 SPA fallback、不设缓存头、也没有反代能力（而生产必须同源反代 /api，
# 否则前端要处理跨域，还要把后端地址暴露到浏览器里）。
# =====================================================================================

# --------------------------------------------------------------------------------------
# 阶段 1：builder —— 用 pnpm 装依赖并构建出 dist/
# --------------------------------------------------------------------------------------
FROM node:24-alpine AS builder

# 固定 pnpm 大版本：lockfile 是 pnpm 10 生成的，用 corepack 保证一致，
# 否则 pnpm 9 会因为 lockfile 版本不同而拒绝安装。
RUN corepack enable && corepack prepare pnpm@10.33.0 --activate

WORKDIR /build

# 只拷 lockfile 与 package.json 先进来：源码改动不会让依赖层缓存失效
COPY package.json pnpm-lock.yaml ./
# 国内加速；不需要可删掉 --registry
RUN pnpm install --frozen-lockfile --registry=https://registry.npmmirror.com

# 再拷源码并构建（build 脚本 = vue-tsc --noEmit && vite build，
# 也就是**类型检查不过就不会产出镜像** —— 这是刻意的门禁）
COPY . .
RUN pnpm build


# --------------------------------------------------------------------------------------
# 阶段 2：runtime —— 只要 nginx + dist/，node_modules 全部丢掉（省几百 MB）
# --------------------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

LABEL org.opencontainers.image.title="knowflow-frontend" \
      org.opencontainers.image.description="KnowFlow 知识库问答前端（Vue 3 + Element Plus）" \
      org.opencontainers.image.version="1.0.0"

# 去掉默认站点配置，换成我们的（含 SPA fallback 与 /api 反代）
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/knowflow.conf

COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
