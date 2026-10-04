import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

/**
 * Vite 配置。
 * 开发环境：`/api` 前缀的请求通过 proxy 转发到后端 127.0.0.1:8000，
 * 这样前端代码里只需要写 `/api/v1/...`，不存在跨域问题，也不用管 CORS。
 */
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      // 注意：SSE 是长连接，必须关闭 proxy 的缓冲（vite 默认不缓冲，这里显式声明便于排障）
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: false,
      },
      // 运维接口在根路径（/health /ready /metrics），一并代理，便于前端做健康检查
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ready': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/metrics': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        // 手动分包：echarts 与 element-plus 体积大，单独拆出来避免主包过大
        manualChunks: {
          echarts: ['echarts'],
          'element-plus': ['element-plus'],
          vendor: ['vue', 'vue-router', 'pinia', 'axios', 'dayjs'],
        },
      },
    },
  },
})
