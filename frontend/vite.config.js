import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// 我们不需要导入 postcss.config.cjs 本身，Vite 会根据路径找到它

export default defineConfig({
  plugins: [react()],
  css: { // <--- 添加这个 css 字段
    postcss: './postcss.config.cjs' // <--- 显式指向我们的 PostCSS 配置文件
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      }
    }
  }
});