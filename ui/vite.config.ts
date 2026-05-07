import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    host: true,
    strictPort: true,
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8002', changeOrigin: true },
    },
  },
});
