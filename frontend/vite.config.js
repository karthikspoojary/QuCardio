import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev proxy: forwards all API paths → FastAPI on :8000
    // This makes relative fetch('/predict') work identically in both
    // "npm run dev" (Vite :5173) and Docker/nginx (same-origin) workflows.
    proxy: {
      '/predict':       { target: 'http://localhost:8000', changeOrigin: true },
      '/health':        { target: 'http://localhost:8000', changeOrigin: true },
      '/docs':          { target: 'http://localhost:8000', changeOrigin: true },
      '/openapi':       { target: 'http://localhost:8000', changeOrigin: true },
      '/circuit':       { target: 'http://localhost:8000', changeOrigin: true },
      '/history':       { target: 'http://localhost:8000', changeOrigin: true },
      '/ablation_data': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
