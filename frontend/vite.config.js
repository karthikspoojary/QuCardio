import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Dev proxy: forwards /predict*, /health, /docs → FastAPI on :8000
    // This makes relative fetch('/predict') work identically in both
    // "npm run dev" (Vite :5173) and Docker/nginx (same-origin) workflows.
    proxy: {
      '/predict': 'http://localhost:8000',
      '/health':  'http://localhost:8000',
      '/docs':    'http://localhost:8000',
      '/openapi': 'http://localhost:8000',
    },
  },
})
