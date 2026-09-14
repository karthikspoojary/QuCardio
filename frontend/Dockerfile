# ── Frontend Dockerfile ───────────────────────────────────────────────────────
# Multi-stage: Node build → nginx serve
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build React/Vite app
FROM node:20-slim AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine AS runner

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# nginx config: proxy /api -> backend:8000 so CORS is never an issue
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
