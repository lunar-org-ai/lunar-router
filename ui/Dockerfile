# ============================================================================
# Lunar Router — UI (multi-stage build)
# ============================================================================
# Usage:
#   docker build -t lunar-ui ./ui
#   docker run -p 3000:80 lunar-ui
# ============================================================================

# --- Stage 1: Build ---
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Build with API URLs pointing to nginx proxy paths
ARG VITE_API_BASE_URL=""
ARG VITE_ROUTER_URL=""
ARG VITE_STATS_URL=""
ARG VITE_ENVIRONMENT=prod

RUN npx vite build

# --- Stage 2: Serve with nginx ---
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
