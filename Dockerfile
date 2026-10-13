# ============================================================
# PromptShield — Multi-stage Dockerfile
# ============================================================
# Stage 1: Build the React frontend (Node 20)
# Stage 2: Run FastAPI + serve built frontend (Python 3.11 slim)
# ============================================================

# ── Stage 1: Frontend build ──────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better layer caching
COPY frontend/package.json frontend/package-lock.json ./

# Install all deps (including devDeps needed for the build)
RUN npm ci

# Copy the rest of the frontend source
COPY frontend/ ./

# Build the production bundle → outputs to /app/frontend/dist
RUN npm run build


# ── Stage 2: Python backend + serve built frontend ───────────
FROM python:3.11-slim AS runtime

# Set env so Python doesn't buffer output (important for Render logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install production Python dependencies
COPY requirements-prod.txt ./
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

# Copy the full backend source
COPY backend/ ./backend/
COPY alembic.ini ./
COPY migrations/ ./migrations/

# ── Copy the built React frontend from Stage 1 ───────────────
# FastAPI (backend/main.py) looks for frontend/dist at runtime
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create data directory for SQLite
RUN mkdir -p ./data ./logs

# Expose the port Render will set via $PORT (default 8000 for local)
EXPOSE 8000

# Start the app — Render injects $PORT at runtime
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
