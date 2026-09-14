# ── Backend Dockerfile ────────────────────────────────────────────────────────
# Uses Python 3.12-slim. Models are mounted as a volume at runtime so the
# image stays small (no 1GB model files baked in).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps needed by opencv, wfdb, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY backend/   ./backend/
COPY src/       ./src/
COPY config.py  .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
