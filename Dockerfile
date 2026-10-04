# JagX AI Backend — production Docker image
# Created by JagX & JRILICENSE
FROM python:3.11-slim-bookworm

# Runtime behaviour
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000 \
    # Keep free-tier friendly defaults
    JAGX_SELF_LEARN_ENABLED=true \
    JAGX_SELF_LEARN_SECONDS=1800

WORKDIR /app

# System deps:
# - ca-certificates: HTTPS to Groq/OpenRouter/HF/OSM/news RSS
# - curl: healthcheck + debugging
# - build-essential: only if a wheel needs compile (kept lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Install Python deps first (better layer cache on code-only changes)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# App source (knowledge packs, extensions, main/app)
COPY . .

# Non-root user (safer on Render/shared hosts)
RUN useradd --create-home --shell /bin/bash jagx \
    && chown -R jagx:jagx /app
USER jagx

EXPOSE 10000

# Render sets PORT; main.py / uvicorn must honour it
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-10000}/health" || exit 1

# Prefer uvicorn if main is thin; main.py already runs uvicorn
CMD ["python", "main.py"]
