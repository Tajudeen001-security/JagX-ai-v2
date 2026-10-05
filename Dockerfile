FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=10000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* && update-ca-certificates
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY jagx_api ./jagx_api
COPY main.py app.py ./
COPY jagx_knowledge.json* ./
RUN useradd --create-home --shell /bin/bash jagx && chown -R jagx:jagx /app
USER jagx
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=8s --start-period=25s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-10000}/health" || exit 1
CMD ["python", "main.py"]
