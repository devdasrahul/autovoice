# syntax=docker/dockerfile:1.4
# Multi-stage Dockerfile for AutoVoice FastAPI Microservice (CPU-Only / Cloud Run)

# Stage 1: Build virtual environment
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic dependency resolution
RUN pip install uv==0.5.18

COPY pyproject.toml .

# Create virtualenv and install serving dependencies with CPU PyTorch
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi==0.115.6" \
    "starlette==0.41.3" \
    "uvicorn[standard]==0.34.0" \
    "pydantic==2.11.7" \
    "pydantic-settings==2.10.1" \
    "python-dotenv==1.0.1" \
    "structlog==25.4.0" \
    "scikit-learn==1.5.2" \
    "pandas==2.2.3" \
    "numpy==1.26.4" \
    "pyarrow==18.1.0" \
    "duckdb==1.1.3" \
    "httpx==0.28.1" \
    "tenacity==9.1.2" \
    "PyYAML==6.0.2"


# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=prod \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user for security compliance
RUN groupadd -g 10001 autovoice && \
    useradd -u 10001 -g autovoice -s /bin/bash -m autovoice

# Copy pre-built environment
COPY --from=builder --chown=autovoice:autovoice /opt/venv /opt/venv

# Copy application artifacts and code
COPY --chown=autovoice:autovoice src/ src/
COPY --chown=autovoice:autovoice models/ models/
COPY --chown=autovoice:autovoice data/processed/ data/processed/
COPY --chown=autovoice:autovoice reports/ reports/
COPY --chown=autovoice:autovoice pyproject.toml .

USER autovoice

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD exec uvicorn autovoice.api.main:app --host 0.0.0.0 --port ${PORT}
