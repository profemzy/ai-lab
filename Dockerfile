# syntax=docker/dockerfile:1.7
# Production-ready container for Labs FastAPI server with GPU support (via NVIDIA Container Toolkit)
# Uses uv for fast, reproducible installs from pyproject.toml + uv.lock

FROM python:3.12-slim AS base

# System settings and environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    # Cache locations (mount a volume at /data to persist model cache across runs)
    HF_HOME=/data/.cache/huggingface

# Install minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install uv (https://github.com/astral-sh/uv)
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && cp /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Copy project files required for dependency resolution
# Keeping these COPY steps higher improves Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the system interpreter (no project install to avoid README.md build requirement)
RUN uv sync --no-install-project --frozen || uv sync --no-install-project

# Copy only runtime application code
COPY labs ./labs

# Install local package into the system interpreter (wheel install, not editable)
RUN python -m pip install .
# Remove uv and build caches to slim runtime
RUN rm -f /usr/local/bin/uv && rm -rf /root/.local/bin /root/.cache/uv

# Create non-root user and set ownership for runtime and cache dir
RUN useradd -m -u 10001 appuser
RUN mkdir -p /data && chown -R appuser:appuser /data /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Default runtime configuration: auto device placement; override with docker run -e VAR=...
ENV LABS_DEVICE_MAP=auto

# Healthcheck: FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Start the API without autoreload, optimized for production.
# WEB_CONCURRENCY can be set by the orchestrator to increase worker count.
CMD ["sh", "-c", "exec python -m uvicorn labs.api:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 30 --workers ${WEB_CONCURRENCY:-1}"]