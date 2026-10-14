# Multi-stage Dockerfile for FraudShield Inference API
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv for ultra-fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml README.md ./

# Install dependencies into a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir .

# Final runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy source code and model artifacts
COPY pyproject.toml README.md ./
COPY ml/ ./ml/
COPY apps/ ./apps/
COPY artifacts/ ./artifacts/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
