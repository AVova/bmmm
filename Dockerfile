# Multi-stage build with uv. One image serves both the FastAPI service and the
# Streamlit dashboard; docker-compose picks the command per service.

# ---- builder: resolve and install dependencies into a venv -------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install deps first (cached unless pyproject/lock change), then the project.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra dashboard

# ---- runtime: slim image with just the venv and the app code -----------------
FROM python:3.12-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    BMMM_ARTIFACTS="/app/artifacts" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY dashboard ./dashboard

# The 92MB trained model is not baked in; mount ./artifacts as a volume instead.
EXPOSE 8000 8501

# Default command runs the API; the dashboard service overrides this in compose.
CMD ["uvicorn", "bmmm.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
