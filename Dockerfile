# syntax=docker/dockerfile:1.7

ARG UV_VERSION=0.11.16

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-provider

FROM python:3.11-slim AS builder

COPY --from=uv-provider /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra context

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra context

# The [context] extra ships the vector-index stack so `lha eval` reaches 6/6
# in-container (docs/DEPLOY.md). The uv source map selects CPU-only PyTorch on
# Linux; a CUDA runtime is unnecessary for this image. Python matches the
# repository's pinned version.
FROM python:3.11-slim AS runtime

COPY --from=uv-provider /uv /uvx /usr/local/bin/

ENV PATH="/opt/venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/lha/.cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --home-dir /home/lha --shell /usr/sbin/nologin lha

WORKDIR /app

COPY --from=builder --chown=lha:lha /opt/venv /opt/venv
COPY --from=builder --chown=lha:lha /app /app

RUN mkdir -p "${HF_HOME}" /app/runs && \
    chown -R lha:lha /home/lha/.cache /app/runs

USER lha

CMD ["lha", "--help"]
