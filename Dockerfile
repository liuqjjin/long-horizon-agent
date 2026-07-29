# syntax=docker/dockerfile:1.7

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d
ARG PYTHON_IMAGE=python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93
ARG EMBEDDER_REPOSITORY=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDER_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

FROM ${UV_IMAGE} AS uv-provider

FROM ${PYTHON_IMAGE} AS builder

ARG EMBEDDER_REPOSITORY
ARG EMBEDDER_REVISION

COPY --from=uv-provider /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra context

RUN /opt/venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${EMBEDDER_REPOSITORY}', revision='${EMBEDDER_REVISION}', local_dir='/opt/lha/models/all-MiniLM-L6-v2', allow_patterns=['1_Pooling/config.json', 'config.json', 'config_sentence_transformers.json', 'model.safetensors', 'modules.json', 'sentence_bert_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'])"

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra context

# The [context] extra ships the vector-index stack so `lha eval` reaches 6/6
# in-container (docs/DEPLOY.md). The uv source map selects CPU-only PyTorch on
# Linux; a CUDA runtime is unnecessary for this image. Python matches the
# repository's pinned version.
FROM ${PYTHON_IMAGE} AS runtime

COPY --from=uv-provider /uv /uvx /usr/local/bin/

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --home-dir /home/lha --shell /usr/sbin/nologin lha

ENV PATH="/opt/venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/lha/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    COCOINDEX_DISABLE_USAGE_TRACKING=1 \
    LHA_EMBEDDER_MODEL=/opt/lha/models/all-MiniLM-L6-v2

WORKDIR /app

COPY --from=builder --chown=lha:lha /opt/venv /opt/venv
COPY --from=builder --chown=lha:lha /opt/lha/models /opt/lha/models
COPY --from=builder --chown=lha:lha /app /app

RUN mkdir -p "${HF_HOME}" /app/runs && \
    chown -R lha:lha /home/lha/.cache /app/runs

USER lha

CMD ["lha", "--help"]
