# Build and release checks

LHA is distributed as a Python CLI and an application image. It is not a hosted
service. A release candidate must be tested as source, wheel, source archive,
container image, and Docker execution backend.

## Build Python packages

```bash
uv build
```

The context flows are package data. Check both archives:

```bash
unzip -l dist/*.whl | grep 'lha/live_context/flows/common.py'
tar -tf dist/*.tar.gz | grep 'src/lha/live_context/flows/common.py'
```

Install from directories outside the checkout so imports cannot fall back to
local source:

```bash
REPO_ROOT="$PWD"

WHEEL_TMP="$(mktemp -d)"
cd "$WHEEL_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.whl lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.whl \
  python -c "import lha.live_context.flows.common"

SDIST_TMP="$(mktemp -d)"
cd "$SDIST_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz \
  python -c "import lha.live_context.flows.common"
```

Return to the repository root before running project commands.

This repository points Linux and Windows installs at the official CPU-only
PyTorch index. Wheel metadata does not carry that source setting. Projects that
install the `context` extra must configure the CPU index themselves or use the
application image.

## Build the application image

```bash
docker build -t lha:release .
docker run --rm lha:release lha --version
docker run --rm lha:release \
  python -c "import lha, cocoindex, sentence_transformers"
```

The image runs as the unprivileged `lha` user and includes the `context` extra.
It does not include the external `ccc` executable.

Check that common credential and build paths are absent:

```bash
docker run --rm lha:release python -c \
  "from pathlib import Path; assert not any((Path('/app') / p).exists() for p in ('.env', '.codex', '.claude', '.mcp.json', 'auth.json', '.ssh', '.aws', '.config/gcloud', '.netrc', '.pypirc', 'dist'))"
```

Do not pass authentication as a build argument or copy it into an image layer.

## Run in the application image

Use volumes for run state and the optional model cache:

```bash
docker volume create lha-runs
docker volume create lha-hf

docker run --rm \
  -v lha-runs:/app/runs \
  -v lha-hf:/home/lha/.cache/huggingface \
  lha:release lha eval
```

The command must exit zero. The model cache is separate from run evidence.

## Test the Docker execution backend

The application image contains the LHA CLI. The execution image is the
disposable environment used for commands from a target repository.

`python:3.12-slim`, the default execution image, does not contain Pytest,
`pytest-json-report`, or Ruff. Code tasks need an image with every declared
tool.

Run the real-container backend tests:

```bash
LHA_DOCKER_TESTS=1 \
LHA_DOCKER_TEST_IMAGE=lha:release \
uv run pytest tests/test_sandbox.py -q
```

A missing Docker daemon is a failed release check.

## Credentials

The normal Codex backend copies the required local CLI authentication into a
temporary `CODEX_HOME` and deletes it after the attempt. Do not mount a
developer home directory into the application image.

For benchmark containers:

- inject authentication only when the task starts;
- keep it out of images, repositories, arguments, logs, and reports;
- stop the Codex process group before deleting temporary files;
- record versions, model settings, image digests, and budgets, not credential
  contents or paths.

Codex `danger-full-access` is allowed only when
`LHA_CODEX_EXTERNAL_SANDBOX=1` confirms that a disposable outer container is
the security boundary.

## Release gate

From the repository root:

```bash
uv run ruff check .
uv run pyright src/lha
uv run pytest -q
LHA_RUNS_DIR=runs/_release uv run lha eval

uv build
docker build -t lha:release .
LHA_DOCKER_TESTS=1 LHA_DOCKER_TEST_IMAGE=lha:release \
  uv run pytest tests/test_sandbox.py -q
docker run --rm lha:release lha --version
```

Also run the facade-isolation scan in `CONTRIBUTING.md` and both package installs
above. Record command output from the release candidate; do not reuse test counts
or benchmark numbers from an earlier commit.
