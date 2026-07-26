# Build and deployment checks

LHA is distributed as a Python CLI and an application container. It is not a
hosted service. A release candidate is valid only when the wheel, source
distribution, container image, and Docker execution backend all pass their
respective checks.

## Build Python distributions

```bash
uv build
```

The packaged context flows are runtime resources, so check both archives:

```bash
unzip -l dist/*.whl | grep 'lha/live_context/flows/common.py'
tar -tf dist/*.tar.gz | grep 'src/lha/live_context/flows/common.py'
```

Test installation outside the checkout. Using `--no-project` prevents `uv` from
importing the source tree by accident:

```bash
REPO_ROOT="$PWD"

WHEEL_TMP="$(mktemp -d)"
cd "$WHEEL_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.whl lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.whl \
  python -c "import importlib.util; assert importlib.util.find_spec('lha.live_context.flows.common')"

SDIST_TMP="$(mktemp -d)"
cd "$SDIST_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz \
  python -c "import importlib.util; assert importlib.util.find_spec('lha.live_context.flows.common')"
```

Return to the repository root before running project commands.

These commands test the core package. On Linux and Windows, this repository's
uv configuration binds `torch` to the official CPU wheel index. That source
mapping is not part of wheel metadata: consumers installing `lha[context]`
outside this checkout must make the same binding in their own uv project or use
the application image. Do not describe a bare external `lha[context]` install
as lightweight.

## Build the application image

```bash
docker build -t lha:release .
docker run --rm lha:release lha --version
docker run --rm lha:release \
  python -c "import lha, cocoindex, sentence_transformers"
```

The image runs as the unprivileged `lha` user. It includes the `context` extra,
but intentionally excludes the external `ccc` executable.

The build context must not copy local secrets or release output. Verify the
known sensitive paths:

```bash
docker run --rm lha:release python -c \
  "from pathlib import Path; assert not any((Path('/app') / p).exists() for p in ('.env', '.codex', '.claude', '.mcp.json', 'auth.json', '.ssh', '.aws', '.config/gcloud', '.netrc', '.pypirc', 'dist'))"
```

`.dockerignore` also excludes `.env.*` except `.env.example`, credentials
directories, run state, coverage output, and build artifacts. Do not pass
authentication as a build argument or bake it into a layer.

## Run the self-evaluation in the image

Use named volumes for writable state and the model cache:

```bash
docker volume create lha-runs
docker volume create lha-hf

docker run --rm \
  -v lha-runs:/app/runs \
  -v lha-hf:/home/lha/.cache/huggingface \
  lha:release lha eval
```

The command must exit zero. Its first invocation can download the configured
embedding model, which is why the Hugging Face cache is persisted separately
from run evidence.

Run a single offline task the same way:

```bash
docker run --rm \
  -v lha-runs:/app/runs \
  lha:release lha run data/tasks/fix_average.yaml
```

## Validate the Docker execution backend

The application image and the Docker execution backend serve different
purposes:

- `lha:release` contains the LHA CLI.
- `LHA_EXEC_IMAGE` is the disposable environment in which target or
  model-influenced commands run.

The default execution image, `python:3.12-slim`, does not contain pytest, the
pytest JSON plugin, or Ruff. Code tasks need a purpose-built execution image
with all declared tools installed.

Run the real-container backend tests:

```bash
docker build -t lha:release .
LHA_DOCKER_TESTS=1 LHA_DOCKER_TEST_IMAGE=lha:release \
  uv run pytest tests/test_sandbox.py -q
```

These tests require a working Docker daemon. A missing daemon is not a successful
release check.

## Codex credentials

The normal Codex CLI backend reads the locally authenticated CLI state, copies
only the required authentication into a temporary `CODEX_HOME`, and removes it
after the attempt. Do not mount a developer's complete home directory into the
application image.

For an official benchmark container, inject authentication only at task runtime:

- mount or copy credentials into a task-local temporary directory;
- never add them to the image, repository, command output, or run report;
- terminate the Codex process group before deleting the temporary directory;
- record CLI version, model, reasoning effort, image digest, and budgets, but
  not credential contents.

`danger-full-access` for Codex is rejected unless
`LHA_CODEX_EXTERNAL_SANDBOX=1` explicitly states that an outer disposable
container is already the security boundary.

## Release checklist

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

Also run the facade-isolation scan in `CONTRIBUTING.md` and both empty-directory
package installs above. Record the actual output in the pull request; do not copy
test counts or benchmark numbers from an earlier commit.
