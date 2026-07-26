# Quickstart

Run commands from the repository root. The core requires Python 3.11 or newer
and [`uv`](https://docs.astral.sh/uv/).

## Install

```bash
uv sync
uv run lha --version
```

`uv sync` installs the development group and document-index dependencies. A
package-only installation can use `pip install lha`. For paper, experiment, and
skill indexing, use this source checkout or the project image: the checked-in
uv configuration selects the official CPU-only PyTorch index on Linux and
Windows. Wheel metadata cannot carry that index selection, so a bare
`pip install "lha[context]"` on Linux may resolve much larger GPU packages.
Consumers that need the context extra from a wheel should bind `torch` to the
CPU index in their own uv project. When an optional backend is unavailable,
required-context steps fail with a typed status rather than silently returning
no matches.

## Run an offline code task

The default `stub` backend is deterministic, does not need credentials, and fixes
the planted bug in `data/sample_repo`:

```bash
uv run lha run data/tasks/fix_average.yaml
```

The command should finish with `status : DONE` only after the actual pytest and
Ruff checks pass. Keep the printed `run_id`, then inspect its evidence:

```bash
uv run lha runs show <run_id>
uv run lha trace <run_id>
uv run lha trace <run_id> --html
```

The HTML command writes a self-contained `trace.html` under the run directory.
It shows the state timeline, patch, approval records, verifier results, repairs,
and persisted model-usage totals.

## Exercise approval and resume

```bash
uv run lha run --runtime langgraph data/tasks/fix_average_approval.yaml
# copy the run_id from the AWAITING_APPROVAL result
uv run lha approve <run_id> --note "reviewed patch"
uv run lha resume --runtime langgraph <run_id>
```

The approval names the step and SHA-256 of the reviewed `patch.json`. The
LangGraph runner checkpoints the prepared artifact before it interrupts, so a
new process resumes the reviewed bytes rather than generating a replacement.
The default loop supports the same approval contract without `--runtime
langgraph`.

## Run the repository self-evaluation

```bash
LHA_RUNS_DIR=runs/_quickstart uv run lha eval
```

The command covers six fixed repository workflows and returns non-zero unless
all six meet their expected status and checks. The first run may download the
configured sentence-transformer model. See [BENCHMARKS.md](BENCHMARKS.md) for
the case definitions.

## Inspect and retain runs

```bash
uv run lha runs list
uv run lha runs show <run_id>
uv run lha runs prune --older-than-days 30
```

`prune` is a dry run unless `--apply` is supplied. Application is limited to
validated `DONE` or `FAILED` runs; locked, active, or corrupt state is refused.

## Use indexed context

Code search uses the optional `ccc` executable:

```bash
uv run lha index data/sample_repo
uv run lha ask "how is average computed" \
  --root data/sample_repo \
  --kinds code
```

Paper, experiment, and skill indexes use the packaged flows:

```bash
uv run lha index-docs
uv run lha ask "what evidence supports the experiment" \
  --kinds paper,experiment,skill
```

Each hit carries a locator and source evidence. A stale index must refresh
successfully before `reject_stale` clears the stale flag.

## Use an authenticated Codex CLI

First authenticate the locally installed `codex` CLI. Then pin the model and
reasoning effort appropriate to the run:

```bash
LHA_CODEX_MODEL=<model-id> \
LHA_CODEX_EFFORT=medium \
uv run lha --llm codex_cli run data/tasks/fix_average.yaml
```

LHA copies the required authentication into a temporary `CODEX_HOME`, passes a
minimal environment, validates the JSONL event stream, records CLI/model/event
provenance, and deletes the temporary home and workspace on success, failure,
timeout, or interruption. Authentication bytes are not written to run
artifacts.

For ablation, the Codex path additionally rejects every tool-use event: the
first attempt must be derived from the source included in the prompt.

## Run the five long-task fixtures

The five cases under `data/long_tasks/` use a fixed 10-step repository protocol.
Their reference-patch and interruption/recovery contracts are exercised
hermetically:

```bash
uv run pytest tests/test_long_tasks.py tests/test_long_task_harness.py -q
```

The tests cover configuration parsing, SQLite migration, concurrent failures,
CLI contracts, and experiment reproducibility. They include a rejected first
patch, a repair, approval resumes, a simulated process exit at a safe boundary,
and comparison with an uninterrupted result.

## Build and test the distributions

```bash
uv build
```

Install the wheel and source distribution from an empty scratch directory, not
from this checkout:

```bash
REPO_ROOT="$PWD"
PACKAGE_TMP="$(mktemp -d)"
cd "$PACKAGE_TMP"
uv run --no-project --with "$REPO_ROOT"/dist/*.whl lha --version
uv run --no-project --with "$REPO_ROOT"/dist/*.tar.gz lha --version
```

See [DEPLOY.md](DEPLOY.md) for container validation and
[ARCHITECTURE.md](ARCHITECTURE.md) for the recovery model.
