# Quickstart

Run commands from the repository root. LHA requires Python 3.11 or newer and
[`uv`](https://docs.astral.sh/uv/). The current process-cleanup boundary supports
Linux, macOS, and WSL2. Native Windows is not supported yet; use WSL2 or Docker.

## Install

```bash
uv sync
uv run lha --version
```

The source checkout installs development and document-index dependencies. The
core package can also be installed from this checkout with `python -m pip install .`,
or from a wheel produced by `uv build`.

When dependencies are resolved on Linux or Windows, this repository points
PyTorch at the official CPU index.
That source setting is not stored in wheel metadata. If another project installs
the `context` extra, it should configure the same index or use the application
image.

## Run an offline code task

The default model backend is deterministic and requires no credentials or
`ccc`:

```bash
uv run lha run data/tasks/fix_average.yaml
```

The run reaches `DONE` only after its Pytest and Ruff checks pass. Keep the
printed `run_id`, then inspect the saved evidence:

```bash
RUN_ID=replace-with-the-printed-run-id
uv run lha runs show "$RUN_ID"
uv run lha trace "$RUN_ID"
uv run lha trace "$RUN_ID" --html
```

This bundled task explicitly makes indexed code context optional. It still runs
the real checks when `ccc` is not installed. `ccc` is an optional code-indexing
feature, not a quickstart prerequisite; tasks that require indexed context still
fail closed when the backend is unavailable.

The HTML report contains the timeline, patch, approvals, check results, repair
events, and recorded model usage.

## Pause for approval and resume

```bash
uv run lha run \
  --runtime langgraph \
  data/tasks/fix_average_approval.yaml

APPROVAL_RUN_ID=replace-with-this-run-id
uv run lha approve "$APPROVAL_RUN_ID" --note "reviewed patch"
uv run lha resume --runtime langgraph "$APPROVAL_RUN_ID"
```

Approval records the step and SHA-256 of the persisted patch. Resume checks the
same bytes before continuing.

## Run the repository self-eval

```bash
LHA_RUNS_DIR=runs/_quickstart uv run lha eval
```

The command runs six fixed workflows and returns non-zero if any expected status
or check is missing. See [BENCHMARKS.md](BENCHMARKS.md) for the cases.

## Use indexed context

Code indexing uses the optional `ccc` executable:

```bash
uv run lha index data/sample_repo
uv run lha ask "how is average computed" \
  --root data/sample_repo \
  --kinds code
```

Document indexing uses the packaged flows:

```bash
uv run lha index-docs
uv run lha ask "what evidence supports the experiment" \
  --kinds paper,experiment,skill
```

Results include locators and source evidence. Required context fails with a
typed status when its backend is unavailable or its index cannot be refreshed.

## Use a local Codex login

Authenticate the installed `codex` CLI, then run:

```bash
uv run lha --llm codex_cli run data/tasks/fix_average.yaml
```

This uses the current CLI model settings. For a measured run, set
`LHA_CODEX_MODEL` and `LHA_CODEX_EFFORT` to values supported by the account and
record them with the result.

LHA gives the CLI a temporary home and workspace, validates its JSONL events,
and records secret-free provenance. Handled exit paths confirm that the original
process group stopped before removing temporary credentials. A tool process
that deliberately leaves that group is outside the host backend's guarantee;
run untrusted tools inside Docker or another outer containment boundary.

## Inspect and prune runs

```bash
uv run lha runs list
uv run lha runs show "$RUN_ID"
uv run lha runs prune --older-than-days 30
```

Pruning is a dry run unless `--apply` is present. Active, locked, corrupt, or
unfinished runs are not deleted.

## Run the long-task checks

Five fixed multi-file fixtures exercise ten repository stages, approval,
repair, interruption, and resume:

```bash
uv run pytest \
  tests/test_long_tasks.py \
  tests/test_long_task_harness.py -q
```

## Build distributions

```bash
uv build
```

Install the wheel and source archive from an empty directory before release.
Exact package and container commands are in [DEPLOY.md](DEPLOY.md). The recovery
model is described in [ARCHITECTURE.md](ARCHITECTURE.md).
