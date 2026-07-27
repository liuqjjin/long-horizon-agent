# LHA: execution, recovery, and verification for coding tasks

LHA runs code changes, experiments, and retrieval-backed tasks as explicit
steps. Registered checks decide whether a step may advance. A failed check can
feed a bounded repair attempt, and an interrupted run can resume from saved
state.

The [Chinese README](../README.md) is the main project page.

## What it handles

A coding task often takes more than one model call. If an early mistake is not
caught, later steps can build on it and still produce a plausible-looking
result. LHA keeps state transitions, budgets, approval, checks, repair, and
rollback in the runner:

```text
context → execute → [approval] → verify → repair or advance → checkpoint
```

The internal checks only decide whether a run advances. Ablation grading uses a
separate execution path: it applies the saved source change to a fresh repository
copy and runs the fixed tests there. The internal decision is not reused as the
evaluation result.

## Run it

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run lha run data/tasks/fix_average.yaml
LHA_RUNS_DIR=runs/_quickstart uv run lha eval
```

The default backend is deterministic and needs no model credentials. To use a
locally authenticated Codex CLI:

```bash
LHA_CODEX_MODEL=YOUR_MODEL_ID \
LHA_CODEX_EFFORT=medium \
uv run lha --llm codex_cli run data/tasks/fix_average.yaml
```

Inspect the saved state and evidence:

```bash
RUN_ID=replace-with-the-run-id
uv run lha runs show "$RUN_ID"
uv run lha trace "$RUN_ID"
uv run lha trace "$RUN_ID" --html
```

## Implementation

- Schema-v2 run state stores the cursor, attempt identifiers, repair counters,
  original budgets, elapsed time, and model usage.
- `state.json` is checksummed and atomically replaced after `fsync`;
  `ledger.jsonl` is append-only.
- A file lock prevents concurrent resume of one run. Idempotency keys prevent
  duplicate completion and approval events.
- `ResolvedPatch` derives the write set from the actual patch instead of model
  metadata.
- `PatchTransaction` records `PREPARED`, `APPLIED`, `VERIFIED`, and `REVERTED`,
  together with manifests and backups.
- Approval binds the step and SHA-256 digest of the persisted patch.
- Code checks run Pytest, Ruff, or repository-defined commands. Experiment
  checks recompute metrics from saved arrays and rerun in a fresh directory.
- Target- or model-influenced commands use the `ExecutionBackend` interface.
- The optional LangGraph runtime uses the same execution and check helpers and
  stores graph state in SQLite.
- Indexed code and documents are accessed through `lha.live_context`, which
  records source digests, freshness, and unavailable states.

## Fixed multi-file tasks

`data/long_tasks/` contains five fixed cases for configuration parsing, SQLite
migration, concurrent updates, CLI contracts, and experiment reproduction.
Each case includes a task, repository adapter, reference patch, and digests.

The cases run ten stages from integrity checks and problem reproduction through
approved editing, targeted tests, full tests, lint, and build. Tests cover a
rejected first patch, repair, approval resume, process interruption, and
equivalent final state after recovery.

## Evaluation status

The repository retains a schema-v2 ablation report as a record of the earlier
protocol. The scoring boundary, error classification, and evidence format have
since changed, so that report is not the current project result. The current
implementation uses schema v4; a complete rerun is required before new ablation
numbers are published.

The Terminal-Bench 2.1 adapter runs tasks and the official verifier through
Harbor. A direct Harbor run measures that model execution; it does not measure
LHA's internal gate or repair loop. No public benchmark score is listed here
before its protocol, raw results, manifest, and summary are committed together.

See [ABLATION.md](ABLATION.md) and [BENCHMARKS.md](BENCHMARKS.md) for the
evaluation methods.

## Limits

- This is maintained for research and engineering evaluation, not as an online
  service.
- `trusted-local` is not a sandbox for hostile code.
- Docker behavior depends on the image, mounts, network, and resource settings.
- `LHA_DEADLINE_S` is checked at durable boundaries; each blocking operation
  still relies on its own timeout for preemption.
- Source freshness and citation checks are weaker than executable tests.
- Passing the registered tests does not prove correctness for every input.
- A benchmark adapter is not a benchmark result.

Build and package checks are documented in [DEPLOY.md](DEPLOY.md). The system
structure is described in [ARCHITECTURE.md](ARCHITECTURE.md).
