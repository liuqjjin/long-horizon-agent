# LHA

LHA is a Python runner for recoverable coding tasks, experiments, and indexed
context:

```text
plan → context → execute → [approval] → verify → repair or advance → checkpoint
```

The [Chinese README](../README.md) is the main project page.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run lha run data/tasks/fix_average.yaml
LHA_RUNS_DIR=runs/_quickstart uv run lha eval
```

The bundled example uses a deterministic backend and needs no model credentials
or code index. With an authenticated Codex CLI:

```bash
LHA_CODEX_MODEL=YOUR_MODEL_ID \
LHA_CODEX_EFFORT=medium \
uv run lha --llm codex_cli run data/tasks/fix_average.yaml
```

Inspect a run with `lha runs show <run-id>` or `lha trace <run-id> --html`.

## What is implemented

- Schema-v2 run state stores the cursor, attempts, fixed budgets, elapsed time,
  repair counts, and model usage.
- Checksummed state, an event-chain ledger, atomic replacement, and a run lock
  support fail-closed recovery.
- `ResolvedPatch` computes the real write set. `PatchTransaction` records
  `PREPARED`, `APPLIED`, `VERIFIED`, or `REVERTED` with redundant backups.
- Approval binds the exact patch digest; resume cannot replace reviewed work.
- Pytest, Ruff, experiment reruns, and repository-defined stages return typed
  evidence. A check that cannot run fails.
- Local and Docker backends share timeout, cleanup, and resource-limit handling.
- The optional LangGraph runtime reuses the same execution and verification
  code and stores graph checkpoints in SQLite.
- Indexed context is accessed only through `lha.live_context`, with source
  digests, freshness, partial availability, and explicit failure states.

## Evaluation

### Formal schema-v4 verification ablation

The formal report evaluated `gpt-5.3-codex-spark` on 17 preregistered defects
with 12 repetitions each: 204 scheduled paired cells, 204 usable cells, and
0 ERROR cells. Rates use the 204 usable cells.

| condition | behavior | independently scored outcome |
|---|---|---|
| `trust` | deliver the first patch | 201 delivered-correct; 3 delivered-wrong |
| `gate` | deliver only after the checks | 201 delivered-correct; 3 wrong patches intercepted; 0 delivered-wrong; 0 correct patches rejected |
| `verify` | allow bounded repair after a failed check | 204/204 delivered-correct; 0 delivered-wrong; 0 not delivered |

The task-cluster exact paired sign-flip p-values are 0.2500 for `trust`
versus `gate` on delivered-wrong outcomes and 0.2500 for `trust` versus
`verify` on delivered-correct outcomes. The horizon composition is a
descriptive projection over measured task rates. It adds no observations and
is not an executed shared-state long task.

The report, frozen patches, scorer evidence, model-call receipts, and all cell
results are under [`benchmarks/`](../benchmarks/).

### Terminal-Bench 2.1

The preregistered Terminal-Bench 2.1 fixed 20-task subset produced **7 PASS,
9 FAIL, and 4 ERROR**, reported as **7/20**. Every task ran once and all errors
remain in the denominator. This is not a full-dataset or leaderboard score.

That adapter calls Codex directly inside Harbor. It does not use LHA's gate or
repair loop, so the result does not measure interception or repair behavior.

See [BENCHMARKS.md](BENCHMARKS.md), [ABLATION.md](ABLATION.md), and
[HORIZON.md](HORIZON.md) for protocols and evidence boundaries.

## Scope and limits

- This is a research and portfolio project, not an online service.
- `trusted-local` is only for trusted repositories; it keeps the user's host
  permissions.
- Docker reduces host exposure but still depends on the image, mounts, daemon,
  kernel, and configured permissions.
- Temporary credentials are cleaned after handled exits, not guaranteed after
  `SIGKILL`, a kernel crash, or power loss.
- A durable deadline is checked at state boundaries; each blocking operation
  still needs its own timeout.
- Source freshness and citations are weaker evidence than executable tests.
- Passing fixed tests does not prove correctness for every input.

Implementation details are in [ARCHITECTURE.md](ARCHITECTURE.md); build, package, and container checks are in [DEPLOY.md](DEPLOY.md).
