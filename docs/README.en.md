# LHA: resumable execution and verification for coding tasks

LHA runs code changes, experiments, and retrieval-backed tasks as a state
machine. A step advances only after its executable checks pass. Failed checks
feed a bounded repair attempt; interrupted runs resume from durable evidence.

The Chinese [README](../README.md) is the primary project page. This file is a
compact English reference.

## Problem

A model can return a plausible patch that is still wrong. In a multi-step task,
that error becomes input to every later step. LHA separates:

- generation: a model proposes work;
- acceptance: the internal gate decides whether the run may advance;
- measurement: an independent scorer decides whether a benchmark delivery is
  actually correct.

The scorer does not reuse the gate's verdict. It applies the frozen source
change to a fresh canonical repository and runs the original tests through a
separate execution backend.

## Runtime

```text
plan → context → execute → [approval] → verify → repair or advance → checkpoint
```

The main recovery properties are:

- `RunState` schema v2 persists cursor, attempts, repair counters, time/step/call
  budgets, and model usage.
- `state.json` is checksummed and written with `fsync` plus atomic replacement;
  `ledger.jsonl` is append-only.
- A run lock rejects concurrent resume of the same `run_id`.
- Stable attempt IDs and ledger idempotency keys prevent duplicate transitions.
- `ResolvedPatch` derives the write set from the real patch, not model metadata.
- `PatchTransaction` records `PREPARED`, `APPLIED`, `VERIFIED`, and `REVERTED`
  with durable patch, manifest, journal, and redundant backup evidence.
- Approval binds the step and SHA-256 of the exact reviewed `patch.json`.
- Unverified, rejected, corrupt, or exhausted changes are rolled back.

The default runtime is implemented directly by `Harness`. The optional
LangGraph runtime uses the same execution and verification helpers with SQLite
checkpointing and a separate approval interrupt node.

## Checks

| task family | checks |
|---|---|
| code | real pytest and Ruff commands, plus typed repository stages |
| experiment | PSNR/SSIM recomputed from saved arrays and a fresh rerun |
| context | source freshness, backend status, and resolvable citations |

A check that cannot run fails. Experiment evidence rejects stale, missing,
non-finite, or digest-mismatched arrays. Context distinguishes empty results from
an unavailable backend, failed index, stale source, and partial kind
availability.

Target or model-influenced commands use one `ExecutionBackend` interface.
`trusted-local` is for trusted development code; Docker is the isolation boundary
for external repositories and independent scoring.

## Long repository tasks

Five fixed multi-file cases live under `data/long_tasks/`:

- configuration precedence and environment parsing;
- transactional SQLite migration;
- concurrent updates and worker exception propagation;
- CLI stdout, stderr, JSON, and exit-code contracts;
- seeded experiment reproduction and artifact digests.

Each case has a fixed repository digest, oracle digest, reference patch, and a
10-step plan covering integrity, setup, baseline, reproduction, context,
approved editing, targeted tests, full tests, lint, and build. Integration tests
exercise a rejected first patch, repair, two approval resumes, process
interruption, and equality with an uninterrupted terminal state.

## Codex CLI backend

The Codex backend runs `codex exec --json` with an attempt-local home, workspace,
and temporary directory. It copies only required authentication, passes a
minimal environment, terminates the process group on timeout or interruption,
and removes temporary credentials on every exit path.

Its event parser rejects malformed JSONL, unknown events, error events,
incomplete turns, and unfinished tool calls. The no-tools ablation path rejects
any tool item. Successful provenance records the configured model, reasoning
effort, CLI version, event summary, usage, and outcome without recording
credential bytes.

## Run it

```bash
uv sync
uv run lha run data/tasks/fix_average.yaml
LHA_RUNS_DIR=runs/_quickstart uv run lha eval
```

Inspect a completed or paused run:

```bash
uv run lha runs show <run_id>
uv run lha trace <run_id>
uv run lha trace <run_id> --html
```

The HTML file is self-contained and includes the timeline, patches, approvals,
verdicts, repairs, and model-usage totals. Safe retention is dry-run by default:

```bash
uv run lha runs prune --older-than-days 30
```

## Ablation and horizon reports

`lha ablate` pairs one first attempt under `trust`, `gate`, and `verify`, then
grades deliveries with the independent scorer. The report preserves error cells,
gate confusion matrices, source and runtime provenance, and a fingerprint used
for safe cache reuse.

Rate intervals use a task-cluster bootstrap in the interior and Wilson score
intervals at all-zero or all-one boundaries. Paired contrasts use the exact
McNemar test.

`lha horizon` keeps three quantities separate:

1. paired `(task, repetition)` cells;
2. observed complete-corpus repetitions;
3. a descriptive independent-step composition.

Cell and episode tests use different units and may have different p-values. The
composition adds no independent samples and reports no McNemar p-value.

### Committed schema-v2 result

The committed run used Codex CLI 0.141.0 with `gpt-5.4-mini`, `low` reasoning,
and the read-only sandbox. It evaluated 17 fixed Python defects over 12
repetitions: 204 paired `(task, repetition)` cells. The final scorer ran the
canonical tests in Docker on fresh repository copies, independently of the
internal gate. The report contains zero `ERROR` cells.

| condition | independently scored result |
|---|---|
| `trust` | 194 correct deliveries; 10 incorrect deliveries accepted |
| `gate` | 194 correct attempts accepted; all 10 incorrect attempts blocked |
| `verify` | 204/204 correct after the bounded repair loop |

For the 204 paired cells, the exact two-sided McNemar value is
`p = 0.001953125` (displayed as `0.00195`). Grouping all 17 tasks in a
repetition into one observed episode gives 2/12 complete successes for `trust`
and 12/12 for `verify`.

The separate composition curve is a descriptive projection from measured
per-task rates. It adds zero independent samples and is not evidence from an
additional long-task run. See the generated
[ablation report](../benchmarks/ablation_report.md), its
[schema-v2 JSON](../benchmarks/ablation_report.json), and the
[horizon report](../benchmarks/horizon_report.md).

## Build checks

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

The current release candidate produced `523 passed, 3 skipped`, 83% statement
coverage, and 6/6 self-evaluation cases.

The wheel and source distribution must also install from empty directories. See
[DEPLOY.md](DEPLOY.md) for the exact package and container smoke checks.

## Limits

- This is a research and portfolio project, not a production service.
- `trusted-local` is not a hostile-code sandbox.
- Prompt injection from indexed content is mitigated by checks, not eliminated.
- Context freshness and citation checks are weaker than an executable oracle.
- The internal corpus is not a public leaderboard.
- Public benchmark adapters are not benchmark results; no Terminal-Bench or
  SWE-bench score is claimed until an official run is completed and committed.
- A composed horizon curve is not a new long-task experiment.

See [ARCHITECTURE.md](ARCHITECTURE.md),
[ABLATION.md](ABLATION.md), and [HORIZON.md](HORIZON.md) for details.
