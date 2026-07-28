# LHA: execution, recovery, and verification for coding tasks

LHA is a Python runner for code changes, experiments, and indexed context. The
runner manages task state, approvals, checks, bounded repair, and rollback:

```text
context → execute → [approval] → verify → repair or advance → checkpoint
```

The [Chinese README](https://github.com/liuqjjin/long-horizon-agent/blob/main/README.md)
is the main project page.

## Quick start

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run lha run data/tasks/fix_average.yaml
LHA_RUNS_DIR=runs/_quickstart uv run lha eval
```

The default backend is deterministic and does not require model credentials.
For a locally authenticated Codex CLI:

```bash
LHA_CODEX_MODEL=YOUR_MODEL_ID \
LHA_CODEX_EFFORT=medium \
uv run lha --llm codex_cli run data/tasks/fix_average.yaml
```

Inspect a saved run:

```bash
RUN_ID=replace-with-the-run-id
uv run lha runs show "$RUN_ID"
uv run lha trace "$RUN_ID"
uv run lha trace "$RUN_ID" --html
```

## Implementation

- Schema-v2 run state stores the cursor, attempt IDs, repair counters, fixed
  budgets, elapsed time, and model usage.
- `state.json` is checksummed and replaced atomically after `fsync`.
  `ledger.jsonl` grows logically by event; each update validates and atomically
  replaces the complete file rather than using `O_APPEND`.
- A per-run file lock rejects concurrent resume. Stable idempotency keys prevent
  duplicate completion and approval events.
- `ResolvedPatch` derives the write set from the persisted patch, rather than
  trusting model-supplied file metadata.
- `PatchTransaction` records `PREPARED`, `APPLIED`, `VERIFIED`, or `REVERTED`
  together with manifests and redundant backups.
- Approval records bind the step and the SHA-256 digest of the reviewed patch.
- Code checks run Pytest, Ruff, or repository-defined commands. Experiment
  checks recompute saved arrays and repeat the run in a fresh directory.
- Target- or model-influenced commands use the `ExecutionBackend` interface.
- The optional LangGraph runtime shares the same execution and verification
  helpers and stores graph checkpoints in SQLite.
- Indexed code and documents are accessed through `lha.live_context`, which
  records source digests, freshness, and unavailable states.

## Fixed multi-file tasks

`data/long_tasks/` contains five fixed cases: configuration precedence, SQLite
migration, concurrent updates, command-line contracts, and experiment
reproduction. Each case includes a task definition, repository adapter,
reference patch, and source and oracle digests.

The cases use ten stages from integrity checks and problem reproduction through
approved editing, targeted tests, full tests, lint, and build. Tests cover an
initial rejected patch, repair, approval resume, process interruption, and an
equivalent terminal state after recovery.

An adapter defines repository setup and check stages. It is not a benchmark
result without a fixed protocol, raw outcomes, provenance, and a committed
summary.

## Measured evaluation

### Terminal-Bench 2.1

The formal fixed-subset run produced **7 PASS, 9 FAIL, and 4 ERROR** over 20
tasks. All `ERROR` results remain in the denominator, so the reported result is
**7/20**.

The run used `gpt-5.5` with `xhigh` reasoning effort, Harbor `0.20.0`, and Codex
CLI `0.141.0`. Each task ran once with Harbor retries disabled. Two errors came
from adapter defects and two from explicit Codex error events. The 20 tasks were
not rerun after those outcomes were observed.

This is a preregistered fixed 20-task subset, not a full leaderboard result.
The adapter calls Codex directly and does not use LHA's gate or repair loop, so
the result is evidence about that model run rather than LHA verification
behavior.

The committed schema-v4 package contains official raw JSON for the 16 `PASS`
and `FAIL` tasks. The four `ERROR` records are public redacted projections bound
to the private originals by SHA-256; private exception traces cannot be
reconstructed from the repository. Protocol details and source attestation are
in [BENCHMARKS.md](BENCHMARKS.md).

### Verification ablation

The repository retains a schema-v2 ablation report from an earlier protocol.
The formal schema-v4 rerun has not completed. The earlier figures are therefore
historical evidence, not the current project result.

The ablation uses a separate Docker scorer that applies the frozen source change
to a fresh repository and runs the fixed tests. It never treats the internal
gate decision as ground truth. See [ABLATION.md](ABLATION.md) for the protocol.

A formal run first commits a registration that fixes the source, corpus, model,
Codex CLI and client settings, Docker image, output path, and witness remote.
At startup it creates an attempt-specific remote Git ref. Formal cells do not
read cache, and an interrupted attempt is recorded as abandoned rather than
resumed or repeated with the same outcome-affecting inputs. No final schema-v4
counts are published before the full registered schedule and its evidence are
committed.

## Limits

- This is a research and portfolio project, not an online service.
- `trusted-local` is not a sandbox for hostile code; target processes keep the
  current user's host permissions.
- Docker isolation depends on the image, mounts, network, and resource limits.
- `LHA_DEADLINE_S` is checked at durable boundaries; blocking operations still
  need their own timeout.
- Temporary credential cleanup runs after normal return, failure, timeout, and
  handled interruption, but not necessarily after `SIGKILL`, a kernel crash, or
  power loss.
- A forced stop during the first write to a write-once artifact can leave an
  incomplete final file. Atomic replacement can leave a restrictive temporary
  file; inconsistent evidence stops recovery.
- Freshness and citation checks provide weaker evidence than executable tests.
- Passing the registered tests does not prove correctness for every input.
- A fixed benchmark subset is not a full benchmark score.

Build and package checks are in [DEPLOY.md](DEPLOY.md). The system structure is
described in [ARCHITECTURE.md](ARCHITECTURE.md).
