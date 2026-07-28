# AGENTS.md

This file records the repository rules that matter when editing or evaluating
LHA. Longer explanations are in `docs/ARCHITECTURE.md`, `CONTRIBUTING.md`, and
`SECURITY.md`.

Do not publish a behavior, test count, coverage value, or benchmark number
until a command in the current checkout has produced it.

## Scope

LHA is a Python 3.11+ runner for code changes, experiments, and indexed
context. A run follows:

```text
context → execute → [approval] → verify → repair or advance → checkpoint
```

The runner owns transitions, budgets, approval, recovery, and rollback. A check
that cannot run fails.

Implementation boundaries:

1. `lha.live_context` is the only entry point to code and document indexes.
2. Target- or model-influenced commands use `ExecutionBackend`.
3. The internal gate decides whether a run advances. An independent scorer
   supplies truth labels for the ablation and public benchmarks.
4. This is a research and portfolio project, not a production service.

## Setup

Run project commands from the repository root with `uv`:

```bash
uv sync
uv run lha run data/tasks/fix_average.yaml
LHA_RUNS_DIR=runs/_scratch uv run lha eval
uv run pytest -q
```

Useful CLI commands:

```text
lha run <task.yaml> [--runtime loop|langgraph] [--auto-approve] [--json]
lha resume <run_id> [--runtime loop|langgraph] [--auto-approve] [--json]
lha approve|reject <run_id> [--note TEXT]
lha trace <run_id> [--html] [--out PATH]
lha runs list|show|prune ...
lha batch <task.yaml>... [--workers N]
lha eval [--quick]
lha ablate [task.yaml...] [--reps N] [--model MODEL]
lha horizon [--from-report PATH] [--out DIR] [--seed N]
lha index <path>
lha index-docs
lha ask <query...> [--root PATH] [--kinds code,paper,...] [--k N]
```

Global options include
`--llm {stub,claude_cli,codex_cli,anthropic}`, `-v`, `-vv`, and `--version`.

Configuration is loaded once in `src/lha/config.py`. `.env.example` lists the
supported `LHA_*` variables. Important defaults:

| setting | default |
|---|---|
| `LHA_LLM_BACKEND` | `stub` |
| `LHA_MAX_STEPS` / `LHA_MAX_REPAIRS` | `20` / `3` |
| `LHA_DEADLINE_S` / `LHA_MAX_LLM_CALLS` | unset |
| `LHA_EXEC_BACKEND` | `trusted-local` |
| `LHA_EXEC_IMAGE` | `python:3.12-slim` |
| `LHA_CODE_BACKEND` | `auto` |
| `LHA_RUNS_DIR` / `LHA_DATA_DIR` | `runs` / `data` |
| `LHA_CODEX_MODEL` / `LHA_CODEX_EFFORT` | unset / `medium` |

Optional extras are `context`, `bench`, `llm`, and `typecheck`. Harbor requires
Python 3.12 or newer even though LHA itself supports Python 3.11.

Do not run project commands from a benchmark fixture; each fixture has its own
`pyproject.toml`. For an isolated package probe, change to a scratch directory
and use `uv run --no-project`.

## Repository map

```text
src/lha/
  harness/        loop, state, checkpoint, approval, manifest, transaction
  live_context/   facade, freshness, backends, packaged index flows
  agents/         planning, context, implementation, experiments, verification
  verifiers/      code, experiment, and context checks
  llm/            stub, CLI and SDK backends, tracing
  sandbox/        trusted-local and Docker execution
  runtime/        optional LangGraph runner
  bench/          ablation, SWE-bench, Terminal-Bench, statistics
  tasks/ tools/   task models, patch resolution, policy, command helpers
  reporting.py    run inspection, HTML reports, retention
  repo_adapter.py typed repository stages
data/
  tasks/          normal tasks and 17 fixed ablation tasks
  bench/          fixed defect repositories and their oracles
  long_tasks/     five fixed multi-file fixtures
tests/            unit, integration, recovery, and packaging checks
benchmarks/       committed generated reports
runs/<id>/        generated state, evidence, worktree, and reports
```

Do not commit `runs/`, indexes, caches, coverage output, build output, or nested
fixture lock files.

## Runtime and recovery

`Harness.run` copies the target repository into a per-run worktree and creates
schema-v2 `RunState`. State stores the cursor, attempts, repair counters, original
limits, elapsed time, and model usage. Resume rejects limit drift.

`state.json` is checksummed and atomically replaced after `fsync`.
`ledger.jsonl` is logically append-only: each update validates the existing
event chain and atomically replaces the complete bytes rather than relying on
an operating-system `O_APPEND` write. A run lock rejects concurrent resume.
Stable attempt IDs and idempotency keys prevent duplicate completion and
approval events. Schema-v1 runs can be inspected but not resumed as schema v2.

`ResolvedPatch` derives the write set from the real diff or file contents.
Policy, backup, apply, approval, manifest, and rollback use that set.

`PatchTransaction` records `PREPARED`, `APPLIED`, `VERIFIED`, or `REVERTED`.
Recovery validates patch bytes, manifests, journals, and redundant backups
before applying, accepting, or reverting anything.

The LangGraph runtime uses the same execution and verification helpers. Prepare,
approval interrupt, and verify are separate nodes so resume cannot replace a
reviewed artifact.

## Long-task fixtures

`data/long_tasks/` contains five fixed cases for configuration parsing, SQLite
migration, concurrency, CLI contracts, and experiment reproduction. Each has a
task, repository adapter, repository, reference patch, and digests.

The ten stages are integrity, setup, baseline, reproduction, context, approved
edit, targeted tests, full tests, lint, and build. Tests cover a rejected first
patch, repair, approval resumes, injected interruption, and equality with an
uninterrupted result. An adapter defines how a fixture is run; it is not a
benchmark result.

Do not edit a fixture, oracle, or reference patch after model output has been
observed.

## Codex backend

`src/lha/llm/codex_cli.py` runs `codex exec --json` in an attempt-local home and
workspace. It copies only required authentication, starts a separate process
group, stops descendants on normal failure, timeout, or handled interruption,
and then removes temporary credentials. `SIGKILL`, a kernel crash, or power
loss can prevent that cleanup; any surviving directory remains protected by
its file mode and must be inspected manually.

The parser rejects malformed JSONL, unknown events, incomplete turns, error
events, and unfinished or disallowed tool use. The no-tools ablation path
rejects any tool item. Successful provenance includes model settings, CLI
version, event summary, usage, and outcome.

Never log `auth.json`, API keys, session cookies, or credential paths.

## Verification and evaluation

| family | checks | evidence |
|---|---|---|
| code | Pytest, Ruff, repository stages | subprocess output |
| experiment | PSNR, SSIM, reproducibility | arrays, hashes, fresh rerun |
| context | freshness, citations | source digests, status, locators |

Experiment reruns use new directories and reject missing, stale, non-finite, or
mismatched arrays. Context records distinguish empty results, unavailable
backends, failed indexes, stale data, and partial availability.

`lha ablate` shares the first attempt across `trust`, `gate`, and `verify`.
Ground truth comes from a frozen source change applied to a fresh repository and
scored through a separate backend. The committed schema-v2 report is a historical
record; a current result requires a complete schema-v4 report and its evidence.
Read historical numbers from `benchmarks/ablation_report.json` instead of
reconstructing them from prose.

A formal 17-task × 12-repetition run requires a committed `REGISTERED` event
that fixes the source tree, corpus manifest, model, Codex CLI identity, client
settings, Docker image, output path, and witness remote. The registration
commit directly follows the source commit and changes only the registry.
Before the first cell, the runner creates a new remote witness ref bound to the
registration and run header. Formal cells never read cache and a formal run
cannot resume. Interruption consumes the attempt: record `ABANDONED` and do not
repeat the same outcome-affecting selection.

`lha horizon` keeps paired cells, complete-corpus repetitions, and descriptive
composition separate. Cell and episode tests can differ. Composition adds no
samples and has no McNemar p-value.

For Terminal-Bench 2.1:

- preregister task IDs before model execution;
- run three smoke tasks before the fixed 20-task scored subset;
- use one task, one attempt, and zero Harbor retries per job;
- keep task and protocol failures in the denominator;
- record model settings, versions, hashes, image digests, and official results;
- do not report direct-Harbor runs as evidence for LHA gate or repair behavior;
- do not publish a score until protocol, manifest, raw results, and summary are
  committed together.

The formal fixed-subset run is complete: 7 `PASS`, 9 `FAIL`, and 4 `ERROR`
across 20 tasks. All four errors remain in the denominator. The evaluated source
commit is `e63f94620ce8ddd322b19ccb159381183fc31933`; its public schema-v4
evidence is under `benchmarks/terminal_bench_2_1/`. Do not rerun any of these
20 scored tasks or describe the result as a full-dataset or leaderboard score.

## Reporting and retention

```bash
uv run lha trace <run_id>
uv run lha trace <run_id> --html
uv run lha runs list
uv run lha runs show <run_id>
uv run lha runs prune --older-than-days 30
```

Reporting validates saved evidence. Pruning is a dry run unless `--apply` is
present and refuses active, locked, unfinished, or corrupt runs.

## Required release gate

```bash
uv run ruff check .
uv run python -m lha.release_claims
uv run python tools/verify_terminal_source_build.py \
  --root . \
  --evidence benchmarks/terminal_bench_2_1
uv run pyright src/lha
uv run pytest -q
LHA_RUNS_DIR=runs/_release uv run lha eval

if grep -rnE "^[[:space:]]*(import|from)[[:space:]]+(cocoindex|cocoindex_code)" \
     --include='*.py' src/lha | grep -v "src/lha/live_context/"; then
  exit 1
fi

uv build
docker build -t lha:release .
LHA_DOCKER_TESTS=1 LHA_DOCKER_TEST_IMAGE=lha:release \
  uv run pytest tests/test_sandbox.py -q
docker run --rm lha:release lha --version
docker run --network none --rm lha:release lha eval
```

Also install the wheel and source archive from empty directories and import
`lha.live_context.flows.common`. `.github/workflows/ci.yml` contains the exact
package and container smoke checks.

## Coding rules

- Ruff line length is 100; Pyright targets Python 3.11.
- Put `from __future__ import annotations` at the top of Python modules.
- Use Pydantic models for boundary data and dataclasses for internal values.
- Use `lha.clock.now()` for timestamps.
- Import optional dependencies inside the function that needs them.
- Explain failure modes in comments instead of restating syntax.
- Use conventional commit subjects.
- Add a registered verifier instead of a special case in the main loop.

## Prohibited changes

- Do not skip, weaken, delete, or mark a test `xfail` to make checks pass.
- Do not turn “could not verify” into success.
- Do not import CocoIndex or run `ccc` outside `src/lha/live_context/`.
- Do not use an internal gate decision as benchmark ground truth.
- Do not edit ablation or long-task corpora after observing model output.
- Do not publish a number without its raw report and provenance.
- Do not store model, GitHub, SSH, or cloud credentials in the repository,
  image, artifact, or log.

## Known limits

- `trusted-local` is not isolation against hostile code; target processes run
  with the current user's host permissions.
- The default Docker execution image does not contain Pytest or Ruff.
- `LHA_DEADLINE_S` is a persisted boundary check, not asynchronous preemption
  of a blocking operation; each such operation needs its own timeout.
- A forced stop during the first write to a write-once artifact can leave an
  incomplete final file. Atomic replacement can leave a mode-protected
  temporary file. Recovery validates known transaction temporaries and
  otherwise stops for manual review; it does not infer missing bytes.
- Temporary Codex credentials are cleaned on handled exit paths, not after
  `SIGKILL`, a kernel crash, or power loss.
- Checks reduce, but do not eliminate, prompt injection from indexed content.
- Source freshness and citation checks are weaker than executable oracles.
- An adapter alone is not a benchmark result; a result requires the committed
  protocol, raw evidence, provenance, and summary.
- A horizon composition is a projection, not an executed long task.
