# AGENTS.md — repository guide for coding agents

These are operational notes for anyone editing this repository. They summarize
the checked implementation; `docs/ARCHITECTURE.md`, `CONTRIBUTING.md`, and
`SECURITY.md` contain the longer explanations.

The repository has one non-negotiable rule:

> **No claim without a runnable check.** Do not publish a behavior, benchmark
> number, test count, or coverage figure until a command in this checkout has
> produced it.

## 1. Project scope

`lha` is a Python 3.11+ task runner for code changes, experiments, and
retrieval-backed work. A run follows:

```
context → execute → [approval] → verify → (repair | advance) → checkpoint
```

A step advances only after its registered checks pass. A check that cannot run
fails; it is never treated as a pass or an implicit skip.

This is a research and portfolio project, not a production service. Its main
implementation boundaries are:

1. The harness owns state transitions, budgets, approval, recovery, and rollback.
2. `lha.live_context` is the only entry point to code and document indexes.
3. The internal gate predicts whether to accept work; an independent scorer
   supplies truth in ablation and public-benchmark adapters.

## 2. Setup and commands

Use [`uv`](https://docs.astral.sh/uv/) from the repository root. Python is pinned
by `.python-version`.

```bash
uv sync
uv run lha run data/tasks/fix_average.yaml
LHA_RUNS_DIR=runs/_scratch uv run lha eval
uv run pytest -q
```

Current CLI surface:

```text
lha run <task.yaml> [--runtime loop|langgraph] [--auto-approve] [--json]
lha resume <run_id> [--runtime loop|langgraph] [--auto-approve] [--json]
lha approve|reject <run_id> [--note TEXT]
lha trace <run_id> [--html] [--out PATH]
lha runs list
lha runs show <run_id>
lha runs prune --older-than-days N [--apply]
lha batch <task.yaml>... [--workers N]
lha eval [--quick]
lha ablate [task.yaml...] [--reps N] [--model MODEL]
           [--scorer-backend trusted-local|docker] [--out DIR]
lha horizon [--from-report PATH] [--out DIR] [--seed N]
lha index <path>
lha index-docs
lha ask <query...> [--root PATH] [--kinds code,paper,...] [--k N]

Global: --llm {stub,claude_cli,codex_cli,anthropic}  -v/-vv  --version
```

Configuration is read once at startup in `src/lha/config.py`. `.env.example`
lists every supported `LHA_*` variable. The settings that most affect behavior
are:

| variable | default | purpose |
|---|---|---|
| `LHA_LLM_BACKEND` | `stub` | deterministic offline backend for tests and self-eval |
| `LHA_MAX_STEPS` / `LHA_MAX_REPAIRS` | `20` / `3` | persisted run budgets |
| `LHA_DEADLINE_S` / `LHA_MAX_LLM_CALLS` | unset | resumable time and call limits |
| `LHA_EXEC_BACKEND` | `trusted-local` | `trusted-local` or `docker` execution |
| `LHA_EXEC_IMAGE` | `python:3.12-slim` | image used by the Docker execution backend |
| `LHA_CODE_BACKEND` | `auto` | `ccc`, `null`, or automatic selection |
| `LHA_RUNS_DIR` / `LHA_DATA_DIR` | `runs` / `data` | durable state locations |
| `LHA_CODEX_MODEL` / `LHA_CODEX_EFFORT` | unset / `medium` | Codex run provenance |

Optional extras are `context`, `bench`, `llm`, and `typecheck`. Harbor requires
Python 3.12 or newer even though the core package supports Python 3.11.

### `uv` pitfalls

- Running `uv run --python X.Y` or `uv run --with ...` inside this project can
  recreate `.venv`. For an isolated package probe, change to a scratch directory
  and use `uv run --no-project`.
- Every benchmark fixture has its own `pyproject.toml`. Run project commands from
  this repository root so `uv` does not select a fixture as the active project.

## 3. Directory map

```text
src/lha/
  harness/        loop, state, checkpoint, approval, manifest, transaction
  live_context/   facade, freshness, backends, packaged CocoIndex flows
  agents/         supervisor, context engineer, implementer, experimenter, verifier
  verifiers/      code, experiment, and context verifier families
  llm/            stub, Claude CLI, Codex CLI, Anthropic, tracing
  sandbox/        trusted-local and Docker execution backends
  runtime/        opt-in LangGraph runner
  bench/          SWE-bench and Terminal-Bench adapters, statistics
  tasks/ tools/   task models, patch resolution, policy, shell helpers
  reporting.py    validated inspection, static HTML, run retention
  repo_adapter.py typed repository stages for long tasks
data/
  tasks/          normal tasks and the 17 fixed ablation tasks
  bench/          planted-bug repositories and their pytest oracles
  long_tasks/     five fixed multi-file repositories and reference evidence
tests/            hermetic unit, integration, recovery, and packaging checks
benchmarks/       committed measured reports; regenerate, never hand-edit numbers
runs/<id>/        state, ledger, transactions, artifacts, reports, worktree
```

Generated state is ignored by Git: `runs/`, `data/.lha_index/`, `data/skills/`,
`.cocoindex_code/`, caches, coverage output, and build output.

## 4. Runtime and recovery

`Harness.run` copies `task.target_repo` into a per-run worktree and creates
schema-v2 `RunState`. The state persists the cursor, attempts, repair counters,
the original step/repair/deadline/model-call limits, their consumption, and model
usage. Resume rejects any change to those four limits. `state.json` is a
checksummed envelope written with `fsync` and atomic replacement;
`ledger.jsonl` is append-only.

Code edits use two typed values:

- `ResolvedPatch` computes the write set from the actual diff or file contents.
  Policy, backup, apply, approval, manifest, and rollback use this same set.
- `PatchTransaction` records `PREPARED`, `APPLIED`, `VERIFIED`, or `REVERTED`.
  Recovery validates the patch, manifest, transaction journal, and redundant
  backups before replaying or rolling back.

A per-run file lock rejects concurrent resume. Stable attempt IDs and ledger
idempotency keys prevent duplicate completion and approval events. State schema
v1 remains inspectable but is not resumed as schema v2.

The LangGraph runtime uses the same execute and verification helpers. Its
prepare, approval interrupt, and verify nodes are separate so resume cannot
regenerate an artifact after a person reviewed it.

## 5. Long-task fixtures

`data/long_tasks/` contains five pre-fixed multi-file cases:

- configuration parsing and precedence;
- SQLite migration and persistence;
- concurrent update and exception propagation;
- CLI stdout/stderr/exit-code contracts;
- seeded experiment and artifact digests.

Each case has `task.yaml`, `adapter.yaml`, a repository, a reference patch, and
a reference manifest with source and oracle digests. The Supervisor emits a
fixed 10-step plan: integrity, setup, baseline, reproduction, context, approved
edit, targeted tests, full tests, lint, and build.

Tests exercise an initial rejected patch, repair, two approval resumptions, a
process interruption at a safe boundary, and equality with an uninterrupted
terminal state. A repository stage that may have started but lacks durable
completion evidence fails closed instead of replaying a possible side effect.

Reference patches and their oracles are corpus evidence. Do not edit them to
improve a result.

## 6. Codex CLI backend

`src/lha/llm/codex_cli.py` runs `codex exec --json` in a temporary home and
workspace. Authentication is copied into the attempt-local `CODEX_HOME`; parent
secrets are not inherited. The process runs in its own group, descendants are
terminated on timeout or interruption, and temporary credentials and files are
removed on every exit path.

The JSONL parser fails closed on malformed JSON, unknown events, incomplete
turns, error events, and unfinished or disallowed tool use. Provenance records
the selected model, reasoning effort, CLI version, event summary, usage, and
outcome. In the no-tools ablation path, any tool item invalidates the attempt.

Do not log `auth.json`, API keys, session cookies, or direct credential paths.

## 7. Verification and statistics

Registered verifier families are:

| family | checks | source of evidence |
|---|---|---|
| code | pytest, Ruff, repository stages | real subprocess output |
| experiment | PSNR, SSIM, reproducibility | arrays, hashes, and a fresh rerun |
| context | freshness, citation | source digests and resolvable locators |

Experiment reruns use fresh directories and reject missing, stale, non-finite,
or digest-mismatched arrays. Context bundles distinguish `empty`,
`backend_unavailable`, `index_failed`, and stale or partially unavailable kinds.

`lha ablate` pairs the same first attempt under `trust`, `gate`, and `verify`.
The internal gate is a prediction. Truth comes from applying the frozen source
change to a fresh canonical repository and running an independent scorer.

`lha horizon` keeps three units separate:

1. paired `(task, repetition)` cells;
2. observed whole-corpus repetitions;
3. a descriptive independent-step composition.

Cell- and episode-level McNemar tests may differ. Composition adds no independent
samples and has no McNemar p-value. Boundary proportions use Wilson intervals.
Do not replace them with a percentile bootstrap that reports a zero-width
interval for all-zero or all-one samples.

### Current measured baseline

The committed schema-v2 report fixes the protocol at 17 tasks × 12 repetitions,
Codex CLI 0.141.0, `gpt-5.4-mini`, low reasoning effort, read-only mode, and a
Docker independent scorer. All 204 paired cells have truth labels; there are
zero `ERROR` cells.

| condition | independently correct | delivery decision |
|---|---:|---|
| `trust` | 194/204 | delivered all 204, including 10 incorrect patches |
| `gate` | 194/204 | accepted 194 correct patches and rejected 10 incorrect patches |
| `verify` | 204/204 | repaired the 10 rejected attempts, then passed independent scoring |

For `trust` versus `verify`, the exact two-sided McNemar result is
`p = 0.00195` (`10` versus `0` discordant cells). At the observed
whole-corpus level, `trust` completes 2/12 episodes and `verify` completes
12/12. The composition curve adds zero independent samples and is only a
model-based projection. The raw source is `benchmarks/ablation_report.json`;
never reconstruct these numbers from prose.

## 8. Reporting and retention

```bash
uv run lha trace <run_id>
uv run lha trace <run_id> --html
uv run lha runs list
uv run lha runs show <run_id>
uv run lha runs prune --older-than-days 30
```

The HTML trace is a self-contained rendering of steps, patches, approvals,
verdicts, repairs, and recorded model usage. Reporting validates persisted
evidence and refuses damaged runs. Pruning is a dry run unless `--apply` is
present, deletes only validated `DONE` or `FAILED` runs, and refuses locked or
corrupt state.

## 9. Required gate

Run the release gate from the repository root:

```bash
uv run ruff check .
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
```

Also install both `dist/*.whl` and `dist/*.tar.gz` from scratch directories with
`uv run --no-project --with ...`, then import
`lha.live_context.flows.common`. `.github/workflows/ci.yml` contains the exact
package and container smoke checks.

Do not write a final test count, coverage percentage, ablation result, or public
benchmark score into docs until these commands have run on the release candidate.
For the current release candidate, the measured local baseline is
`523 passed, 3 skipped`, 83% statement coverage, and `lha eval` at 6/6.

## 10. Coding rules

- Ruff line length is 100 with import sorting enabled; Pyright targets Python 3.11.
- Put `from __future__ import annotations` at the top of Python modules.
- Use Pydantic models for boundary data and dataclasses for internal value objects.
- Use `lha.clock.now()` for timestamps.
- Route target/model-influenced subprocesses through `ExecutionBackend`.
- Import optional dependencies inside the function that needs them.
- Comments explain the failure mode a decision prevents, not the obvious syntax.
- Use conventional commit subjects: `feat:`, `fix:`, `docs:`, `test:`, `ci:`,
  `chore:`, or `refactor:`.
- Add a registered verifier rather than a special case in the harness loop.

## 11. Prohibited changes

- Never skip, delete, weaken, or mark a test `xfail` to make the gate green.
- Never turn “could not verify” into success.
- Never import CocoIndex or execute `ccc` outside `src/lha/live_context/`.
- Never let an internal gate verdict serve as ablation ground truth.
- Never edit the ablation or long-task corpus after observing model output.
- Never commit generated run, index, cache, coverage, or nested fixture-lock state.
- Never publish a benchmark number without its raw report and provenance.
- Never store Codex, Anthropic, GitHub, SSH, or cloud credentials in the repository,
  a container image, an artifact, or a log.

## 12. Known limits

- `trusted-local` scrubs the environment and manages process groups, but it is not
  isolation against hostile code. Use Docker for external repositories.
- The default Docker execution image does not include pytest or Ruff; supply a
  task image containing every required tool.
- Prompt injection from indexed content is reduced by objective checks, not
  eliminated.
- Context freshness and citations are weaker evidence than an executable oracle.
- Public benchmark adapters are not leaderboard results. No Terminal-Bench or
  SWE-bench score is claimed until an official run is completed and committed.
- A horizon composition is a projection over measured cells, not an additional
  long-task experiment.
