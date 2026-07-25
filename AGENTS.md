# AGENTS.md — repository guide for coding agents

Authoritative working notes for Cursor / Codex / any agent editing this repo.
Everything here was checked against the code and against commands that were
actually run. There is no `CLAUDE.md` in this repository, so nothing here
overrides one; `docs/ARCHITECTURE.md` and `CONTRIBUTING.md` remain the prose
sources and this file is the operational summary.

The repo's own rule applies to you as much as to a human contributor:

> **No claim without a runnable check.** To state a behavior, number, or
> benchmark result, run it and quote the real output.

## 1. What this project is

`lha` (package `lha`, distribution name `lha`, Python 3.11+) is a
**verification-first long-horizon agent harness**. Every step of an agent run is
gated on an *objective oracle* — a real `pytest` run, an image metric recomputed
from the output arrays, a reproducibility re-run, an index-freshness check — and
the loop only advances when the oracle passes. A step that cannot be verified
**fails**; it never passes by default.

It is a research/portfolio project, not a production service. The three things
that make it what it is:

1. **The loop**: `context → execute → [approval gate] → verify → (repair | advance) → checkpoint`.
2. **The facade**: everything indexed (code / papers / experiments / skills) is
   reachable only through `lha.live_context`.
3. **Prediction ≠ truth**: wherever the harness grades itself (`lha ablate`, the
   bench adapters), the internal gate only *predicts*; an independent scorer or
   an official third-party harness supplies the *truth*.

## 2. Environment and commands

Package manager is [`uv`](https://docs.astral.sh/uv/); Python is pinned by
`.python-version` (3.11). **Always run from the repo root** — several commands
resolve `data/...` relative to the cwd.

```bash
uv sync                                       # install (dev group incl. context extra)
uv run lha run data/tasks/fix_average.yaml    # verified fix of a planted bug, no API key
uv run lha eval                               # self-eval across the five workflows -> 5/5
uv run lha eval --quick                       # the three fast cases only
uv run pytest -q                              # unit suite
```

CLI surface (`src/lha/cli.py`):

```
lha run <task.yaml> [--runtime loop|langgraph] [--auto-approve] [--json]
lha resume <run_id> [--runtime …]      lha approve|reject <run_id> [--note …]
lha trace <run_id>                     lha batch <task.yaml>… [--workers N]
lha eval [--quick]                     lha ablate [task.yaml…] [--reps N] [--model M] [--scorer-backend trusted-local|docker]
lha horizon [--from-report PATH] [--out DIR] [--seed S]   # compounding curve; no model calls
lha index <path>                       lha index-docs        lha ask <query…> [--root R] [--kinds code,paper,…] [--k N]
Global: --llm {stub,claude_cli,anthropic}   -v/-vv   --version
```

Configuration is environment variables read once at startup
(`src/lha/config.py`); `README.md` has the full table. The ones that change
agent behavior most:

| variable | default | effect |
|---|---|---|
| `LHA_LLM_BACKEND` | `stub` | `stub` is deterministic and offline — keep it for tests/eval |
| `LHA_CLAUDE_MODEL` | – | pin a full model snapshot; a floating alias breaks reproducibility |
| `LHA_MAX_STEPS` / `LHA_MAX_REPAIRS` | 20 / 3 | loop budgets |
| `LHA_DEADLINE_S` / `LHA_MAX_LLM_CALLS` | unset = unlimited | pause (resumable) instead of burning time/tokens |
| `LHA_EXEC_BACKEND` / `LHA_EXEC_IMAGE` | `trusted-local` / `python:3.12-slim` | where target code executes |
| `LHA_CODE_BACKEND` | `auto` | `ccc` \| `null` \| `auto` |
| `LHA_RUNS_DIR` / `LHA_DATA_DIR` | `runs` / `data` | state locations |

Optional extras: `context` (cocoindex + sentence-transformers, needed for
paper/experiment/skill search), `bench` (swebench, harbor — harbor needs
Python ≥ 3.12), `llm` (anthropic SDK), `typecheck` (pyright).

**`uv` traps that have already cost a session:**

- `uv run --python X.Y …` or `uv run --with pkg …` *inside this project* deletes
  and recreates `.venv` at that Python. For a throwaway probe always use
  `uv run --no-project` from a scratch directory; if the swap happened,
  `uv sync` restores the pinned env.
- `uv` treats the *nearest* `pyproject.toml` as the project. The bench fixtures
  under `data/bench/*` each have one, so a `uv run` issued from inside a fixture
  directory silently targets that fixture (creating stray `.venv`/`uv.lock`, and
  exiting 0 as a no-op). Pin the cwd explicitly before any `uv run`.

## 3. Directory map

```
src/lha/
  harness/        loop · state · checkpoint · budget · approval · manifest · errors
  live_context/   facade + models + freshness + backends/   <- the ONLY door to indexers
  agents/         supervisor · context_engineer · implementer · experimenter · verifier_agent
  verifiers/      base · registry · verdict · code/ · experiment/ · context/
  llm/            base · stub · claude_cli · anthropic_client · trace (budget + per-call log)
  sandbox/        ExecutionBackend: trusted-local · docker   <- the execution seam
  bench/          swebench · terminal_bench · stats          <- public-benchmark adapters
  runtime/        langgraph_runner                           <- opt-in durable runtime
  tasks/ tools/   task specs · policy (protected oracle paths) · patch/shell helpers
  ablation.py  horizon.py  eval.py  memory.py  orchestrator.py  cli.py  config.py  clock.py  artifacts.py
flows/            papers · experiments · skills CocoIndex apps (imported only by coco_flow)
tests/            23 test modules + conftest; hermetic (stub LLM, null code backend, tmp_path)
data/tasks/       task specs: fix_average(_approval) · run_sr_experiment(_strict) · bench_*.yaml (17)
data/bench/<n>/   17 fixture repos: one planted bug + a pytest oracle each
data/sample_repo/ toy off-by-one bug (the stub's scripted fix target)
data/sample_experiment/  deterministic bicubic 4x SR baseline
benchmarks/       committed ablation snapshot (json + md)
docs/             ARCHITECTURE · VERIFICATION_FIRST · ABLATION · BENCHMARKS · QUICKSTART · DEPLOY · demo
runs/<id>/        state.json · ledger.jsonl · plan · patch · verify.json · steps/<id>/ · backups/ · llm_trace.jsonl · graph.sqlite · workdir/
```

Generated and gitignored, safe to delete and rebuild: `runs/`,
`data/.lha_index/`, `data/skills/`, `.cocoindex_code/`, `.venv`, caches.

## 4. Architecture and data flow

**One run.** `Harness.run` (`src/lha/harness/loop.py`) copies `task.target_repo`
into `runs/<id>/workdir/`, plans once via `Supervisor`, then per step:

1. `ContextEngineer.gather` → `ContextBundle` (items + provenance + freshness +
   `status`). On a repair step the bundle's code items are overlaid from the
   *current* sandbox, so the second attempt reasons over the failing state.
2. `Harness._execute` dispatches on `step.action`:
   `gather_context`/`answer_query` → the bundle itself; `edit_code` →
   `Implementer` → `Patch` → **policy check** → `ArtifactManifest` →
   `apply_patch` (+ persisted `Backup`); `run_experiment` → `Experimenter` →
   `ExperimentResult`.
3. If `step.requires_approval`: pause `AWAITING_APPROVAL`, binding the request
   to the SHA-256 of the reviewed `patch.json` bytes.
4. `VerifierAgent.verify` runs the step's registered verifiers (concurrently,
   order-preserving) → one `Verdict`.
5. Pass → advance the cursor; fail → re-issue the step as a repair carrying
   `verdict.failures`, or (budget exhausted) revert and fail the run.
6. `save_state` (checksummed envelope) + `append_ledger` (unique `event_id`).

**Artifacts** are pydantic models in `src/lha/artifacts.py` /
`verifiers/verdict.py` / `live_context/models.py`, persisted both flat and under
`runs/<id>/steps/<step_id>/` so a multi-step plan keeps per-step provenance.

**Verifier families** (`select_verifiers(step)` over `verifiers/registry.py`):

| family | verifiers | oracle |
|---|---|---|
| code | `pytest`, `ruff` | real subprocess run on the patched sandbox |
| experiment | `psnr`, `ssim`, `reproducibility` | metrics **recomputed** from saved arrays + a re-run whose `input_sha256` must match |
| context | `freshness`, `citation` | index-vs-source drift; every citation resolves to a bundle locator |

**Durable runtime** (`--runtime langgraph`) drives the same agents through a
`StateGraph` with `SqliteSaver`. It is deliberately split into three nodes —
`prepare` (context + execute, checkpointed *before* the interrupt), `gate` (only
`interrupt()`), `verify` — because LangGraph replays a node from the top on
resume; a single node would re-run the implementer and could apply a patch the
human never saw.

**Ablation** (`lha ablate`) scores one first attempt under `trust` / `gate` /
`verify` (paired). The internal gate predicts; truth comes from freezing the
effective source diff, applying it to a fresh canonical copy (tests restored),
and running pytest through a *separate* execution backend.

## 5. Key invariants — do not break these

These are the load-bearing properties. Each has a test; changing behavior here
means changing the test on purpose, with a reason.

1. **A check that cannot run must fail.** `ruff` exiting 2/124/127, non-JSON on
   stdout, `pytest` collecting 0 tests, an unregistered verifier, an empty
   verifier list, a crashing verifier, an artifact the citation verifier does
   not understand — all produce a *failing* `Check`. Never a pass, never a skip.
2. **Context fails closed.** `ContextBundle.status` distinguishes `ok` / `empty`
   / `backend_unavailable` / `index_failed`, and `unavailable_kinds` records a
   kind that was dark even when another kind returned hits. Steps default to
   `context_requirement="required"`; only an explicit `"optional"` lets a step
   proceed without retrieval. Backends raise `BackendUnavailable` — they must
   never return `[]` to signal failure.
3. **`reject_stale` only clears the stale flag after a verifiably successful
   reindex.** A failed refresh raises `StaleContextError` and the bundle stays
   stale.
4. **A patch may never touch the oracle.** `lha.tools.policy` refuses patches
   touching `tests/`, `test_*.py`, `*_test.py`, `conftest.py`, `pyproject.toml`,
   `setup.py|cfg`, `tox.ini`, `noxfile.py`, `pytest.ini`, `ruff.toml`,
   `.github/`, `.ci/` — case-insensitively, parsed from the diff headers (with
   git C-quoting stripped), not from the patch's self-declared `touched_files`.
   Only `TaskSpec.allowed_protected_files` may authorize an exact path.
5. **An approval binds to bytes.** A decision carries `step_id` *and* the
   SHA-256 of the reviewed `patch.json`. On resume the harness executes those
   exact bytes; a mismatch reverts the change and fails the run. Actions with no
   reviewable patch bind by `step_id` alone (a hash would livelock the gate,
   since they regenerate on every resume).
6. **An unverified change never survives in the sandbox.** Failed verification
   with the repair budget exhausted, an approval rejection, a tampered artifact,
   or an unexpected mid-step fault all revert via the persisted `Backup` — in
   *both* runtimes.
7. **Checkpoints refuse damage.** `state.json` is a `{schema_version, sha256,
   payload}` envelope written fsync + atomic-rename; a checksum mismatch, an
   unreadable file, or a cursor outside the plan raises `CheckpointCorrupt`
   rather than resuming from a guess. In `ledger.jsonl` a torn *final* line is
   dropped as a crash artifact; corruption anywhere else raises.
8. **Budgets bound the whole run, not one process.** `steps_used` / `elapsed_s`
   are persisted and re-seeded on resume, and limits are checked *before* the
   step is consumed so a pause records an exact count.
9. **Prediction and truth never share a mechanism.** In `lha ablate` the scorer
   uses a fresh directory, canonical tests, and its own backend instance; a
   correct fix the gate wrongly refused is counted as a false negative. `ERROR`
   cells are reported, excluded from rates, and **never cached**. The cache
   fingerprint covers task bytes, corpus digest, model, scorer, repair budget,
   harness version, and the source of `lha/llm/base.py` — so editing the prompt
   re-samples.
10. **Re-expressing evidence must never inflate it.** `lha horizon` composes
    measured cells onto a longer horizon. One repetition of the corpus is one
    independent episode, so `R` repetitions give exactly `R` episodes and the
    paired test at the terminal step must return what the same cells return at
    the step level. Averaging over more orderings changes the effect size, never
    the p-value. Pinned by
    `tests/test_horizon.py::test_composition_does_not_manufacture_significance`;
    if you touch `horizon.py`, that test is the one that matters.
11. **Boundary proportions use Wilson, not a percentile bootstrap.** Resampling
    an all-identical sample reports `0%–0%`, which is an artifact of the method
    rather than certainty.
12. **Only verified successes become memory.** `SkillMemory.record` requires
    `status == "DONE"` *and* a passing `verify.json`, and the note names the
    SHA-256 of that verdict.
13. **Nothing escapes the sandbox.** `_safe_target` refuses absolute paths,
    `..`, and writes through a symlink; `_safe_seg` collapses a malicious
    `step_id` into one path segment; dynamic plans with unsafe step ids or
    unregistered verifiers are rejected in favor of the template.
14. **Facade isolation.** Nothing outside `src/lha/live_context/` may import
    `cocoindex`/`cocoindex_code` or invoke `ccc`. Enforced twice: a grep in CI
    and an AST walk in `tests/test_hardening.py`.

## 6. The gate — run all of it before finishing

This is exactly what `.github/workflows/ci.yml` runs. Measured on this machine
at branch HEAD `7631612`:

```bash
uv run ruff check .           # All checks passed!
uv run pyright src/lha        # 0 errors, 0 warnings, 0 informations
uv run pytest -q              # 210 passed, 1 skipped   (~190 s; the skip is docker, opt-in)
uv run lha eval               # score: 5/5              (~46 s)

# facade isolation — must print nothing
grep -rnE "^[[:space:]]*(import|from)[[:space:]]+(cocoindex|cocoindex_code)" \
  --include='*.py' src/lha | grep -v "src/lha/live_context/"
```

Opt-in, needs a Docker daemon:

```bash
LHA_DOCKER_TESTS=1 uv run pytest tests/test_sandbox.py -q   # real containers
docker build -t lha . && docker run --rm lha lha --version
```

Notes that matter in practice:

- `lha eval` writes only to `LHA_RUNS_DIR` and `data/.lha_index/` (both
  gitignored). To keep an audit run out of the normal tree:
  `LHA_RUNS_DIR=runs/_scratch uv run lha eval`.
- The first `lha eval` downloads a small sentence-transformers model (tens of
  MB, one-time).
- `lha eval` also rewrites `data/skills/*.md` (gitignored skill memory) as a
  normal consequence of a verified `DONE` run.
- Tests are hermetic by design: `tests/conftest.py::hermetic_task` flips tasks
  to `context_requirement="optional"` and configs use `code_backend="null"` with
  a `tmp_path` `data_dir`. Keep new tests that way — no network, no `ccc`, no
  model download.
- Coverage, if you want it: `uv run pytest --cov=lha --cov-report=term`. It was
  **81%** (3969 statements, 771 uncovered) over 211 tests. The
  uncovered lines are mostly the network/CLI-bound backends (`ccc` MCP I/O, the
  `claude_cli`/`anthropic` clients) that cannot be exercised hermetically. Add a
  meaningful test, never padding to move the number.

## 7. Coding conventions

- **ruff**, line length 100, `extend-select = ["I"]` (import sorting).
  `runs`, `data`, `.cocoindex_code`, `flows/_scratch` are excluded.
- **pyright** must stay at 0 errors over `src/lha` (Python 3.11 target). Prefer
  an explicit `None`-guard or `Literal` narrowing over a blanket `# type: ignore`.
- `from __future__ import annotations` at the top of every module.
- Data that crosses a boundary is a **pydantic model**; internal value objects
  are `@dataclass`. Never pass free-form dicts between roles.
- Timestamps come from `lha.clock.now()` (tz-aware UTC) — never
  `datetime.now()`.
- Subprocesses go through `lha.tools.shell.run` or, for anything
  target/model-influenced, `lha.sandbox.ExecutionBackend`.
- Optional dependencies (`cocoindex`, `anthropic`, `harbor`, `sentence_transformers`,
  `skimage`) are imported **inside** the function that needs them, so the core
  imports cleanly without the extras.
- Comments explain *why* a line is the way it is (usually the failure mode it
  prevents). The existing code does this consistently; match it and do not add
  narration of what the code obviously does.
- Conventional commits: `feat:` / `fix:` / `docs:` / `test:` / `ci:` / `chore:` /
  `refactor:`, with a scope where it helps (`fix(policy): …`).
- New objective checks belong in a verifier family under
  `src/lha/verifiers/<family>/` and are registered in `registry.py` — not
  special-cased in the loop.

## 8. Prohibited

- **Never weaken a check to make the gate green.** No skipping, `xfail`ing,
  deleting, or commenting out a test; no lowering a threshold; no loosening a
  verifier. Fix the cause or revert.
- **Never import `cocoindex`/`cocoindex_code` or shell out to `ccc` outside
  `src/lha/live_context/`.**
- **Never let "could not verify" become "verified"** — the single most important
  rule in the codebase.
- **Never reuse the internal gate's verdict as ground truth** in the ablation or
  a bench adapter.
- **Never publish a number that was not produced by a command in this repo.** If
  a measurement changes, regenerate `benchmarks/ablation_report.{json,md}` and
  update every doc that quotes it (`README.md`, `docs/ABLATION.md`,
  `CHANGELOG.md`) in the same change.
- **Never commit generated state**: `runs/`, `data/.lha_index/`, `data/skills/`,
  `.coverage`, `.cocoindex_code/`, or a `uv.lock` inside `data/bench/*`.
- **Never run `git push`, create a tag, or cut a release without being asked.**
  There is currently no configured git remote.
- **Do not edit the bench corpus or its oracles to improve a result.** Fixtures
  were authored and calibrated before any model output on them was observed;
  changing a fixture's bytes also busts the ablation cache fingerprint and
  re-samples every cell.
- `trusted-local` is **not** a sandbox against hostile code. Use
  `LHA_EXEC_BACKEND=docker` for any external target repo.

## 9. Known limitations

- **`trusted-local` isolation is environment-scrubbing only** (no inherited
  secrets, process-group kill, opt-in rlimits). On macOS `RLIMIT_NPROC` is
  per-user, so host rlimits are off by default.
- **Prompt injection through indexed content is mitigated, not prevented** — a
  poisoned suggestion still has to pass the objective gate.
- **The `docker` backend's image must carry the tools**: the default
  `python:3.12-slim` has no `pytest`/`pytest-json-report`/`ruff`, so
  code-verification tasks need a purpose-built image.
- **Public benchmarks are adapters only.** SWE-bench Verified and Terminal-Bench 2
  are wired and contract-tested; **no evaluation run has been executed and no
  number is claimed.** Both need Docker and paid model calls. The
  Terminal-Bench path also needs a freshly built wheel (`uv build`) and
  `uvx --python 3.12` because `harbor` requires Python ≥ 3.12.
- **The ablation measures a mechanism, not a leaderboard.** At the committed
  snapshot the effect is 2 false successes in 51 paired cells (exact McNemar
  p = 0.50); the boundary CIs (`0% (0–0%)`, `100% (100–100%)`) are degenerate
  percentile-bootstrap artifacts. Scorer independence is state-level, not
  environment-level, unless `--scorer-backend docker` is used. The implementer's
  tool denial is a deny-list and the `claude` CLI version is not in the
  provenance fingerprint.
- **The context family is a weaker oracle than a test suite.** Freshness and
  citation-resolution are the best available signal where no objective oracle
  exists; they are labelled as context-family checks, not ground truth.
- **`ccc` code search is optional and stateful.** A daemon churned by many
  ad-hoc runs can report a transient code-context miss; `ccc daemon restart`
  and re-run. Ephemeral per-run workdirs are deliberately *not* indexed — the
  loop indexes the stable `target_repo` instead.
- **The facade is a process-global singleton**, which is why cross-task
  parallelism (`lha batch`) uses worker subprocesses rather than threads.
