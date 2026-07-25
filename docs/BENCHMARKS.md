# Benchmarks

Three layers, by who grades the work:

1. **Self-eval** (below) — the harness checking its own six workflows on a
   deterministic stub. Runs in CI.
2. **Verification ablation** ([ABLATION.md](ABLATION.md)) — a real LLM through the
   harness, graded by an independent scorer. Committed snapshot in
   [`benchmarks/`](../benchmarks/).
3. **Public benchmarks** (bottom of this page) — SWE-bench Verified and
   Terminal-Bench 2 adapters, graded by their official harnesses. **No runs have
   been executed yet; no numbers are claimed.** The adapters and their contract
   tests are in `src/lha/bench/` and `tests/test_bench_adapters.py`.

# Self-eval

The harness checking itself: six workflows, each with an objective pass/fail. It
exercises the project's thesis end to end — a step is "done" only when an external
oracle says so.

## Reproduce

```bash
uv sync
uv run lha eval            # all six tasks; writes runs/eval_report.json
uv run lha eval --quick    # the three fast cases (skips the experiment runs)
```

Every number below is produced by that command in this repo. To re-derive the
per-task oracle, read `src/lha/eval.py` (one function per case) and the task specs
under `data/tasks/`.

## Results

Verbatim output of `uv run lha eval`:

```
# Self-eval — 6/6

| dimension | case | result | detail |
|---|---|---|---|
| issue-to-PR | fix_average | PASS | status=DONE verified=True |
| resume | pause_resume | PASS | first=PAUSED resumed=DONE |
| freshness | edit_reindex | PASS | initial_fresh=True stale_after_edit=True fresh_after_reject=True |
| fail-closed context | required_context_unavailable | PASS | status=FAILED verdict_named_the_reason=True |
| paper-to-experiment | bicubic_sr | PASS | status=DONE verified=True |
| verification-ablation | strict_threshold_caught | PASS | status=FAILED psnr_correctly_rejected=True reached_psnr_step=True |

score: 6/6
```

The companion unit suite is green too (`uv run pytest`).

## What each task verifies

| # | Dimension | Task | Objective oracle | Pass condition |
|---|-----------|------|------------------|----------------|
| 1 | issue-to-PR | `fix_average` | a real `pytest` run + `ruff` on the patched sandbox | run reaches `DONE` **and** `verify.json.passed` (tests pass, lint clean) |
| 2 | resume | `pause_resume` | re-entering a checkpointed run in a fresh harness | first run `PAUSED` (budget), then `resume` → `DONE` + verified |
| 3 | freshness | `edit_reindex` | mtime/index-generation vs. source, then incremental reindex | context `fresh → stale (after edit) → fresh (after reject_stale)` |
| 4 | paper-to-experiment | `bicubic_sr` | PSNR/SSIM **recomputed from the saved arrays** + a reproducibility re-run | `DONE` + verified (PSNR ≥ 24 dB, SSIM ≥ 0.80, deterministic re-run, seed/versions recorded) |
| 5 | verification-ablation | `strict_threshold_caught` | the PSNR verifier against an unreachable bar | run is `FAILED`, the `psnr` check failed, and the experiment step was actually reached |
| 6 | fail-closed context | `required_context_unavailable` | a step that requires context against a backend forced dark | run is `FAILED` **and** the `freshness` check names the unavailable context — a failure for the right reason, not any failure |

Tasks 1, 4, 5, and 6 live in `data/tasks/*.yaml`; tasks 2 and 3 are driven directly
in `src/lha/eval.py`. Verifier thresholds are explicit in the task specs
(`psnr_min: 24.0`, `ssim_min: 0.80`, `data_range: 1.0`).

## The verification-ablation case

This case shows the verifier changes the outcome.

The bundled experiment (`data/sample_experiment/experiment.py`) is a deterministic
bicubic 4× super-resolution baseline on `skimage.data.astronaut()`. The
experiment verifiers **recompute** the metrics from the saved output (they do not
trust the experiment's self-reported numbers):

- **PSNR ≈ 25.07 dB, SSIM ≈ 0.8246** (`data_range = 1.0`).

The normal task (`run_sr_experiment.yaml`) asks for `psnr_min: 24.0` → the harness
verifies and reports `DONE`. The ablation task
(`run_sr_experiment_strict.yaml`) asks for an unreachable `psnr_min: 40.0`:

- **With** the PSNR verifier (the harness): the recomputed 25.07 dB < 40 dB, the
  `psnr` check fails, and the run is reported **`FAILED`** — the agent refuses to
  claim a result it cannot verify.
- **Without** a verifier (a typical orchestrate-and-trust agent): the same run
  would end "successfully" with a wrong 25 dB result reported as a pass.

That gap (`FAILED` vs. a false `DONE`) is what the verifier buys, on a runnable task.
It also guards against fabricated metrics: because the verifier recomputes from the
arrays, a doctored `metrics.json` is caught (see
`tests/test_experiment_verifiers.py::test_psnr_catches_fabricated_metric`).

## Honesty notes

- **Scope.** This is a small self-check on bundled tasks — it measures that
  the harness's own workflows behave correctly end-to-end, not performance against
  an external SWE/agent benchmark. No external leaderboard numbers are claimed.
- **Reproducibility.** From a clean checkout, `uv sync && uv run lha eval` (run from
  the repo root) reproduces `6/6` — verified by deleting all gitignored generated
  state (`runs/`, `data/.lha_index/`, `data/skills/`) and re-running. The first run
  downloads a small sentence-transformers model (~tens of MB, one-time).
- **Environment independence.** Every case asserts the same thing with or without a
  code-search backend. The loop cases (1, 2) declare retrieval optional and are
  graded by a real `pytest` run; case 4 forces the backend dark rather than
  depending on whether `ccc` is installed. An earlier version loaded task 1 with
  its default `context_requirement: required`, so it scored 5/5 on a machine with
  `ccc` and 3/5 on CI — the harness was right to fail closed, and the claim was
  the thing that was wrong.
- **Determinism.** The experiment is seeded and deterministic, and the freshness
  case is tested via index-generation timestamps rather than wall-clock races.

# Public benchmarks (adapters ready, not yet run)

`src/lha/bench/` connects the harness to two public evaluators. In both, the
grading is done by the official harness on frozen predictions — the same
prediction/truth separation the ablation uses. **No evaluation runs have been
executed; the tables above contain the only measured numbers in this repo.**
Running either benchmark costs real model calls and needs Docker.

## SWE-bench Verified

Dataset `SWE-bench/SWE-bench_Verified` (500 instances), evaluated by
`swebench` ≥ 4.1. The adapter writes predictions in the official three-field
JSONL (`instance_id`, `model_name_or_path`, `model_patch`), refuses duplicate
instance ids, and parses the official `schema_version: 2` report with
evaluation ERRORs kept in the denominator (`resolved_rate = resolved /
submitted`, never `resolved / completed`).

```bash
uv sync --extra bench
# 1. produce predictions with the harness (one lha run per instance), then:
python -c "
from lha.bench import write_predictions, eval_command
from lha.bench.swebench import prediction_from_run
preds = [prediction_from_run('runs/<run_id>', '<instance_id>', 'lha+claude-haiku-4-5-20251001')]
print(' '.join(eval_command(write_predictions(preds, 'preds.jsonl'), run_id='lha-v0')))
"
# 2. run the printed official command; on Apple silicon add namespace='' so
#    images build locally (upstream images are x86_64).
```

The internal gate may run the target repo's own tests, but it never sees
SWE-bench's held-out FAIL_TO_PASS tests — those are applied by the official
harness inside its own containers.

## Terminal-Bench 2 (Harbor)

Dataset `terminal-bench/terminal-bench-2` (89 frozen tasks; TB 2.1 exists and
is newer — 2.0 is pinned for comparability), driven by the `harbor` framework.
The adapter (`lha.bench.terminal_bench.build_agent()`) is a Harbor
`BaseInstalledAgent` that installs lha from a wheel plus the claude CLI into
the task container, maps the instruction to a `TaskSpec`, and copies the
workdir onto the graded filesystem only when the run ends `DONE` — an edit
that failed verification never reaches the grader.

```bash
uv build                                # dist/lha-<version>-py3-none-any.whl
# harbor needs Python >= 3.12 (lha itself supports 3.11):
LHA_WHEEL=dist/lha-*.whl uvx --python 3.12 --with 'harbor>=0.20' \
  harbor run -d terminal-bench/terminal-bench-2 \
  -a module=lha.bench.terminal_bench:build_agent -l 5
```

Honest constraints, stated up front: the container has no code-search backend
(tasks run with `context_requirement: optional`), the loop targets
file-editing tasks (service/OS-state tasks will simply fail their checks), and
`ANTHROPIC_API_KEY` must be provided to the container by harbor's env
passthrough.

## Paired statistics

`lha.bench.stats` carries the comparison tools for when runs exist: an exact
McNemar test on discordant pairs (two conditions on the same instances) and a
seeded task-cluster bootstrap for CIs. The ablation already reports with the
same cluster-bootstrap method.
