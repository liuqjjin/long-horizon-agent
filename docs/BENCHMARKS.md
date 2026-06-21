# ResearchAgentBench-Lite

The harness measuring itself. Five tasks, each with an **objective** pass/fail —
no LLM judges the outcome. It is the executable form of the project's thesis: a
step is only "done" when an external oracle says so.

## Reproduce

```bash
uv sync
uv run lha eval            # all five tasks; writes runs/eval_report.json
uv run lha eval --quick    # the three fast cases (skips the experiment runs)
```

Every number below is produced by that command in this repo. To re-derive the
per-task oracle, read `src/lha/eval.py` (one function per case) and the task specs
under `data/tasks/`.

## Results

Verbatim output of `uv run lha eval`:

```
# ResearchAgentBench-Lite — 5/5

| dimension              | case                    | result | detail |
|------------------------|-------------------------|--------|--------|
| issue-to-PR            | fix_average             | PASS   | status=DONE verified=True |
| resume                 | pause_resume            | PASS   | first=PAUSED resumed=DONE |
| freshness              | edit_reindex            | PASS   | initial_fresh=True stale_after_edit=True fresh_after_reject=True |
| paper-to-experiment    | bicubic_sr              | PASS   | status=DONE verified=True |
| verification-ablation  | strict_threshold_caught | PASS   | status=FAILED psnr_correctly_rejected=True reached_psnr_step=True |

score: 5/5
```

The companion unit suite is green too: `uv run pytest` → 33 passed.

## What each task verifies

| # | Dimension | Task | Objective oracle | Pass condition |
|---|-----------|------|------------------|----------------|
| 1 | issue-to-PR | `fix_average` | a real `pytest` run + `ruff` on the patched sandbox | run reaches `DONE` **and** `verify.json.passed` (tests pass, lint clean) |
| 2 | resume | `pause_resume` | re-entering a checkpointed run in a fresh harness | first run `PAUSED` (budget), then `resume` → `DONE` + verified |
| 3 | freshness | `edit_reindex` | mtime/index-generation vs. source, then incremental reindex | context `fresh → stale (after edit) → fresh (after reject_stale)` |
| 4 | paper-to-experiment | `bicubic_sr` | PSNR/SSIM **recomputed from the saved arrays** + a reproducibility re-run | `DONE` + verified (PSNR ≥ 24 dB, SSIM ≥ 0.80, deterministic re-run, seed/versions recorded) |
| 5 | verification-ablation | `strict_threshold_caught` | the PSNR verifier against an unreachable bar | run is `FAILED`, the `psnr` check failed, and the experiment step was actually reached |

Tasks 1, 4, and 5 live in `data/tasks/*.yaml`; tasks 2 and 3 are driven directly
in `src/lha/eval.py`. Verifier thresholds are explicit in the task specs
(`psnr_min: 24.0`, `ssim_min: 0.80`, `data_range: 1.0`).

## The centerpiece: verification-ablation

This case exists to prove the loop's verifier is **load-bearing**, not decorative.

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
  would end "successfully" with a silently-wrong 25 dB result.

That gap — `FAILED` vs. a false `DONE` — is the entire value proposition,
demonstrated on a runnable task rather than asserted. It also guards against
*fabricated* metrics: because the verifier recomputes from the arrays, a doctored
`metrics.json` is caught (see `tests/test_experiment_verifiers.py::test_psnr_catches_fabricated_metric`).

## Honesty notes

- **Scope.** This is a *lite* benchmark on small, bundled tasks — it measures that
  the harness's own workflows behave correctly end-to-end, not performance against
  an external SWE/agent benchmark. No external leaderboard numbers are claimed.
- **Reproducibility.** From a clean checkout, `uv sync && uv run lha eval` (run from
  the repo root) reproduces `5/5` — verified by deleting all gitignored generated
  state (`runs/`, `data/.lha_index/`, `data/skills/`) and re-running. The first run
  downloads a small sentence-transformers model (~tens of MB, one-time).
- **Determinism.** The experiment is seeded and deterministic; the freshness case
  is tested via index-generation timestamps (not wall-clock races). A clean
  checkout with a fresh `ccc` daemon reproduces `5/5`; a daemon heavily churned by
  many prior ad-hoc runs can briefly report a transient code-context miss — re-run
  on a fresh daemon. (Hardening this is tracked in `OVERNIGHT_LOG.md`.)
