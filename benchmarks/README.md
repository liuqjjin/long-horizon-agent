# Benchmark reports

This directory contains committed, machine-checkable reports. Generated runs
under `runs/` are ignored by Git; files are copied here only after the report,
source revision, and execution provenance have been validated.

## Current formal run

The current ablation report uses schema v2:

- Codex CLI 0.141.0;
- model `gpt-5.4-mini`, reasoning effort `low`, sandbox `read-only`;
- 17 fixed Python tasks × 12 repetitions = 204 paired cells;
- Docker final scorer on fresh repository copies;
- zero `ERROR` cells.

The measured outcomes are:

| condition | independent-scorer outcome |
|---|---|
| `trust` | 194/204 correct; 10 incorrect attempts accepted |
| `gate` | 194 correct attempts accepted; all 10 incorrect attempts blocked |
| `verify` | 204/204 correct after repair |

The 204 paired cells have an exact two-sided McNemar value of `0.001953125`.
The generated Markdown currently formats that field to two decimal places and
therefore prints `0.00`; this is display rounding, not a zero p-value. Public
summary text uses `0.00195`. Twelve complete-corpus repetitions provide 12
observed episodes: `trust` succeeds on 2/12 and `verify` on 12/12. The
composition curve is only a projection over empirical per-task rates and adds
zero independent samples.

## Files

- [`ablation_report.json`](ablation_report.json) is the source of record. It
  contains schema version, every cell, condition summaries, model-call audits,
  source hashes, clean-worktree status, runtime versions, scorer image, and a
  report fingerprint.
- [`ablation_report.md`](ablation_report.md) is generated from that JSON.
- [`horizon_report.json`](horizon_report.json) keeps cell, episode, and
  composition estimands in separate objects.
- [`horizon_report.md`](horizon_report.md) and
  [`horizon_curve.svg`](horizon_curve.svg) are generated renderings.

`trust` and `gate` score the same first attempt. The internal gate predicts
whether work may advance; it never supplies benchmark truth. The Docker scorer
applies the frozen change to a fresh canonical repository and runs the original
tests independently.

## Reproduce

Build the scorer image first. It must contain the task's scoring dependencies;
the default slim Python image is insufficient.

```bash
docker build -t lha:release .

LHA_EXEC_IMAGE=lha:release \
LHA_CODEX_MODEL=gpt-5.4-mini \
LHA_CODEX_EFFORT=low \
uv run lha --llm codex_cli ablate \
  --model gpt-5.4-mini \
  --reps 12 \
  --scorer-backend docker \
  --out runs/ablation

uv run lha horizon \
  --from-report runs/ablation/ablation_report.json \
  --out runs/horizon
```

A rerun is new evidence and may differ from the committed result. Do not
overwrite this directory unless the generated report records a clean source
tree, the intended CLI/model configuration, all 204 cells, and zero errors.
Copy both machine-readable and rendered artifacts together.

Before publishing or changing any number, run:

```bash
uv run python -m lha.release_claims
```

This check derives counts and exact paired tests from the raw records,
reproduces the horizon artifacts, and compares public claims with the committed
JSON. Method details are in [`docs/ABLATION.md`](../docs/ABLATION.md) and
[`docs/HORIZON.md`](../docs/HORIZON.md).

The deterministic `lha eval` and public-benchmark adapters are documented in
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md). Adapter support is not a
Terminal-Bench or SWE-bench result, and no public benchmark score is claimed.
