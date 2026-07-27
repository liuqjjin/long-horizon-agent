# Benchmark reports

This directory contains committed, machine-checkable reports. Generated runs
under `runs/` are ignored by Git; files are copied here only after the report,
source revision, and execution provenance have been validated.

## Historical schema-v2 record

The committed ablation and horizon files were produced with the schema-v2
protocol. They are retained so the earlier run and its raw cells can be
inspected.

The scoring boundary, error classification, and evidence format have since
changed. The schema-v2 files are therefore not the current project result and
their numbers should not be copied into the README or a resume. New claims
require a complete schema-v4 rerun with its raw records, configuration, and
rendered reports committed together.

## Files

- [`ablation_report.json`](ablation_report.json) is the machine-readable source
  for the historical run. It
  contains schema version, every cell, condition summaries, model-call audits,
  source hashes, clean-worktree status, runtime versions, scorer image, and a
  report fingerprint.
- [`ablation_report.md`](ablation_report.md) is generated from that JSON.
- [`horizon_report.json`](horizon_report.json) keeps cell, episode, and
  composition estimands in separate objects.
- [`horizon_report.md`](horizon_report.md) and
  [`horizon_curve.svg`](horizon_curve.svg) are generated renderings.

`trust` and `gate` score the same first attempt. The internal gate predicts
whether work may advance; it is not reused as the benchmark result. The
schema-v2 scoring path applies the frozen change to a fresh repository and runs
the original tests through a separate execution backend.

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

A rerun is new evidence and may differ from the historical record. Do not
overwrite this directory unless the schema-v4 run is complete and its protocol
accepts every result or error into the denominator. Copy machine-readable and
rendered artifacts together.

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
