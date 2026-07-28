# Benchmark reports

This directory contains committed, machine-checkable reports. Generated runs
under `runs/` are ignored by Git; files are copied here only after the report,
source revision, and execution provenance have been validated.

## Internal ablation: historical schema-v2 record

The committed ablation and horizon files were produced with the schema-v2
protocol. They are retained so the earlier run and its raw cells can be
inspected.

The scoring boundary, error classification, and evidence format have since
changed. The schema-v2 files are therefore not the current project result and
their numbers should not be copied into the README or a resume. New claims
require a complete schema-v4 rerun with its raw records, configuration, and
rendered reports committed together.

The formal schema-v4 protocol requires a committed attempt registration, a
create-only remote start witness, a new output directory, no cache reads, and
no resume. An interrupted attempt is recorded as abandoned. The final
schema-v4 result has not been committed, so this directory does not publish its
counts.

## Terminal-Bench 2.1 fixed subset

[`terminal_bench_2_1/`](terminal_bench_2_1/) is the committed schema-v4 evidence
for one fixed 20-task subset run: 7 `PASS`, 9 `FAIL`, and 4 `ERROR`. All four
errors remain in the denominator, so the result is 7/20. This is not a
full-dataset or leaderboard score.

The package binds the protocol, selected task IDs, source commit and wheel,
model settings, Harbor and Codex versions, task outcomes, and summary. Official
result JSON is public for the 16 `PASS` or `FAIL` tasks. The four error records
are redacted projections bound to private originals by SHA-256, so their
private exception text cannot be reconstructed from this repository.

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
- [`terminal_bench_2_1/evidence.json`](terminal_bench_2_1/evidence.json) indexes
  the fixed-subset protocol, manifests, source attestation, records, trials,
  and summary.

`trust` and `gate` score the same first attempt. The internal gate predicts
whether work may advance; it is not reused as the benchmark result. The
schema-v2 scoring path applies the frozen change to a fresh repository and runs
the original tests through a separate execution backend.

## Exploratory ablation

Build the scorer image first. It must contain the task's scoring dependencies;
the default slim Python image is insufficient.

```bash
docker build -t lha:release .

LHA_EXEC_IMAGE=lha:release \
LHA_CODEX_MODEL=gpt-5.4-mini \
LHA_CODEX_EFFORT=low \
uv run lha --llm codex_cli ablate \
  --model gpt-5.4-mini \
  --reps 1 \
  --scorer-backend docker \
  --out runs/ablation-exploratory

uv run lha horizon \
  --from-report runs/ablation-exploratory/ablation_report.json \
  --out runs/horizon-exploratory
```

This command is exploratory evidence and may differ from the historical
record. The formal 17 × 12 command is accepted only from a clean checkout whose
registration fixes the exact source, corpus, model, CLI/client settings, Docker
image, output path, and witness remote. See
[`docs/ABLATION.md`](../docs/ABLATION.md); do not turn an exploratory output
into a formal claim.

Before publishing or changing any number, run:

```bash
uv run python -m lha.release_claims
```

This check derives counts and exact paired tests from the raw records,
reproduces the horizon artifacts, and compares public claims with the committed
JSON. Method details are in [`docs/ABLATION.md`](../docs/ABLATION.md) and
[`docs/HORIZON.md`](../docs/HORIZON.md).

The deterministic `lha eval` and public-benchmark adapters are documented in
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md). Adapter support alone is not a
Terminal-Bench or SWE-bench result. The only public external result currently
claimed is the committed Terminal-Bench fixed 20-task subset above.
