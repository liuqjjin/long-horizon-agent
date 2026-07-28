# Evaluation

LHA uses three separate evaluation paths. Their results answer different
questions and are not combined.

| evaluation | grader | question |
|---|---|---|
| repository self-eval | LHA's registered checks | Do the bundled workflows still run? |
| verification ablation | an independent Docker scorer | Do the gate and repair loop prevent incorrect delivery? |
| public benchmark | the official benchmark harness | How does the selected model perform on an external task set? |

## Repository self-eval

Run the six fixed workflows:

```bash
uv sync
LHA_RUNS_DIR=runs/_eval uv run lha eval
```

They cover code checks, approval and resume, stale-index recovery, an unavailable
context backend, reproducible experiment output, and an unreachable metric that
must fail. This command is a repository regression check, not an external
benchmark.

## Verification ablation

The committed schema-v2 report records an earlier run over 17 fixed Python
defects with 12 repetitions. Each paired cell shares one first attempt across
`trust`, `gate`, and `verify`; only `verify` may repair after a failed check.

The scorer applies the frozen source change to a fresh repository and runs the
original tests in Docker. It does not reuse the gate decision. This separation
allows the experiment to distinguish a gate prediction from the scorer's
correctness label.

The evidence format and scoring checks have since moved to schema v4. A
schema-v4 formal rerun has not completed, so the schema-v2 numbers are retained
as historical evidence and are not presented as the current result. Read the
generated files instead of copying figures from prose:

- [`benchmarks/ablation_report.json`](../benchmarks/ablation_report.json)
- [`benchmarks/ablation_report.md`](../benchmarks/ablation_report.md)
- [Ablation protocol](ABLATION.md)
- [Horizon analysis](HORIZON.md)

The horizon composition is a projection over measured cells. It adds no
independent samples and has no paired-test p-value.

## SWE-bench Verified adapter

`lha.bench.swebench` writes the official prediction fields and parses schema-v2
reports. Duplicate instance IDs are rejected, and evaluator errors remain in
the denominator. The adapter does not receive held-out `FAIL_TO_PASS` tests.

This repository does not publish a SWE-bench score.

## Terminal-Bench 2.1 fixed 20-task subset

The formal run used the official
[`terminal-bench/terminal-bench-2-1`](https://hub.harborframework.com/datasets/terminal-bench)
dataset through Harbor.

| outcome | count |
|---|---:|
| PASS | 7 |
| FAIL | 9 |
| ERROR | 4 |
| total | 20 |

The result is **7/20**. All four `ERROR` tasks remain in the denominator. This
is a preregistered 20-task subset, not a full-dataset or leaderboard result.

### Fixed protocol

- All 89 task IDs were sorted by `(SHA-256(instance_id), instance_id)`.
- The first 20 IDs formed the scored subset; the next three formed the smoke
  subset.
- All three smoke tasks completed before any scored task started.
- Each scored task ran exactly once, with one attempt and Harbor
  `--max-retries 0`.
- None of the 20 scored tasks was rerun after its result was observed.
- The model was `gpt-5.5` with `xhigh` reasoning effort.
- The runner versions were Harbor `0.20.0` and Codex CLI `0.141.0`.

The full task list, time limits, image digests, binary hashes, request limits,
and command envelopes are in
[`protocol.json`](../benchmarks/terminal_bench_2_1/protocol.json) and
[`scored_manifest.json`](../benchmarks/terminal_bench_2_1/scored_manifest.json).

### ERROR accounting

Two `ERROR` results came from adapter defects discovered during the formal run:

- `caffe-cifar-10` exceeded Harbor's default asynchronous stream line limit;
- `video-processing` was rejected by an overly strict accepted-request bound.

The current adapter fixes both cases, but the formal tasks were not rerun.

The other two `ERROR` results, `configure-git-webserver` and
`make-doom-for-mips`, contain explicit Codex error events. They are not converted
to task failures or removed from the denominator.

### What this run does not measure

The Harbor adapter invokes Codex directly inside each task container. It does
not pass model output through LHA's gate or repair loop. The run therefore does
not measure interception count, false rejection, or repair success. Those
metrics belong to the separate verification ablation.

### Public evidence

The schema-v4 evidence package is in
[`benchmarks/terminal_bench_2_1/`](../benchmarks/terminal_bench_2_1/). It binds
the result to:

- source commit `e63f94620ce8ddd322b19ccb159381183fc31933`;
- Git tree `c5b410a98ebec58d482a8ebc889758bc67662985`;
- wheel SHA-256
  `d34e0569943102c73b8ac4d6209bf5a3a061fada285f7e00c6f70c107f10fac0`;
- evidence-tree SHA-256
  `fb4c374fa75310c5d8f5cfe96976367d772a514881947edb611c52e8147a893f`.

The package contains the official raw JSON payloads for the 16 `PASS` and
`FAIL` tasks. For the four `ERROR` tasks, it contains a redacted projection and
the SHA-256 digest of the private original. A holder of the private result can
later check its exact bytes against that digest. The public package does not
reveal credentials, user-specific paths, or the private exception traces, and
cannot reconstruct those traces.

Validate the committed package without running a benchmark task:

```bash
uv run python tools/run_terminal_bench_2_1.py \
  validate benchmarks/terminal_bench_2_1
```
