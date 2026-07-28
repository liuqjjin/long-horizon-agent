# Horizon analysis

`lha horizon` reads delivered-correctness labels from an ablation report and
calculates three quantities. They are kept separate because they use different
units.

## Paired cell

A cell is one `(task, repetition)` pair with `true_success` for both `trust` and
`verify`. In schema 4, `true_success` means that the condition delivered an
artifact and the independent scorer marked that artifact correct. The
cell-level McNemar test compares those two delivered outcomes.

With `T` tasks and `R` complete repetitions, there can be at most `T × R`
paired cells. An `ERROR` cell has no truth label and is not converted to success
or failure.

## Observed complete repetition

One complete repetition of the corpus is one observed episode. It succeeds only
when every task in that repetition succeeds. Therefore `R` complete repetitions
provide `R` paired episodes, not `T × R`.

Several failed cells in one repetition become one failed episode. The
episode-level McNemar test uses a different unit from the cell-level test, so
their p-values may differ.

## Descriptive composition

The composition curve inserts each task's empirical success rate into an
independent-step model. At horizon `k`, it averages the probability that all
steps succeed over uniformly ordered task subsets of size `k`.

`src/lha/horizon.py::compounding_curve` computes this value from the measured
per-task rates. Its task bootstrap describes sensitivity to the observed task
mix.

The curve adds no independent samples and has no McNemar p-value. It is a
projection, not an additional long-task experiment.

## Conditions and intervals

| condition | source | interpretation |
|---|---|---|
| `trust-chain` | `trust.true_success` | a wrong delivered step breaks the chain |
| `verify-chain` | `verify.true_success` | failed checks may enter bounded repair |

The scorer-only `artifact_correct` field is not used as chain success. A correct
artifact that the system rejects was not delivered, so that step has
`true_success=false`.

Boundary episode rates use Wilson score intervals. A percentile bootstrap on an
all-zero or all-one sample would otherwise return a misleading zero-width
interval.

## Report status

The committed horizon report must be regenerated from the schema-4 ablation
cells. Older reports used the scorer verdict as `true_success`, including for
correct artifacts that were rejected. They remain historical records and are
not current chain-success evidence.

No cell, episode, p-value, or composition value is copied from the older report
into current documentation. The generated schema-4 report is the only source
for those numbers.

The five fixtures under `data/long_tasks/` are separate executed 10-step
workflows. They test state transitions and recovery; they do not add cells or
episodes to this analysis.

## Generate the report

```bash
uv run lha horizon \
  --from-report runs/ablation-schema4/ablation_report.json \
  --out runs/horizon
```

Output:

```text
runs/horizon/horizon_report.json
runs/horizon/horizon_report.md
runs/horizon/horizon_curve.svg
```

The JSON keeps cells, episodes, and composition in separate objects. Markdown
and SVG are renderings of the same data.

Before citing a result, check that model and runtime provenance match the
intended protocol, `ERROR` cells are accounted for, the two measured units are
labelled separately, and composition still reports zero added samples.

The currently committed links below are historical schema-v2 output, not a
current chain-success result:
[`benchmarks/horizon_report.md`](../benchmarks/horizon_report.md), with source
data in [`benchmarks/horizon_report.json`](../benchmarks/horizon_report.json).
Regression tests are in `tests/test_horizon.py`.
