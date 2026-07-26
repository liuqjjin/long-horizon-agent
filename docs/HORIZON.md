# Error compounding over a horizon

`lha horizon` reads measured truth labels from an `ablation_report.json` and
reports three different quantities. Keeping them separate prevents a descriptive
curve from being presented as additional experimental evidence.

## The three units

### Paired cell

A cell is one `(task, repetition)` pair for which both `trust` and `verify` have
an independent-scorer result. The cell-level McNemar test asks whether those two
conditions disagree on the same task attempt.

If an ablation has `T` tasks and `R` complete repetitions, it can contribute up
to `T × R` paired cells. A cell recorded as `ERROR` has no truth label and is not
silently converted to success or failure.

### Observed episode

An episode is one complete repetition of the entire corpus. It succeeds only if
every task in that repetition succeeds. `R` complete repetitions therefore
provide exactly `R` paired episodes, not `T × R`.

Several failed cells in one repetition collapse into one failed episode. The
episode-level McNemar test consequently uses a different unit from the cell
test, and the two p-values can differ. Neither is substituted for the other.

### Descriptive composition

The composition inserts each task's empirical success rate into an
independent-step, uniformly random ordering model. For a horizon of `k`, the
projected survival probability is the degree-`k` elementary symmetric polynomial
of the per-task rates divided by `C(T, k)`.

`src/lha/horizon.py::compounding_curve` evaluates this expression exactly. The
task bootstrap around the curve describes sensitivity to the observed task mix.
It is not an episode confidence interval.

Composition adds **zero** independent samples and has no McNemar p-value.
Reordering the same measured cells can change the displayed effect size, not the
amount of evidence.

## Conditions

| horizon condition | source condition | meaning |
|---|---|---|
| `trust-chain` | `trust` | a wrong step is accepted and the chain is already incorrect |
| `verify-chain` | `verify` | each step uses the gate and bounded repair |

Both read the `true_success` label supplied by the independent scorer. Internal
gate acceptance is not treated as truth.

## Intervals and paired tests

Boundary episode rates use Wilson score intervals. A percentile bootstrap over
an all-zero or all-one sample would produce a misleading zero-width interval.

The generated report includes:

- cell pair count, success counts, discordant directions, and exact McNemar
  p-value;
- complete episode count, end-to-end outcomes, Wilson intervals, discordant
  directions, and exact McNemar p-value;
- the composition curve with task-bootstrap intervals and an explicit
  `independent_samples_added: 0`.

Regression tests include a case where multiple discordant cells fall inside one
episode, proving that cell- and episode-level p-values are allowed to differ.

## Committed schema-v2 result

The current input is the Docker-scored ablation produced with Codex CLI 0.141.0,
`gpt-5.4-mini`, low reasoning effort, and read-only mode. It contains 17 tasks
× 12 repetitions and no `ERROR` cells. The generated report keeps the three
estimands separate:

| estimand | paired units | `trust` success | `verify` success | discordant | exact McNemar |
|---|---:|---:|---:|---:|---:|
| measured cell | 204 | 194 | 204 | 10 / 0 | 0.001953125 |
| observed whole-corpus episode | 12 | 2 | 12 | 10 / 0 | 0.001953125 |
| descriptive composition | 0 new samples | — | — | — | none |

The equal p-values in this particular report follow from its observed
discordant counts; the two tests use different units and are not required to
agree. In prose, the measured comparison is reported as `p = 0.00195`.

The composition curve is a model-based projection over the 204 measured cells.
It does not turn task orderings, bootstrap draws, or horizon points into new
experiments.

## Relation to the executed long tasks

The horizon composition and the long-task fixtures answer different questions.

- `lha horizon` projects how measured, independent ablation tasks compound.
- `data/long_tasks/` executes five stateful 10-step repository plans covering
  integrity, baseline reproduction, approved editing, targeted/full checks,
  lint, build, repair, and interruption recovery.

The 10-step runs demonstrate state transitions and recovery. They do not increase
the ablation's cell or episode sample count, and the composed curve is not
reported as their measured success rate.

## Generate a report

Compose the committed ablation input without model calls:

```bash
uv run lha horizon \
  --from-report benchmarks/ablation_report.json \
  --out runs/horizon
```

Or use a newly generated ablation report:

```bash
uv run lha horizon \
  --from-report runs/ablation/ablation_report.json \
  --out runs/horizon
```

The command writes:

```text
runs/horizon/horizon_report.json
runs/horizon/horizon_report.md
runs/horizon/horizon_curve.svg
```

The JSON contains the three estimands as separate objects. The Markdown and SVG
are renderings of the same data.

## Result policy

The summary above mirrors the current generated report. Full-precision results
belong in the generated files under `benchmarks/`, together with their source
report, model, backend, repetitions, and fingerprint. When a release measurement
changes, regenerate both ablation and horizon artifacts before updating this
summary; do not edit a percentage or p-value in isolation.

Before citing a result, verify that:

1. the ablation report has no unaccounted `ERROR` cells;
2. the model and CLI provenance match the intended protocol;
3. cell and episode units are labelled separately;
4. composition still reports zero added samples;
5. every cited number appears in the committed generated report.

Hermetic coverage for the calculations and input validation is in
`tests/test_horizon.py`.
