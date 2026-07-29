# Error compounding over a horizon

corpus: 17 separately measured tasks · model: `gpt-5.3-codex-spark` · complete paired repetition aggregates: **12** · per-step truth from `benchmarks/ablation_report.json`

Coverage: scheduled paired cells **204** · usable paired cells **204** · unavailable/error cells **0** · scheduled repetitions **12** · complete paired repetitions **12**.

This report keeps three estimands separate. The cell and complete-corpus aggregate tests use different paired units; the composition is a descriptive model projection and adds no observations.

A complete-corpus repetition aggregate is constructed after execution from separately measured task cells that share a repetition number. It is not an executed shared-state long task: tasks do not share a worktree, context, checkpoint, or failure history here.

## Estimand 1 — paired cells

Unit: `task × repetition cell (inference clustered by task)` · pairs: **204** · `trust` true success: 201/204 · `verify` true success: 204/204.

Discordant cells (verify-only / trust-only): 3/0. These cell counts are descriptive; repeated cells from one task do not receive separate inferential weight.
Task-cluster inference: 3/17 tasks have a non-zero paired effect · exact paired sign-flip p = 0.2500 — **not significant**

## Estimand 2 — complete-corpus repetition aggregates

An aggregate is correct only if every measured task cell in that repetition truly succeeded. Multiple failed cells in the same repetition still make one failed aggregate.

| condition | aggregate correct | failing task(s) |
|---|---|---|
| `trust-chain` | 10/12 (55%–95%, Wilson) | `bench_brackets`, `bench_caesar`, `bench_lru` |
| `verify-chain` | 12/12 (76%–100%, Wilson) | — |

Discordant complete-corpus aggregates (verify-only / trust-only): 2/0 of 12 paired aggregates · exact McNemar p = 0.5000 — **not significant**

The task-cluster cell test and repetition-aggregate McNemar test answer different questions. Their p-values need not match: the former gives each task one inferential contribution, while the latter pairs complete-corpus aggregates by repetition.

## Estimand 3 — descriptive composition

The curve inserts empirical per-task success rates into an independent-step, uniform-random-order model. Its task-bootstrap interval describes sensitivity to the observed task mix; it is not a confidence interval for complete-corpus aggregates and has no McNemar p-value.

Independent samples added by composition: **0**.

Composition uses every available measurement for each task. Per-task sample sizes may differ after ERROR or missing cells; these measurements are reused descriptively and do not add observations.

| task | `trust-chain` measured rate | `verify-chain` measured rate |
|---|---:|---:|
| `bench_basen` | 100% (n=12) | 100% (n=12) |
| `bench_brackets` | 92% (n=12) | 100% (n=12) |
| `bench_bsearch` | 100% (n=12) | 100% (n=12) |
| `bench_caesar` | 92% (n=12) | 100% (n=12) |
| `bench_csvlite` | 100% (n=12) | 100% (n=12) |
| `bench_flatten` | 100% (n=12) | 100% (n=12) |
| `bench_lru` | 92% (n=12) | 100% (n=12) |
| `bench_median` | 100% (n=12) | 100% (n=12) |
| `bench_merge_intervals` | 100% (n=12) | 100% (n=12) |
| `bench_paginate` | 100% (n=12) | 100% (n=12) |
| `bench_roman` | 100% (n=12) | 100% (n=12) |
| `bench_rpn` | 100% (n=12) | 100% (n=12) |
| `bench_runstats` | 100% (n=12) | 100% (n=12) |
| `bench_slugify` | 100% (n=12) | 100% (n=12) |
| `bench_spans` | 100% (n=12) | 100% (n=12) |
| `bench_urljoin` | 100% (n=12) | 100% (n=12) |
| `bench_window` | 100% (n=12) | 100% (n=12) |

| k | `trust-chain` (95% task-bootstrap interval) | `verify-chain` (95% task-bootstrap interval) | gap |
|---:|---|---|---:|
| 1 | 99% (97%–100%) | 100% (100%–100%) | +1.5 pp |
| 2 | 97% (94%–100%) | 100% (100%–100%) | +2.9 pp |
| 4 | 94% (89%–100%) | 100% (100%–100%) | +5.8 pp |
| 6 | 91% (83%–100%) | 100% (100%–100%) | +8.6 pp |
| 8 | 89% (79%–100%) | 100% (100%–100%) | +11.3 pp |
| 10 | 86% (74%–100%) | 100% (100%–100%) | +14.0 pp |
| 12 | 83% (69%–100%) | 100% (100%–100%) | +16.7 pp |
| 14 | 81% (65%–100%) | 100% (100%–100%) | +19.2 pp |
| 16 | 78% (61%–100%) | 100% (100%–100%) | +21.7 pp |
| 17 | 77% (59%–100%) | 100% (100%–100%) | +23.0 pp |

Conditions:
- `trust-chain` — no gate: a wrong step is accepted silently.
- `verify-chain` — gate plus repair at every step.

Only newly measured complete repetitions increase the aggregate count. Reordering or composing existing cells changes the projected effect size, not the amount of measured evidence, and does not turn the aggregates into executed shared-state long tasks.
