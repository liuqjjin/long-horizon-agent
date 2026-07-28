# Error compounding over a horizon — legacy snapshot

corpus: 17 independent subtasks · model: `gpt-5.4-mini` · complete paired repetitions: 12 → **12 independent observed episodes** · per-step truth from `benchmarks/ablation_report.json`

Coverage: scheduled paired cells **204** · usable paired cells **204** · unavailable/error cells **0** · scheduled repetitions **12** · complete paired repetitions **12**.

This report keeps three estimands separate. The cell and episode tests use different paired units; the composition is a descriptive model projection and adds no observations.

## Estimand 1 — paired cells

Unit: `task × repetition cell` · pairs: **204** · `trust` true success: 194/204 · `verify` true success: 204/204.

Discordant cells (verify-only / trust-only): 10/0 · exact McNemar p = 0.0020

## Estimand 2 — observed episodes

An episode is one complete corpus repetition and is correct only if every subtask in that repetition truly succeeded. Multiple failed cells in the same repetition still make one failed episode.

| condition | end-to-end correct | first failing subtask(s) |
|---|---|---|
| `trust-chain` | 2/12 (5%–45%, Wilson) | `bench_caesar`, `bench_urljoin` |
| `verify-chain` | 12/12 (76%–100%, Wilson) | — |

Discordant episodes (verify-only / trust-only): 10/0 of 12 paired episodes · exact McNemar p = 0.0020

The cell- and episode-level p-values may coincide for a particular dataset, but equality is not a statistical contract: aggregation changes the paired unit and can collapse many cell disagreements into one episode disagreement.

## Estimand 3 — descriptive composition

The curve inserts empirical per-task success rates into an independent-step, uniform-random-order model. Its task-bootstrap interval describes sensitivity to the observed task mix; it is not an episode confidence interval and has no McNemar p-value.

Independent samples added by composition: **0**.

Composition uses every available measurement for each task. Per-task sample sizes may differ after ERROR or missing cells; these measurements are reused descriptively and do not add observations.

| task | `trust-chain` measured rate | `verify-chain` measured rate |
|---|---:|---:|
| `bench_basen` | 100% (n=12) | 100% (n=12) |
| `bench_brackets` | 100% (n=12) | 100% (n=12) |
| `bench_bsearch` | 100% (n=12) | 100% (n=12) |
| `bench_caesar` | 75% (n=12) | 100% (n=12) |
| `bench_csvlite` | 100% (n=12) | 100% (n=12) |
| `bench_flatten` | 100% (n=12) | 100% (n=12) |
| `bench_lru` | 100% (n=12) | 100% (n=12) |
| `bench_median` | 100% (n=12) | 100% (n=12) |
| `bench_merge_intervals` | 100% (n=12) | 100% (n=12) |
| `bench_paginate` | 100% (n=12) | 100% (n=12) |
| `bench_roman` | 100% (n=12) | 100% (n=12) |
| `bench_rpn` | 100% (n=12) | 100% (n=12) |
| `bench_runstats` | 100% (n=12) | 100% (n=12) |
| `bench_slugify` | 100% (n=12) | 100% (n=12) |
| `bench_spans` | 100% (n=12) | 100% (n=12) |
| `bench_urljoin` | 42% (n=12) | 100% (n=12) |
| `bench_window` | 100% (n=12) | 100% (n=12) |

| k | `trust-chain` (95% task-bootstrap interval) | `verify-chain` (95% task-bootstrap interval) | gap |
|---:|---|---|---:|
| 1 | 95% (87%–100%) | 100% (100%–100%) | +4.9 pp |
| 2 | 90% (76%–100%) | 100% (100%–100%) | +9.7 pp |
| 4 | 81% (57%–100%) | 100% (100%–100%) | +19.0 pp |
| 6 | 72% (42%–100%) | 100% (100%–100%) | +27.8 pp |
| 8 | 64% (30%–100%) | 100% (100%–100%) | +36.2 pp |
| 10 | 56% (22%–100%) | 100% (100%–100%) | +44.2 pp |
| 12 | 48% (15%–100%) | 100% (100%–100%) | +51.7 pp |
| 14 | 41% (10%–100%) | 100% (100%–100%) | +58.9 pp |
| 16 | 34% (7%–100%) | 100% (100%–100%) | +65.6 pp |
| 17 | 31% (5%–100%) | 100% (100%–100%) | +68.8 pp |

Conditions:
- `trust-chain` — no gate: a wrong step is accepted silently.
- `verify-chain` — gate plus repair at every step.

Only new complete repetitions increase the episode sample count. Reordering or composing the existing cells changes the projected effect size, not the number of independent observed episodes.
