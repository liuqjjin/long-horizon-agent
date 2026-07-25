# Error compounding over a horizon

corpus: 17 independent subtasks · model: `claude-haiku-4-5-20251001` · repetitions: 3 → **3 independent episodes** · per-step truth from `benchmarks/ablation_report.json`

An episode is correct through step k only if every one of steps 1..k truly succeeded, as graded by the ablation's independent scorer.

| k | `trust-chain` (95% CI) | `verify-chain` (95% CI) | gap |
|---:|---|---|---:|
| 1 | 96% (90%–100%) | 100% (100%–100%) | +3.9 pp |
| 2 | 92% (81%–100%) | 100% (100%–100%) | +7.8 pp |
| 4 | 85% (65%–100%) | 100% (100%–100%) | +15.2 pp |
| 6 | 78% (52%–100%) | 100% (100%–100%) | +22.3 pp |
| 8 | 71% (42%–100%) | 100% (100%–100%) | +29.1 pp |
| 10 | 64% (33%–100%) | 100% (100%–100%) | +35.5 pp |
| 12 | 58% (26%–100%) | 100% (100%–100%) | +41.7 pp |
| 14 | 53% (20%–100%) | 100% (100%–100%) | +47.5 pp |
| 16 | 47% (15%–100%) | 100% (100%–100%) | +52.9 pp |
| 17 | 44% (13%–100%) | 100% (100%–100%) | +55.6 pp |

Conditions:
- `trust-chain` — no gate: a wrong step is accepted silently.
- `verify-chain` — gate plus repair at every step.

## Observed episodes

| condition | end-to-end correct | first failing subtask(s) |
|---|---|---|
| `trust-chain` | 1/3 (6%–79%, Wilson) | `bench_lru`, `bench_median` |
| `verify-chain` | 3/3 (44%–100%, Wilson) | — |

Paired at the terminal step: discordant 2/0 of 3 episodes · exact McNemar p = 0.5000 — **not significant**

## What this does and does not show

The curve is the compounding model evaluated at the measured per-task p. Independence holds by construction (each subtask is its own repository and its own model call), so the composition is exact — but it is a re-expression of the per-step measurement, not a second experiment. Composing cells into more orderings cannot create information.

The evidence is the 3 observed episodes. To reach p < 0.05 at the observed discordance rate, re-run `lha ablate` with about **9 repetitions** and regenerate this report.
