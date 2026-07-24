# Verification ablation

implementer: `claude_cli` · model: `claude-haiku-4-5-20251001` · tasks: 17 · repetitions: 3 · paired (trust/gate score the same attempt) · final scorer: `trusted-local` (fresh copy, canonical tests, independent of the internal gate)

| condition | n | claimed | true success (95% CI) | false success (95% CI) | mean repairs | errors |
|---|---|---|---|---|---|---|
| `trust` | 51 | 100% | 96% (90%–100%) | 4% (0%–10%) | 0.00 | 0 |
| `gate` | 51 | 96% | 96% (90%–100%) | 0% (0%–0%) | 0.00 | 0 |
| `verify` | 51 | 100% | 100% (100%–100%) | 0% (0%–0%) | 0.04 | 0 |

Conditions:
- `trust` — apply the first attempt and accept it; no gate, no repair.
- `gate` — apply it, run the internal test gate, refuse on failure.
- `verify` — gate plus repair loop.

Internal gate vs final scorer (per attempt):
- `gate`: TP=49 FP=0 TN=2 FN=0 · precision=100% recall=100% FPR=0% FNR=0%
- `verify`: TP=51 FP=0 TN=0 FN=0 · precision=100% recall=100% FPR=n/a FNR=0%

Paired contrasts:
- `trust` vs `gate` on false success: discordant 2/0 of 51 pairs · exact McNemar p = 0.50
- `gate` vs `verify` on true success: discordant 0/2 of 51 pairs · exact McNemar p = 0.50

Without the gate, 4% of accepted fixes are wrong (scorer-graded); the gate (same attempts) reduces that to 0%, and the repair loop raises true success from 96% to 100%. The gate discarded 0 correct fix(es) (false negatives).

## Per-task outcomes

| task | trust | gate | verify |
|---|---|---|---|
| `bench_basen` | pass | pass | pass |
| `bench_brackets` | pass | pass | pass |
| `bench_bsearch` | pass | pass | pass |
| `bench_caesar` | pass | pass | pass |
| `bench_csvlite` | pass | pass | pass |
| `bench_flatten` | pass | pass | pass |
| `bench_lru` | pass | pass | pass |
| `bench_median` | pass | pass | pass |
| `bench_merge_intervals` | pass | pass | pass |
| `bench_paginate` | pass | pass | pass |
| `bench_roman` | pass | pass | pass |
| `bench_rpn` | pass | pass | pass |
| `bench_runstats` | pass | pass | pass |
| `bench_slugify` | pass | pass | pass |
| `bench_spans` | pass | pass | pass |
| `bench_urljoin` | pass | pass | pass |
| `bench_window` | pass | pass | pass |

Legend: pass = true success · fail = refused · false-pass = claimed but wrong. Modal outcome across repetitions; outcomes are the final scorer's.
