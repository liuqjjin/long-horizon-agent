# Verification ablation

implementer: `claude_cli` · model: `haiku` · tasks: 11 · repetitions: 3 · paired (trust/gate score the same attempt)

| condition | claimed | true success | false success | mean repairs |
|---|---|---|---|---|
| `trust` | 100% | 61% | 39% | 0.00 |
| `gate` | 61% | 61% | 0% | 0.00 |
| `verify` | 85% | 85% | 0% | 0.39 |

Conditions:
- `trust` — apply the first attempt and accept it; no gate, no repair.
- `gate` — apply it, run the test gate, refuse on failure.
- `verify` — gate plus repair loop.

Without the gate, 39% of accepted fixes are wrong; the gate (same attempts) drives that to 0%, and the repair loop raises true success from 61% to 85%.

## Per-task outcomes

| task | trust | gate | verify |
|---|---|---|---|
| `bench_basen` | false-pass | fail | pass |
| `bench_brackets` | pass | pass | pass |
| `bench_bsearch` | pass | pass | pass |
| `bench_caesar` | pass | pass | pass |
| `bench_flatten` | pass | pass | pass |
| `bench_median` | pass | pass | pass |
| `bench_merge_intervals` | false-pass | fail | fail |
| `bench_paginate` | pass | pass | pass |
| `bench_roman` | false-pass | fail | pass |
| `bench_rpn` | pass | pass | pass |
| `bench_window` | false-pass | fail | fail |

Legend: pass = true success · fail = refused · false-pass = claimed but wrong. Modal outcome across repetitions.
