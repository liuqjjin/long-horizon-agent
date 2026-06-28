# Verification Ablation

implementer: `claude_cli` · model: `haiku` · tasks: 11 · repetitions: 3 · paired (trust/gate score the same attempt)

| condition | claimed | **true success** | **false success** | mean repairs |
|---|---|---|---|---|
| `trust` | 100% | **61%** | **39%** | 0.00 |
| `gate` | 61% | **61%** | **0%** | 0.00 |
| `verify` | 85% | **85%** | **0%** | 0.39 |

_condition legend:_
- `trust` — apply the first attempt and accept it — no objective gate, no repair
- `gate` — apply it, run the test gate, refuse on failure — catch but don't fix
- `verify` — the full harness: gate + repair loop

**Headline.** On the same first attempts, orchestrate-and-trust ships **39%** silently-wrong fixes; the objective gate drives that to **0%**. The repair loop then lifts true success from **61%** (first try) to **85%** (full harness).

## Per-task outcomes

| task | trust | gate | verify |
|---|---|---|---|
| `bench_basen` | ⚠ | ✗ | ✓ |
| `bench_brackets` | ✓ | ✓ | ✓ |
| `bench_bsearch` | ✓ | ✓ | ✓ |
| `bench_caesar` | ✓ | ✓ | ✓ |
| `bench_flatten` | ✓ | ✓ | ✓ |
| `bench_median` | ✓ | ✓ | ✓ |
| `bench_merge_intervals` | ⚠ | ✗ | ✗ |
| `bench_paginate` | ✓ | ✓ | ✓ |
| `bench_roman` | ⚠ | ✗ | ✓ |
| `bench_rpn` | ✓ | ✓ | ✓ |
| `bench_window` | ⚠ | ✗ | ✗ |

Legend: ✓ true success · ✗ failed (refused) · ⚠ **false success** (claimed but wrong). Modal outcome across repetitions.
