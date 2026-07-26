# Verification ablation

implementer: `codex_cli` (codex-cli 0.141.0 model=gpt-5.4-mini effort=low sandbox=read-only) · model: `gpt-5.4-mini` · tasks: 17 · repetitions: 12 · paired (trust/gate score the same attempt) · final scorer: `docker` (fresh copy, canonical tests, independent of the internal gate)

| condition | n | claimed | true success (95% CI) | false success (95% CI) | mean repairs | errors |
|---|---|---|---|---|---|---|
| `trust` | 204 | 100% | 95% (87%–100%) | 5% (0%–13%) | 0.00 | 0 |
| `gate` | 204 | 95% | 95% (87%–100%) | 0% (0%–2%) | 0.00 | 0 |
| `verify` | 204 | 100% | 100% (98%–100%) | 0% (0%–2%) | 0.05 | 0 |

Conditions:
- `trust` — apply the first attempt and accept it; no gate, no repair.
- `gate` — apply it, run the internal test gate, refuse on failure.
- `verify` — gate plus repair loop.

Internal gate vs final scorer (per attempt):
- `gate`: TP=194 FP=0 TN=10 FN=0 · precision=100% recall=100% FPR=0% FNR=0%
- `verify`: TP=204 FP=0 TN=0 FN=0 · precision=100% recall=100% FPR=n/a FNR=0%

Paired contrasts:
- `trust` vs `gate` on false success: discordant 10/0 of 204 pairs · exact McNemar p = 0.00
- `gate` vs `verify` on true success: discordant 0/10 of 204 pairs · exact McNemar p = 0.00

Without the gate, 5% of accepted fixes are wrong (scorer-graded); the gate (same attempts) reduces that to 0%, and the repair loop raises true success from 95% to 100%. The gate discarded 0 correct fix(es) (false negatives).

## Per-task outcomes

| task | trust | gate | verify |
|---|---|---|---|
| `bench_basen` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_brackets` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_bsearch` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_caesar` | pass 9/12 · false-pass 3/12 | pass 9/12 · fail 3/12 | pass 12/12 |
| `bench_csvlite` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_flatten` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_lru` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_median` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_merge_intervals` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_paginate` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_roman` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_rpn` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_runstats` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_slugify` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_spans` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_urljoin` | pass 5/12 · false-pass 7/12 | pass 5/12 · fail 7/12 | pass 12/12 |
| `bench_window` | pass 12/12 | pass 12/12 | pass 12/12 |

Legend: pass = true success · fail = refused · false-pass = claimed but wrong. Exact counts across repetitions; outcomes are the final scorer's.

Provenance:
- source: `fef2eb8d4c6fa0e30147da92379f4343ead2e1988e9845cd9a19e7d06ebf6c97`
- git: `a1e72ffaac8690ee64f9477ab8ae0bf5cc652298` · dirty: `no`
- runtime: Python `3.11.15` · pytest `9.1.1`
- LLM call audits: 214 · loaded from cell cache: 214
