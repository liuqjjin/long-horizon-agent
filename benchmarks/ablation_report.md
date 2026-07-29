# Verification ablation

implementer: `codex_cli` (codex-cli 0.141.0 model=gpt-5.3-codex-spark effort=high sandbox=read-only permission_model=profile cli_sha256=51f848c212ee24e8da923a7175813a74c113d47e01f0d40f1fea46b12644c363 cli_trusted=false) · model: `gpt-5.3-codex-spark` · tasks: 17 · repetitions: 12 · paired (trust/gate score the same attempt) · final scorer: `docker` (fresh copy, canonical tests, independent of the internal gate)

| condition | n | delivered | artifact correct (95% CI) | delivered correct (95% CI) | delivered wrong (95% CI) | mean repairs | errors |
|---|---|---|---|---|---|---|---|
| `trust` | 204 | 100% | 99% (97%–100%) | 99% (97%–100%) | 1% (0%–3%) | 0.00 | 0 |
| `gate` | 204 | 99% | 99% (97%–100%) | 99% (97%–100%) | 0% (0%–2%) | 0.00 | 0 |
| `verify` | 204 | 100% | 100% (98%–100%) | 100% (98%–100%) | 0% (0%–2%) | 0.01 | 0 |

`n` counts usable measurements. `errors` stay in the scheduled denominator (`n + errors`) and are never relabelled as incorrect patches.

Conditions:
- `trust` — apply the first attempt and accept it; no gate, no repair.
- `gate` — apply it, run the internal test gate, refuse on failure.
- `verify` — gate plus repair loop.

Internal gate vs artifact correctness (per attempt):
- `gate`: TP=201 FP=0 TN=3 FN=0 · precision=100% recall=100% FPR=0% FNR=0%
- `verify`: TP=204 FP=0 TN=0 FN=0 · precision=100% recall=100% FPR=n/a FNR=0%

Paired contrasts:
- `trust` vs `gate` on false success: task-cluster exact paired sign-flip p = 0.2500 (3/17 tasks have a non-zero paired effect) · descriptive cell discordance 3/0 of 204; no cell-level inferential p-value
- `gate` vs `verify` on true success: task-cluster exact paired sign-flip p = 0.2500 (3/17 tasks have a non-zero paired effect) · descriptive cell discordance 0/3 of 204; no cell-level inferential p-value

Without the gate, 1% of accepted fixes are wrong (scorer-graded); the gate (same attempts) reduces that to 0%, and the repair loop raises true success from 99% to 100%. The gate discarded 0 independently correct artifact(s) (false negatives).

## Per-task outcomes

| task | trust | gate | verify |
|---|---|---|---|
| `bench_basen` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_brackets` | pass 11/12 · false-pass 1/12 | pass 11/12 · fail 1/12 | pass 12/12 |
| `bench_bsearch` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_caesar` | pass 11/12 · false-pass 1/12 | pass 11/12 · fail 1/12 | pass 12/12 |
| `bench_csvlite` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_flatten` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_lru` | pass 11/12 · false-pass 1/12 | pass 11/12 · fail 1/12 | pass 12/12 |
| `bench_median` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_merge_intervals` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_paginate` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_roman` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_rpn` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_runstats` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_slugify` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_spans` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_urljoin` | pass 12/12 | pass 12/12 | pass 12/12 |
| `bench_window` | pass 12/12 | pass 12/12 | pass 12/12 |

Legend: pass = delivered and independently correct · fail = not delivered · false-pass = delivered but independently wrong. Artifact correctness is reported separately from delivery.

Provenance:
- source: `05ede9d65d4db1a16e66894dab0d7ac96678dedf3e161711404426ea7090f7d7`
- git: `95cc1a109c6a4b479e3390a95f437d31d94060f8` · dirty: `no`
- formal attempt: `56fed4eb17d7e43fbb3d73e21b8ab464489bc19e1ae05731137e7e157ad5f00f`
- registration: `95cc1a109c6a4b479e3390a95f437d31d94060f8` · registry: `benchmarks/formal_ablation_attempts.json`
- runtime: Python `3.11.15` · pytest `9.1.1`
- LLM call audits: 207 · loaded from cell cache: 0
