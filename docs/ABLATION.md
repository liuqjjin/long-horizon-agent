# Verification ablation

`lha ablate` measures whether executable checks prevent incorrect code changes
from being delivered, and whether bounded repair recovers work the gate rejects.
It is a mechanism experiment, not a public-model leaderboard.

## Paired design

The fixed corpus contains 17 small Python repositories under `data/bench/`.
Each has a planted defect, a symptom-level task, and a canonical pytest oracle.
The corpus and its oracles must not be modified after observing model output.

For each `(task, repetition)`, the harness draws one first attempt and evaluates
that same attempt under:

| condition | gate | repair | delivered when |
|---|---|---|---|
| `trust` | off | none | the model returns a patch |
| `gate` | on | none | the original patch passes the checks |
| `verify` | on | bounded | the original or a repaired patch passes |

`trust` and `gate` therefore differ in verification, not in the sampled first
attempt. `verify` may make additional model calls after receiving concrete
failure evidence.

## Prediction and truth

The internal gate only predicts whether a patch should be accepted. It does not
grade itself.

For every evaluated output, the independent scorer:

1. freezes the effective source change and records its SHA-256;
2. creates a fresh copy of the canonical repository;
3. restores the original tests and protected configuration;
4. applies only the frozen source change;
5. runs pytest through a separate `ExecutionBackend` instance.

The scorer's result is `true_success`. A patch that is claimed as successful but
fails this scorer is `false_success`. Gate predictions are compared with truth
as TP, FP, TN, and FN.

For external or untrusted repositories, use
`--scorer-backend docker`. A `trusted-local` scorer is state-independent but not
host-isolated.

## Oracle protection

The first-attempt worktree excludes test files. Model output is reduced to its
effective source write set; changes to tests, `conftest.py`, package/build
configuration, or CI files are rejected by policy unless a task explicitly
allowlists an exact protected path.

The scorer starts from the canonical corpus rather than the gate's worktree. A
dirty or stale gate directory therefore cannot silently become ground truth.

Repair prompts may include compact pytest failure excerpts. That feedback is the
repair mechanism being measured; repaired output still has to pass the complete
canonical oracle.

## Codex no-tools protocol

The Codex CLI does not expose the same file-tool deny-list as the Claude CLI.
For an ablation attempt, LHA therefore audits the complete `codex exec --json`
stream and rejects the result if any tool item appears.

The parser also fails closed on:

- malformed JSONL or an unknown event type;
- an error or failed-turn event;
- a turn that starts but does not complete;
- a started item that never reaches a valid completion;
- a missing completed agent message;
- non-zero exit, timeout, authentication failure, or interrupted process.

Each call uses a temporary `HOME`, `CODEX_HOME`, workspace, and temp directory.
The parent environment is reduced to an explicit allowlist, the CLI process and
descendants are stopped before cleanup, and temporary authentication is removed
on every exit path.

The report records secret-free provenance: selected model, reasoning effort,
Codex CLI version, event summary, usage, outcome, Python and pytest versions,
Git state, source-tree digest, task digests, corpus digests, scorer backend, and
Docker image identity when applicable.

## Cache and transient failures

Completed non-error cells can be reused only when the provenance fingerprint
matches. The fingerprint binds the task bytes, corpus bytes, complete installed
`lha` source tree, model/backend settings, scorer settings, repair/retry
configuration, versions, and runtime details.

Protocol violations are deterministic failures and are not retried as transport
errors. Only failures explicitly classified as transient service or connection
problems use the configured bounded retry path.

If a transient problem persists, the cell is written as `ERROR`, shown in the
report, excluded from rate denominators, and never cached. It is not counted as
a model failure or quietly dropped from the record.

## Statistics

The report gives exact counts and rates per condition. For uncertainty:

- interior rates use a seeded task-cluster bootstrap because repetitions from
  one task are correlated;
- all-zero or all-one boundary rates use a Wilson score interval, because a
  percentile bootstrap would produce a false zero-width interval;
- paired contrasts use the exact two-sided McNemar test on matched
  `(task, repetition)` cells.

The gate confusion matrix and error count should be read alongside headline
rates. A zero observed false-success rate is not proof that the underlying rate
is exactly zero.

## Committed schema-v2 result

The committed report uses 17 fixed tasks and 12 repetitions, giving 204 paired
`(task, repetition)` cells. Its protocol is Codex CLI 0.141.0,
`gpt-5.4-mini`, low reasoning effort, read-only mode, and an independent Docker
scorer. All 204 cells were scored; the report contains 0 `ERROR` cells.

| condition | delivered | independently correct | incorrect delivery | result |
|---|---:|---:|---:|---|
| `trust` | 204 | 194 | 10 | every first attempt was delivered |
| `gate` | 194 | 194 | 0 | accepted 194 correct attempts and rejected all 10 incorrect attempts |
| `verify` | 204 | 204 | 0 | repaired the 10 rejected attempts before delivery |

For `trust` versus `verify`, the discordant counts are 10 in favor of
`verify` and 0 in favor of `trust`. The exact two-sided McNemar result is
`p = 0.001953125`, reported in prose as `p = 0.00195`.

These are mechanism results on the repository's fixed corpus. They are not a
Terminal-Bench, SWE-bench, or general model-quality score. Raw records,
provenance, configuration, and the generated table are in
`benchmarks/ablation_report.{json,md}`.

## Run the experiment

The committed protocol can be rerun after building the scorer image:

```bash
docker build -t lha:release .
LHA_CODEX_MODEL=gpt-5.4-mini \
LHA_CODEX_EFFORT=low \
LHA_CODEX_SANDBOX=read-only \
LHA_EXEC_IMAGE=lha:release \
uv run lha --llm codex_cli ablate \
  --reps 12 \
  --model gpt-5.4-mini \
  --scorer-backend docker \
  --out runs/ablation
```

The scorer image selected by `LHA_EXEC_IMAGE` must contain pytest and
`pytest-json-report`. The default slim Python image does not.

Output layout:

```text
runs/ablation/
  ablation_report.json
  ablation_report.md
  results/<task>__r<rep>.json
```

The per-cell files allow an interrupted run to continue only when their
fingerprint still matches. `ERROR` files are recomputed.

## Result policy

The summary above mirrors the current generated artifacts. The authoritative
values and full precision remain:

- `benchmarks/ablation_report.json` for raw records and provenance;
- `benchmarks/ablation_report.md` for the generated table;
- `benchmarks/horizon_report.*` for the separate horizon analysis.

Before publishing a result:

1. freeze the code and corpus;
2. use a new output directory or a matching fingerprint;
3. complete the registered repetition count;
4. review every `ERROR` and keep it visible;
5. confirm the report's Git state and source digest;
6. copy the generated reports without editing their numbers;
7. update every public statement that cites the old report.

Do not write a planned sample count as a completed result, substitute a
different model or scorer into this result, or report a Terminal-Bench or
SWE-bench score from this internal corpus.

Hermetic tests for pairing, scoring independence, cache invalidation, Wilson
boundaries, error handling, and report provenance are in
`tests/test_ablation.py` and `tests/test_codex_backend.py`.
