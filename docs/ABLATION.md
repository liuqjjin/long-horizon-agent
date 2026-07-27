# Verification ablation

`lha ablate` measures two questions on a fixed internal corpus:

1. do executable checks stop incorrect patches from being delivered?
2. can a bounded repair recover patches rejected by those checks?

It is not a public-model leaderboard.

## Paired design

The corpus contains 17 small Python repositories under `data/bench/`. Each has
a planted defect, a symptom-level task, and a fixed Pytest oracle. At startup,
the task file and repository are copied to a content-addressed input snapshot.
Every cell reads that snapshot rather than the live corpus directory.

For each `(task, repetition)`, one first attempt is reused across three
conditions:

| condition | internal gate | repair | delivery rule |
|---|---|---|---|
| `trust` | off | no | deliver the first patch |
| `gate` | on | no | deliver only when the first patch passes |
| `verify` | on | bounded | deliver the first or repaired patch after it passes |

This keeps the first attempt paired. `verify` may make additional calls only
after receiving check failures.

## Independent scoring

The internal gate decides whether LHA may advance; it does not supply the
experiment's truth label.

For every output, the scorer:

1. freezes the effective source change and records its SHA-256;
2. creates a new copy of the canonical repository;
3. restores original tests and protected configuration;
4. applies only the frozen source change;
5. runs the canonical tests through a separate execution backend.

The report keeps two fields separate:

- `artifact_correct`: the scorer's verdict on the frozen patch;
- `true_success`: the condition delivered the patch and `artifact_correct` is
  true.

A correct patch rejected by `gate` therefore has
`artifact_correct=true` and `true_success=false`. Gate TP, FP, TN, and FN use
`artifact_correct`. Chain success and delivered-success rates use
`true_success`.

Before a candidate patch is applied, the scorer collects the canonical Pytest
node IDs. The scored process must then produce all of the following:

- the expected node-ID set;
- a normal Pytest return code;
- per-phase hook records;
- a post-session receipt containing a random nonce;
- a matching content-addressed scorer-evidence file.

The receipt is written only after `pytest.main` returns. Printing a forged
summary and calling `os._exit(0)` produces no receipt and is classified as
`ERROR`, not pass.

This is not containment against hostile Python running with the scorer's UID.
Such code can inspect the process and its writable mount; Docker protects the
host but does not by itself make the in-container oracle immutable. The current
ablation assumes fixed, non-adversarial corpus programs. A hostile-code scorer
would additionally need a separate control process and read-only oracle mount.
`trusted-local` must only be used for repositories already trusted by the user.

The first-attempt workspace excludes tests. Changes to tests, `conftest.py`,
package or build configuration, and CI files are rejected unless an exact path
is allowlisted. The scorer starts from the canonical corpus, not the gate's
working directory.

## Codex protocol

The ablation path asks Codex for a single no-tools attempt. The complete JSONL
stream is audited, and any tool item invalidates the attempt. The parser also
rejects:

- malformed JSON or an unknown event;
- an error or failed turn;
- a turn or item without a valid completion;
- a missing completed agent message;
- non-zero exit, timeout, authentication failure, or interruption.

Each call gets a temporary home, `CODEX_HOME`, workspace, and temporary
directory. The parent environment is reduced to an allowlist. The process group
is stopped before temporary credentials are removed.

Other model backends remain available for exploratory runs, but the public
release check accepts only evidence produced by the hardened `codex_cli`
backend. Formal reports also require Docker for both the prediction-side gate
and the independent scorer, with the scorer image bound by its immutable image
ID.

The report records model and CLI settings, event summary, usage, runtime
versions, Git state, source and task digests, scorer backend, and container
identity where applicable. It does not record credentials.

## Errors, cache, and statistics

A completed cell can be reused only if its schema-6 cache fingerprint still
matches and its patch artifact and scorer receipt both validate. The
fingerprint covers the input snapshot, LHA source, model settings, scorer,
repair and retry settings, and runtime versions.

Only errors classified as transient service or connection failures use the
configured bounded retry path. A persistent failure is written as `ERROR`,
remains visible, is excluded from rate estimates, and is never cached.

The report uses:

- a seeded task-cluster bootstrap for interior rates;
- Wilson score intervals for all-zero or all-one rates;
- an exact two-sided McNemar test for paired contrasts.

An observed rate of zero incorrect deliveries is not proof that the underlying
rate is zero.

## Report status

Schema 4 is the first report format that binds delivery correctness, immutable
inputs, patch artifacts, and scorer receipts. Reports through schema 3 remain
readable as historical records but cannot be published as current formal
evidence.

The 17-task × 12-repetition run must be repeated under schema 4 before this
document or the README states a current result. Values from the older schema-2
run are not carried forward.

## Reproduce

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

The scorer image must contain Pytest and `pytest-json-report`.

Output:

```text
runs/ablation/
  ablation_report.json
  ablation_report.md
  input_snapshots/<sha256>/
  artifacts/<sha256>.json
  scorer_evidence/<sha256>.json
  results/<task>__r<rep>.json
```

Before publishing a result, finish the registered repetitions, keep every
`ERROR`, and commit the JSON report, generated Markdown, patch artifacts, and
scorer evidence together. `release_claims` recomputes the LHA source tree plus
the task and corpus digests from the checkout.

The authoritative committed files are
[`benchmarks/ablation_report.json`](../benchmarks/ablation_report.json) and
[`benchmarks/ablation_report.md`](../benchmarks/ablation_report.md).
Statistical and recovery tests are in `tests/test_ablation.py` and
`tests/test_codex_backend.py`.
