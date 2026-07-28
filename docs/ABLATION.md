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
Such code can inspect the process and its writable mount. Docker reduces the
host's exposure, but the remaining boundary still depends on the image, mounts,
network, and container permissions; it does not make the in-container oracle
immutable. The current ablation assumes fixed, non-adversarial corpus programs.
A hostile-code scorer would additionally need a separate control process and
read-only oracle mount. `trusted-local` must only be used for repositories
already trusted by the user.

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
is stopped before temporary credentials are removed on normal return, failure,
timeout, or handled interruption. `SIGKILL`, a kernel crash, or power loss can
bypass this cleanup and leave a mode-protected temporary directory.

Other model backends remain available for exploratory runs, but the public
release check accepts only evidence produced by `codex_cli` with its JSONL and
process-cleanup checks enabled. Formal reports also require Docker for both the
prediction-side gate and the independent scorer, with the scorer image bound by
its immutable image ID.

The report records model and CLI settings, event summary, usage, runtime
versions, Git state, source and task digests, scorer backend, and container
identity where applicable. It does not record credentials.

## Formal attempt registration

The formal 17-task × 12-repetition schedule is different from an exploratory
`lha ablate` run. Before execution, `benchmarks/formal_ablation_attempts.json`
must contain one open `REGISTERED` event that fixes:

- the source commit and source-tree digest;
- the corpus manifest;
- the model, reasoning effort, Codex CLI version and executable digest;
- no-tools, sandbox, permission, retry, timeout, and backoff settings;
- the immutable Docker image ID;
- the single-use output path and witness remote.

The registration commit must directly follow the source commit and may change
only the registry. At startup the runner creates a deterministic witness commit
and pushes it to the attempt-specific remote ref only if that ref does not
already exist. The witness binds the registration digest, protocol digest,
random outcome key, and run-header digest. The run proceeds only after
`ls-remote` confirms the exact ref. The release check later requires the same
remote ref to remain available.

The registry is an append-only state machine. `COMPLETED` binds one report to
the registration. A preflight failure, interruption, or incomplete schedule is
recorded as `ABANDONED`. Registering an equivalent source tree, corpus, model,
CLI/client configuration, and image again is rejected even if a new empty Git
commit is created.

This witness records that the registered attempt started. It does not make the
remote service, host, Docker daemon, or private Git history trustworthy.

## Errors, cache, and statistics

Exploratory runs may reuse a completed cell when its cache-key format v8
fingerprint and all referenced artifacts and receipts validate. The fingerprint
covers the input snapshot, LHA source, model settings, scorer, repair and retry
settings, and runtime versions.

Formal runs do not read cache and do not resume. Their output directory must be
new and contain only the full-run lock before initialization. The runner writes
one run header, then one start marker and one terminal seal for each of the 204
cells. Existing markers, terminal files, or copied exploratory cache entries
stop the run.

The formal schema-4 report uses `ERROR` for one narrow case: every bounded
first-call attempt failed before Codex produced a patch. Each failed call is
saved as a content-addressed receipt. The three conditions then share one
cell-level `ERROR`; it remains in the scheduled total but is excluded from rate
estimates and paired tests.

If a patch has been produced and a later repair, scorer, filesystem, or
container operation fails, the formal attempt stops instead of converting the
partial cell to `ERROR`. Any interruption consumes the registered attempt; the
partial directory is retained for audit and cannot be resumed.

The report separates scheduled cells, usable paired cells, and cell-level
errors. Rate denominators use usable cells. A repetition with any unavailable
task is excluded from the complete-episode analysis; the descriptive
composition reports the actual per-task sample counts and adds no observations.

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

The committed report must cover the complete registered 17-task ×
12-repetition schedule, contain no cache hits, and match one completed
registration and its remote start witness. Values from the older schema-2 run
are not carried forward into a schema-4 claim.

The current committed `ablation_report.json` is the historical schema-2 record.
No final schema-4 count should be quoted until the registered attempt finishes
and all required evidence is committed.

## Reproduce

First commit the source and corpus manifest. Then append the exact
`REGISTERED` event, including the fresh attempt ID, output path
`runs/formal_ablation/<attempt-id>`, and witness remote, in a registration-only
commit. Push both commits before starting. The following command is valid only
for that clean registered checkout:

```bash
docker build -t lha:release .
ATTEMPT_ID=replace-with-the-registered-64-hex-id
LHA_CODEX_MODEL=gpt-5.4-mini \
LHA_CODEX_EFFORT=low \
LHA_CODEX_SANDBOX=read-only \
LHA_EXEC_IMAGE=lha:release \
uv run lha --llm codex_cli ablate \
  --reps 12 \
  --model gpt-5.4-mini \
  --scorer-backend docker \
  --out "runs/formal_ablation/$ATTEMPT_ID"
```

The scorer image must contain Pytest and `pytest-json-report`.

Output:

```text
runs/formal_ablation/<attempt-id>/
  .formal-ablation.lock
  formal_run.json
  ablation_report.json
  ablation_report.md
  input_snapshots/<sha256>/
  artifacts/<sha256>.json
  scorer_evidence/<sha256>.json
  llm_call_receipts/<sha256>.json
  results/<task>__r<rep>.started.json
  results/<task>__r<rep>.json
```

Before publishing a result, finish the registered repetitions, keep every
`ERROR`, and commit the JSON report, generated Markdown, patch artifacts,
scorer evidence, LLM call receipts, run header, terminal seals, and matching
`COMPLETED` registry event together. `release_claims` recomputes the LHA source
tree plus task and corpus digests, requires all 204 fresh start/terminal pairs,
rejects cache hits and unreferenced receipts, verifies the registration history,
and confirms the remote witness ref.

The generated files for the historical schema-v2 run are
[`benchmarks/ablation_report.json`](../benchmarks/ablation_report.json) and
[`benchmarks/ablation_report.md`](../benchmarks/ablation_report.md).
Statistical and recovery tests are in `tests/test_ablation.py` and
`tests/test_codex_backend.py`.
