# Evaluation

The repository has three kinds of evaluation. They answer different questions
and must be reported separately.

| evaluation | grader | purpose |
|---|---|---|
| self-eval | registered checks in LHA | confirm the bundled workflows still run |
| verification ablation | an independent scorer | measure gate and repair behavior |
| public benchmarks | the official benchmark harness | compare on an external task set |

## Repository self-eval

Run all six fixed workflows:

```bash
uv sync
LHA_RUNS_DIR=runs/_eval uv run lha eval
```

The cases cover:

1. a code fix checked by Pytest and Ruff;
2. approval followed by resume in a new process;
3. stale context followed by reindexing;
4. required context with its backend deliberately unavailable;
5. an experiment checked by recomputed metrics and a fresh rerun;
6. an unreachable metric threshold that must end as `FAILED`.

`lha eval` is a repository regression check. It is not an external agent
benchmark and should not be reported as one.

## Verification ablation

The committed schema-v2 experiment uses 17 fixed Python defects and 12
repetitions. One first attempt is shared across `trust`, `gate`, and `verify`;
only `verify` may repair after a failed check.

The independent scorer applies the frozen source change to a fresh canonical
repository and runs the original tests in Docker. It does not reuse the gate's
verdict. The report contains no `ERROR` cells. Counts and paired statistics are
kept in the generated report and summarized in [ABLATION.md](ABLATION.md). They
describe the gate and repair mechanism on this corpus, not general model quality.

Reproduce the configured run with:

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

The generated JSON is authoritative:

- [`benchmarks/ablation_report.json`](../benchmarks/ablation_report.json)
  contains rows, provenance, errors, and full-precision statistics;
- [`benchmarks/ablation_report.md`](../benchmarks/ablation_report.md) is its
  rendered table;
- [ABLATION.md](ABLATION.md) describes pairing, scoring, and cache rules;
- [HORIZON.md](HORIZON.md) explains the separate composition analysis.

## SWE-bench Verified adapter

`lha.bench.swebench` writes the official prediction fields
(`instance_id`, `model_name_or_path`, `model_patch`) and parses schema-v2
reports. Duplicate instance IDs are rejected, and evaluator errors stay in the
denominator.

The adapter never sees held-out `FAIL_TO_PASS` tests. Those tests remain with
the official evaluator. No SWE-bench score is published in this repository.

## Terminal-Bench 2.1 fixed subset

The adapter uses Harbor 0.20 with the official
[`terminal-bench/terminal-bench-2-1`](https://hub.harborframework.com/datasets/terminal-bench)
dataset. The formal run is being hardened and executed; no Terminal-Bench score
is published yet.

The protocol is fixed before the first model run:

- download all 89 official task definitions;
- sort task names by `(SHA-256(instance_id), instance_id)`;
- use the first 20 tasks as the scored subset and the next three for smoke
  testing;
- finish the three smoke jobs before starting the scored jobs;
- run one task and one attempt per Harbor job, with Harbor retries set to zero;
- do not retry task setup, credential staging, agent installation, or the task itself;
- Codex itself may retry one failure while establishing a request, but its
  response-stream retry is disabled;
- the credential broker buffers at most 16 MiB through a valid
  `response.completed` event before returning any bytes. For allowlisted
  post-header stream failures it may reopen the identical upstream request up
  to four times, subject to one shared limit of 12 retries for the whole task
  attempt. Failed attempts' partial bytes are discarded. The receipt records
  accepted client requests, physical upstream requests, retried requests, and
  the largest retry count on any one request separately;
- if the upstream omits `Content-Type`, the broker requires every data-bearing
  frame to be a structurally valid Responses SSE event and still requires a
  valid `response.completed` event before supplying the canonical
  `text/event-stream` type downstream. An explicit type remains restricted to
  SSE; this rule does not accept arbitrary untyped response bodies;
- do not retry deterministic policy failures, malformed responses, task
  failures, or model failures;
- allow one `codex exec`, at most 1,800 seconds, 60 accepted model requests,
  at most 12 additional broker attempts after interrupted streams, and 128
  audited tool items per task;
- freeze the model, reasoning effort, Harbor and Codex versions, selected task
  names, wheel and binary hashes, and container-image digests.

The standalone Codex binary is the Linux x86-64 build used inside the task
containers, not the host macOS executable. Host authentication stays in the
credential broker. A task receives only a short-lived, attempt-bound capability
and the broker's TLS certificate; its temporary `CODEX_HOME` contains no host
credential copy and is removed on every exit path. Credential bytes, capability
values, and private paths are excluded from public artifacts.

The implementation is in `src/lha/bench/terminal_bench.py`:

- `create_protocol()` writes the preregistration;
- `build_harbor_commands()` creates exactly three smoke jobs or 20 scored jobs;
- `validate_harbor_results()` checks job configuration, trials, task IDs,
  retries, image attestations, and agent audit files;
- `derive_terminal_bench_records()` reads official verifier results;
- `summarize_records()` revalidates the source files before writing a summary.

Missing verifier output and protocol failures remain `ERROR` and stay in the
20-task denominator. The result may be described only as
“Terminal-Bench 2.1 fixed 20-task subset,” not as a full leaderboard score.

This Harbor adapter calls Codex directly inside the task container. It does not
run LHA's internal gate or repair loop. Therefore interception counts, false
rejections, repair success, and internal model-call counts are not measured and
must be reported as unavailable rather than zero.

The 128-item limit is only a guard against runaway output. It is not described
as “128 execution steps”: Codex can emit several parallel command items in one
model response, while LHA's own step counter is not part of this direct Harbor
adapter.

Build and protocol contract tests are in
`tests/test_bench_adapters.py`. A score can be published only after the official
result files, protocol, execution manifest, hashes, and summary are committed
together.
