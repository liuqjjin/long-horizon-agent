# Benchmarks

Three layers, by who grades the work:

1. **Self-eval** (below) — the harness checking its own six workflows on a
   deterministic stub. Runs in CI.
2. **Verification ablation** ([ABLATION.md](ABLATION.md)) — a real LLM through the
   harness, graded by an independent scorer. Committed snapshot in
   [`benchmarks/`](../benchmarks/).
3. **Public benchmarks** (bottom of this page) — SWE-bench Verified and
   Terminal-Bench 2.1 adapters, graded by their official harnesses. **No runs have
   been executed yet; no numbers are claimed.** The adapters and their contract
   tests are in `src/lha/bench/` and `tests/test_bench_adapters.py`.

# Self-eval

The harness checking itself: six workflows, each with an objective pass/fail. It
exercises the project's thesis end to end — a step is "done" only when an external
oracle says so.

## Reproduce

```bash
uv sync
uv run lha eval            # all six tasks; writes runs/eval_report.json
uv run lha eval --quick    # the three fast cases (skips the experiment runs)
```

Every number below is produced by that command in this repo. To re-derive the
per-task oracle, read `src/lha/eval.py` (one function per case) and the task specs
under `data/tasks/`.

## Results

Verbatim output of `uv run lha eval`:

```
# Self-eval — 6/6

| dimension | case | result | detail |
|---|---|---|---|
| issue-to-PR | fix_average | PASS | status=DONE verified=True |
| resume | pause_resume | PASS | first=AWAITING_APPROVAL resumed=DONE |
| freshness | edit_reindex | PASS | initial_fresh=True stale_after_edit=True fresh_after_reject=True |
| fail-closed context | required_context_unavailable | PASS | status=FAILED verdict_named_the_reason=True |
| paper-to-experiment | bicubic_sr | PASS | status=DONE verified=True |
| verification-ablation | strict_threshold_caught | PASS | status=FAILED psnr_correctly_rejected=True reached_psnr_step=True |

score: 6/6
```

The release-candidate gate also produced `523 passed, 3 skipped` with 83%
statement coverage.

## What each task verifies

| # | Dimension | Task | Objective oracle | Pass condition |
|---|-----------|------|------------------|----------------|
| 1 | issue-to-PR | `fix_average` | a real `pytest` run + `ruff` on the patched sandbox | run reaches `DONE` **and** `verify.json.passed` (tests pass, lint clean) |
| 2 | resume | `pause_resume` | re-entering an approval-paused run in a fresh harness | first run `AWAITING_APPROVAL`, persisted approval, then `resume` → `DONE` + verified |
| 3 | freshness | `edit_reindex` | mtime/index-generation vs. source, then incremental reindex | context `fresh → stale (after edit) → fresh (after reject_stale)` |
| 4 | paper-to-experiment | `bicubic_sr` | PSNR/SSIM **recomputed from the saved arrays** + a reproducibility re-run | `DONE` + verified (PSNR ≥ 24 dB, SSIM ≥ 0.80, deterministic re-run, seed/versions recorded) |
| 5 | verification-ablation | `strict_threshold_caught` | the PSNR verifier against an unreachable bar | run is `FAILED`, the `psnr` check failed, and the experiment step was actually reached |
| 6 | fail-closed context | `required_context_unavailable` | a step that requires context against a backend forced dark | run is `FAILED` **and** the `freshness` check names the unavailable context — a failure for the right reason, not any failure |

Tasks 1, 4, 5, and 6 live in `data/tasks/*.yaml`; tasks 2 and 3 are driven directly
in `src/lha/eval.py`. Verifier thresholds are explicit in the task specs
(`psnr_min: 24.0`, `ssim_min: 0.80`, `data_range: 1.0`).

## The verification-ablation case

This case shows the verifier changes the outcome.

The bundled experiment (`data/sample_experiment/experiment.py`) is a deterministic
bicubic 4× super-resolution baseline on `skimage.data.astronaut()`. The
experiment verifiers **recompute** the metrics from the saved output (they do not
trust the experiment's self-reported numbers):

- **PSNR ≈ 25.07 dB, SSIM ≈ 0.8246** (`data_range = 1.0`).

The normal task (`run_sr_experiment.yaml`) asks for `psnr_min: 24.0` → the harness
verifies and reports `DONE`. The ablation task
(`run_sr_experiment_strict.yaml`) asks for an unreachable `psnr_min: 40.0`:

- **With** the PSNR verifier (the harness): the recomputed 25.07 dB < 40 dB, the
  `psnr` check fails, and the run is reported **`FAILED`** — the agent refuses to
  claim a result it cannot verify.
- **Without** a verifier (a typical orchestrate-and-trust agent): the same run
  would end "successfully" with a wrong 25 dB result reported as a pass.

That gap (`FAILED` vs. a false `DONE`) is what the verifier buys, on a runnable task.
It also guards against fabricated metrics: because the verifier recomputes from the
arrays, a doctored `metrics.json` is caught (see
`tests/test_experiment_verifiers.py::test_psnr_catches_fabricated_metric`).

## Honesty notes

- **Scope.** This is a small self-check on bundled tasks — it measures that
  the harness's own workflows behave correctly end-to-end, not performance against
  an external SWE/agent benchmark. No external leaderboard numbers are claimed.
- **Reproducibility.** From a clean checkout, `uv sync && uv run lha eval` (run from
  the repo root) reproduces `6/6` — verified by deleting all gitignored generated
  state (`runs/`, `data/.lha_index/`, `data/skills/`) and re-running. The first run
  downloads a small sentence-transformers model (~tens of MB, one-time).
- **Environment independence.** Every case asserts the same thing with or without a
  code-search backend. The loop cases (1, 2) declare retrieval optional and are
  graded by a real `pytest` run; case 6 forces the backend dark rather than
  depending on whether `ccc` is installed. An earlier version loaded task 1 with
  its default `context_requirement: required`, so it scored 5/5 on a machine with
  `ccc` and 3/5 on CI — the harness was right to fail closed, and the claim was
  the thing that was wrong.
- **Determinism.** The experiment is seeded and deterministic, and the freshness
  case is tested via index-generation timestamps rather than wall-clock races.

# Verification ablation (schema v2)

The committed ablation uses 17 fixed tasks × 12 repetitions = 204 paired cells.
Codex CLI 0.141.0 ran `gpt-5.4-mini` with low reasoning effort in read-only
mode. A separate Docker execution backend supplied the truth labels, and all
204 cells completed with 0 `ERROR` cells.

| condition | delivered | independently correct | incorrect delivery | outcome |
|---|---:|---:|---:|---|
| `trust` | 204 | 194 | 10 | accepted every first attempt |
| `gate` | 194 | 194 | 0 | accepted 194 correct attempts and blocked all 10 incorrect attempts |
| `verify` | 204 | 204 | 0 | repaired the 10 blocked attempts before delivery |

For `trust` versus `verify`, the exact two-sided McNemar result is
`p = 0.001953125` (`p = 0.00195` in prose). Across the 12 measured
whole-corpus repetitions, `trust` completed 2/12 episodes and `verify`
completed 12/12.

The horizon composition is a separate model-based projection over these
measured cells. It adds 0 independent samples and is not evidence from
additional long-task executions. See [ABLATION.md](ABLATION.md),
[HORIZON.md](HORIZON.md), and the generated reports under `benchmarks/`.

# Public benchmarks (adapters ready, not yet run)

`src/lha/bench/` connects the harness to two public evaluators. In both, the
grading is done by the official harness on frozen predictions — the same
prediction/truth separation the ablation uses. **No evaluation runs have been
executed; the tables above contain the only measured numbers in this repo.**
Running either benchmark costs real model calls and needs Docker.

## SWE-bench Verified

Dataset `SWE-bench/SWE-bench_Verified` (500 instances), evaluated by
`swebench` ≥ 4.1. The adapter writes predictions in the official three-field
JSONL (`instance_id`, `model_name_or_path`, `model_patch`), refuses duplicate
instance ids, and parses the official `schema_version: 2` report with
evaluation ERRORs kept in the denominator (`resolved_rate = resolved /
submitted`, never `resolved / completed`).

```bash
uv sync --extra bench
# 1. produce predictions with the harness (one lha run per instance), then:
python -c "
from lha.bench import write_predictions, eval_command
from lha.bench.swebench import prediction_from_run
preds = [prediction_from_run('runs/<run_id>', '<instance_id>', 'lha+<model-id>')]
print(' '.join(eval_command(write_predictions(preds, 'preds.jsonl'), run_id='lha-v0')))
"
# 2. run the printed official command; on Apple silicon add namespace='' so
#    images build locally (upstream images are x86_64).
```

The internal gate may run the target repo's own tests, but it never sees
SWE-bench's held-out FAIL_TO_PASS tests — those are applied by the official
harness inside its own containers.

## Terminal-Bench 2.1 fixed-subset protocol (Harbor)

The adapter targets the official
[`terminal-bench/terminal-bench-2-1`](https://hub.harborframework.com/datasets/terminal-bench)
dataset with Harbor 0.20. No scored run has been executed, so this repository
does not claim a Terminal-Bench score.

The evaluation protocol is fixed before model execution:

- sort all official task names by lowercase hexadecimal
  `SHA-256(instance_id)`; the first 20 are scored and the next three are
  separate smoke tasks;
- create one Harbor job for each selected task, with one attempt, one task and
  Harbor retries disabled; this makes the submitted set inspectable instead of
  relying on an operator-supplied free-form filter;
- bind the full task name to the agent and compare it with Harbor's trial
  session before Codex starts;
- allow one shared retry across pre-agent installation and credential staging.
  Once Codex starts, neither a task failure nor a protocol failure is retried;
- enforce one `codex exec`, a 1,800-second deadline and at most 20 audited tool
  actions. A missing completion event, malformed JSON, unknown event or
  non-zero process exit is `ERROR`;
- record the exact model, reasoning effort, Harbor and Codex versions, LHA
  wheel digest, standalone Codex binary digest, selected task names, and a
  separate container-image digest for every selected task.

The 20-tool limit is an adapter acceptance limit over Codex's public JSONL
events. The CLI does not expose a reliable count of its internal model calls
or a repair counter, so this protocol does not claim limits it cannot measure.

Build the wheel, pin Harbor, and download the official dataset metadata:

```bash
uv build --wheel
uvx --python 3.12 --with harbor==0.20.0 \
  harbor datasets download terminal-bench/terminal-bench-2-1 \
  --output-dir .terminal-bench-metadata
```

Use the Linux x86-64 standalone Codex release that will run in the upstream
task containers. Verify `codex --version` inside a Linux container before
creating the protocol; do not substitute a host macOS binary. The development
wheel is written as `dist/lha-<version>-py3-none-any.whl`; resolve the exact
file instead of hard-coding a future release number.

```bash
curl -fL -o codex-x86_64-unknown-linux-musl.tar.gz \
  https://github.com/openai/codex/releases/download/rust-v0.141.0/codex-x86_64-unknown-linux-musl.tar.gz
tar -xzf codex-x86_64-unknown-linux-musl.tar.gz
docker run --rm --platform linux/amd64 \
  -v "$PWD/codex-x86_64-unknown-linux-musl:/tmp/codex:ro" \
  alexgshaw/password-recovery:20251031 /tmp/codex --version
# expected: codex-cli 0.141.0
```

The following script reads all 89 task definitions, resolves every configured
container tag to a registry digest, and writes the secret-free
preregistration. Image digests must be measured before the first smoke run; a
placeholder digest invalidates the protocol.

```python
import re
import subprocess
import tomllib
from pathlib import Path

from lha.bench.terminal_bench import (
    HARBOR_VERSION,
    create_protocol,
    write_protocol,
)

dataset_root = Path(
    ".terminal-bench-metadata/terminal-bench-2-1"
)
tasks = []
for task_file in sorted(dataset_root.glob("*/task.toml")):
    task = tomllib.loads(task_file.read_text())
    tasks.append((task["task"]["name"], task["environment"]["docker_image"]))

image_digests = {}
for instance_id, image in tasks:
    output = subprocess.check_output(
        ["docker", "buildx", "imagetools", "inspect", image],
        text=True,
    )
    match = re.search(r"^Digest:\s+(sha256:[0-9a-f]{64})$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"no immutable digest reported for {image}")
    image_digests[instance_id] = match.group(1)

wheel_path = next(Path("dist").glob("lha-*.whl"))
protocol = create_protocol(
    [instance_id for instance_id, _ in tasks],
    model="<pinned-model-id>",
    reasoning_effort="<fixed-effort>",
    harbor_version=HARBOR_VERSION,
    codex_cli_version="codex-cli 0.141.0",
    codex_target="x86_64-unknown-linux-musl",
    codex_binary_path="/absolute/path/to/codex-x86_64-unknown-linux-musl",
    task_image_digests=image_digests,
    wheel_path=wheel_path,
)
write_protocol(protocol, "terminal-bench-2.1-protocol.json")
print(*protocol.subset.smoke_instance_ids, sep="\n")
print(*protocol.subset.scored_instance_ids, sep="\n")
```

`build_harbor_commands()` produces exactly three smoke jobs or exactly 20
scored jobs. The credential path is checked by the builder but is deliberately
absent from the command line and Harbor job configuration; it is passed only
through the local process environment.

```bash
export LHA_CODEX_AUTH_FILE="/absolute/path/to/explicit/auth.json"
uv run python - <<'PY'
import os
import subprocess
from pathlib import Path

from lha.bench.terminal_bench import (
    TerminalBenchProtocol,
    build_harbor_commands,
)

root = Path.cwd()
wheel_path = next((root / "dist").glob("lha-*.whl"))
protocol_path = root / "terminal-bench-2.1-protocol.json"
protocol = TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
commands = build_harbor_commands(
    protocol,
    "smoke",  # change to "scored" only after all three smoke jobs are sound
    protocol_path=protocol_path,
    wheel_path=wheel_path,
    codex_binary_path="/absolute/path/to/codex-x86_64-unknown-linux-musl",
    auth_path=os.environ["LHA_CODEX_AUTH_FILE"],
    jobs_dir=root / "terminal-bench-jobs",
)
for command in commands:
    subprocess.run(command.argv, check=True, env=os.environ.copy())
PY
```

The public Harbor class is
`lha.bench.terminal_bench:LhaAgent`. Harbor 0.20 imports it directly from the
built wheel on Python 3.12. At installation time the adapter uploads the
preregistered standalone Codex binary, verifies its exact version, and does no
network package installation inside the task container.

The Harbor task container is the outer security boundary. Inside that
container the agent runs one tool-enabled `codex exec --ephemeral --json
--ignore-user-config --ignore-rules --sandbox danger-full-access` in the task's
current working directory. It can use the terminal and edit files required by
non-Python or multilingual tasks. LHA's ablation patch generator and its
internal test gate are not used; only Harbor's official verifier supplies the
task result.

An explicitly supplied `auth.json` is uploaded at task run time, copied to a
temporary `CODEX_HOME` with mode `0600`, and deleted after success, failure or
cancellation. Do not put this file in an image, repository, command argument,
artifact or public CI secret. This path is intended for a trusted local Harbor
runner.

After Harbor finishes, regenerate the same command objects and reverse-check
the actual `config.json` and `result.json` files:

```python
import os
from pathlib import Path

from lha.bench.terminal_bench import (
    TerminalBenchProtocol,
    build_harbor_commands,
    validate_harbor_results,
)

root = Path.cwd()
wheel_path = next((root / "dist").glob("lha-*.whl"))
protocol_path = root / "terminal-bench-2.1-protocol.json"
protocol = TerminalBenchProtocol.model_validate_json(protocol_path.read_text())
commands = build_harbor_commands(
    protocol,
    "smoke",
    protocol_path=protocol_path,
    wheel_path=wheel_path,
    codex_binary_path="/absolute/path/to/codex-x86_64-unknown-linux-musl",
    auth_path=os.environ["LHA_CODEX_AUTH_FILE"],
    jobs_dir=root / "terminal-bench-jobs",
)

manifest = validate_harbor_results(
    protocol,
    "smoke",
    commands,
    protocol_path=protocol_path,
    manifest_path="terminal-bench-smoke-manifest.json",
)
print(*manifest.observed_instance_ids, sep="\n")
```

Validation fails unless the result set is exactly the registered three or 20
tasks, with one trial per job and Harbor retries still at zero. The manifest
records the expected and observed IDs, their per-task image digest mapping,
the protocol digest and all job directories.

For a scored run, first call `validate_harbor_results(..., "scored", ...)` and
persist its `HarborExecutionManifest`. Then call
`derive_terminal_bench_records()` to derive all 20 rows from the still-matching
official `result.json` files, followed by `summarize_records()` with the same
protocol, commands, and manifest. The summary API revalidates those files; it
does not accept manually constructed task rows. Missing verifier output and
protocol failures remain `ERROR` in the denominator. The heading always says
“Terminal-Bench 2.1 固定 20 题子集”; it must not be presented as a complete
leaderboard result.

This contract reports only values that the official Harbor result can support:
PASS, FAIL, ERROR, success rate over all 20 submitted tasks, duration
percentiles, and protocol-error count. The current Harbor agent invokes one
`codex exec` directly; it does not run LHA's gate or repair loop. Consequently,
incorrect deliveries, interceptions, false rejections, and repair success are
stored as `null` and rendered as “未测” or “不适用”, never as zero.

Codex CLI 0.141 JSONL exposes token usage and auditable tool items, but not a
reliable count of internal model calls or an LHA repair count. This protocol
therefore does not claim a 60-model-call limit, three repairs, or any related
mechanism metric. A harness-backed Harbor evaluation would be a separate
implementation and must be completed before those metrics can be published.

## Paired statistics

`lha.bench.stats` carries the comparison tools for when runs exist: an exact
McNemar test on discordant pairs (two conditions on the same instances) and a
seeded task-cluster bootstrap for CIs. The ablation already reports with the
same cluster-bootstrap method.
