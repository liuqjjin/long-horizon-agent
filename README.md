# Long-Horizon Research Agent

**A verification-first agent harness — and a method for measuring how good its own
verifier actually is.**

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![lint: ruff](https://img.shields.io/badge/lint-ruff-261230)
![tests](https://img.shields.io/badge/tests-pytest-0a9edc)

Long-horizon agents fail because errors compound: if each step succeeds with
probability `p`, an `n`-step task succeeds with about `pⁿ`. Asking the model to check
its own work raises `p` only a little — there is no external signal. This harness
gates every step on an objective oracle (a real test run, a metric recomputed from
the output, a reproducibility re-run) and **a step that cannot be verified fails**.

The unusual part is the measurement. The internal gate only *predicts*; truth comes
from an independent scorer that re-applies the frozen diff to a pristine repository on
a separate execution backend. That decoupling is what makes the gate's own **false-negative
rate** observable — the correct fixes a gate wrongly throws away, which a gate-graded
design cannot see by construction.

Honest headline, stated before the good part: at single-step difficulty the effect is
small and **not statistically significant** (2 of 51 paired cells, exact McNemar
p = 0.50). Carried onto the horizon the thesis is actually about, the same per-step
effect compounds to **44% → 100%** over 17 steps — a bigger effect, on the same
evidence, and [the report says so itself](docs/HORIZON.md).

---

**验证优先的 Agent 执行框架 —— 以及一套量化「验证器自身有多差」的方法。**

长程 agent 失败于误差复利：单步成功率为 `p` 时，`n` 步任务约为 `pⁿ`。让模型自查只能
把 `p` 抬高一点点——没有外部信号。本框架把每一步都压在客观 oracle 上（真实测试运行、
从输出重算的指标、可复现性重跑），**无法验证的步骤判失败，而不是默认通过**。

不常见的部分是测量方法。内部门禁只给出*预测*；真值来自独立评分器——把冻结的 diff
重新施加到全新仓库副本、恢复原始测试、用另一套执行后端打分。正是这个解耦让门禁
**自身的假阴性率**变得可观测，即被门禁误杀的正确修复；而用门禁自己的判决当真值的
设计，在构造上永远看不到这个量。

先说不利的一半：单步难度下这个效应很小且**不显著**（51 个配对单元中 2 个，精确
McNemar p = 0.50）。把同一个效应放到论点真正针对的多步 horizon 上，17 步端到端
成功率是 **44% → 100%**——效应更大，证据不变，[报告本身会指出这一点](docs/HORIZON.md)。

---

## Measured effect of the gate

[`lha ablate`](docs/ABLATION.md) runs a real LLM through the same harness under three
conditions and scores the identical first attempt under each, so the conditions differ
only in the gate:

Implementer `claude_cli` on `claude-haiku-4-5-20251001`, 17 bug-fix tasks × 3 reps
(51 paired cells per condition, 0 transient errors), graded by an independent final
scorer, not by the gate itself:

| condition | claimed | true success (95% CI) | false success (95% CI) |
|---|---|---|---|
| `trust` — apply the fix, no gate | 100% | 96% (90–100%) | 4% (0–10%) |
| `gate` — run the test gate, refuse on failure | 96% | 96% (90–100%) | 0% |
| `verify` — gate plus repair loop | 100% | 100% | 0% |

Without the gate, 2 of 51 accepted fixes are wrong and ship silently. `trust` and
`gate` score the same attempts, so the difference is exactly what the gate catches: it
refused those same 2 cells and discarded no correct fix (TN=2, FP=0, FN=0 against the
independent scorer). The repair loop then fixed both refusals. The effect is real but
small at this corpus difficulty — haiku-4.5's first attempt is already right 96% of the
time here; the numbers say what the gate buys on top of that, no more. The experiment is
paired and leak-free (the implementer never sees the tests, a patch cannot touch the
oracle); method, statistics, and limits are in [docs/ABLATION.md](docs/ABLATION.md),
raw data in [benchmarks/ablation_report.json](benchmarks/ablation_report.json).

## What that compounds to

A 4% per-step error rate is easy to dismiss, so `lha horizon` carries it onto the axis
the thesis argues about. An *episode* is `k` independent subtasks; it is correct
through step `k` only if every one of steps `1..k` truly succeeded:

| k | `trust-chain` (no gate) | `verify-chain` (gate + repair) | gap |
|---:|---|---|---:|
| 1 | 96% (90–100%) | 100% | +3.9 pp |
| 8 | 71% (42–100%) | 100% | +29.1 pp |
| 17 | **44% (13–100%)** | **100%** | **+55.6 pp** |

The curve is the compounding model evaluated at the measured per-task `p`, computed
exactly rather than sampled. It changes the effect *size*, not the *confidence*:
composing measured cells into more orderings cannot create information, so the paired
test at 17 steps returns the same `p = 0.50` as the single-step test on the same
cells. Only more repetitions change that — the number needed (and a registered
prediction, fixed before running them) is in [docs/HORIZON.md](docs/HORIZON.md).

## The loop

```mermaid
flowchart LR
    T[Task] --> S[Supervisor<br/>plan]
    subgraph LOOP["verification loop  (max-steps · checkpoint · resume · approval gate)"]
        direction LR
        C[Context<br/>Engineer] --> X[Implementer /<br/>Experimenter]
        X --> V{Verify}
        V -- pass --> N[next step]
        V -- fail --> R[repair]
        R --> C
    end
    S --> C
    N --> CP[(checkpoint)]

    C -. only via facade .-> F[[live_context facade]]
    F --> CODE[(code: ccc / MCP)]
    F --> DOCS[(papers · experiments · skills:<br/>CocoIndex flows)]

    V --> VF[code: pytest · ruff]
    V --> VE[experiment: PSNR · SSIM · repro]
    V --> VC[context: freshness · citation]
```

The rest of the system reaches indexed context only through the `live_context`
facade (`search_code` / `search_papers` / `search_experiments` / `search_skills` /
`get_fresh_context` / `reject_stale`); CocoIndex and the code indexer live entirely
behind it, and a `grep` check keeps them there.

## Quickstart

```bash
uv sync                                        # install (Python 3.11+)
uv run lha run data/tasks/fix_average.yaml     # find a bug, fix it, verify with real pytest
uv run lha eval                                # run the self-eval across all five workflows
```

Run from the repo root. `uv sync && uv run lha eval` reproduces `5/5` from a clean
checkout (generated indexes and runs are gitignored and rebuilt on demand); the first
`lha eval` downloads a small embedding model (tens of MB, one-time) for the
paper/experiment/freshness cases.

The default implementer is a deterministic stub, so a real `pytest` verifies a real
fix with no API key and no network. Swap in an LLM with `--llm claude_cli` (the
authenticated `claude` CLI) or `--llm anthropic`.

## Self-eval

`lha eval` is the harness checking itself: five workflows, each with an objective
pass/fail. Output from this repo:

```
# Self-eval — 5/5

| dimension              | case                    | result | detail |
|------------------------|-------------------------|--------|--------|
| issue-to-PR            | fix_average             | PASS   | status=DONE verified=True |
| resume                 | pause_resume            | PASS   | first=PAUSED resumed=DONE |
| freshness              | edit_reindex            | PASS   | initial_fresh=True stale_after_edit=True fresh_after_reject=True |
| paper-to-experiment    | bicubic_sr              | PASS   | status=DONE verified=True |
| verification-ablation  | strict_threshold_caught | PASS   | status=FAILED psnr_correctly_rejected=True reached_psnr_step=True |

score: 5/5
```

Reproduce with `uv run lha eval`. In the verification-ablation case, the bicubic
baseline reaches PSNR ≈ 25.07 dB / SSIM ≈ 0.8246 (recomputed by the verifier with
`data_range=1.0`), so when the task asks for a 40 dB bar the harness reports `FAILED`
instead of claiming a result it cannot verify.

## How it works

- **The loop.** `context → execute → verify → repair → checkpoint → repeat`, with
  max-steps, durable checkpoint/resume, and a human-approval gate. State is a JSON
  checkpoint plus an append-only `ledger.jsonl`.
- **Structured artifacts.** Each role emits a typed artifact: Supervisor →
  `Plan`, Context Engineer → `ContextBundle` (with provenance), Implementer → `Patch`,
  Verifier → `Verdict`.
- **Three verifier families behind one interface.** `code` (`pytest`, `ruff`),
  `experiment` (`psnr`, `ssim`, `reproducibility`), `context` (`freshness`,
  `citation`). The core has no metric-specific assumptions, and a verifier that
  cannot run its check fails.
- **Live context, isolated.** Code search goes through `ccc` (cocoindex-code) over
  its MCP tool; paper/experiment/skill context comes from CocoIndex flows. All of it
  sits behind the `live_context` facade.
- **Durable runtime (opt-in).** `lha run --runtime langgraph` drives the same plan
  through a LangGraph `StateGraph` checkpointed by `SqliteSaver`, using `interrupt()`
  for the approval gate.
- **Skill memory.** A verified run is distilled to a retrievable note and surfaced as
  context for later tasks. Only verified successes are recorded.

## Design notes

Most agent frameworks orchestrate tool calls and, if they evaluate at all, use an LLM
as judge. This harness inverts that: the loop is only as reliable as its verifier, so
the verifier is an objective oracle wherever one exists — a test suite, an image
metric, a freshness check — and the harness fails loudly when it cannot verify. The
core stays small and readable, with the live-context machinery quarantined behind a
facade, though the embedding/index stack pulls a GB-scale dependency set (torch,
cocoindex) on the first `uv sync`.

## Documentation

- [docs/QUICKSTART.md](docs/QUICKSTART.md) — clone to a verified run in a few minutes.
- [docs/VERIFICATION_FIRST.md](docs/VERIFICATION_FIRST.md) — why error compounding
  makes an objective oracle the highest-leverage place to intervene.
- [docs/ABLATION.md](docs/ABLATION.md) — the measured effect: a real LLM with
  verification off vs. on, and what the gate and the repair loop each buy.
- [docs/HORIZON.md](docs/HORIZON.md) — what that per-step effect compounds to over a
  horizon, a registered prediction, and why composition changes the effect size but
  not the confidence.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the loop, the facade, the verifier
  families, the durable runtime.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — the self-eval, what each case verifies,
  and how to reproduce `5/5`.

## CLI

```
lha run <task.yaml> [--runtime loop|langgraph] [--llm stub|claude_cli|anthropic]
lha resume <run_id>             lha approve|reject <run_id>    lha trace <run_id>
lha eval [--quick]              lha batch <task.yaml> ...      # parallel, process-isolated
lha ablate [task.yaml ...]      # verification ablation: trust vs gate vs verify (real LLM)
lha horizon                     # error compounding: the per-step effect across n steps
lha index <path>                lha index-docs                 lha ask <query> --kinds code,paper,...
```

`-v`/`--verbose` raises log verbosity; `lha trace <run_id>` renders a run's ledger
timeline. To run in a container, see [docs/DEPLOY.md](docs/DEPLOY.md).

## Configuration (`LHA_*` environment variables)

All configuration is environment variables read once at startup (`src/lha/config.py`):

| variable | default | meaning |
|---|---|---|
| `LHA_LLM_BACKEND` | `stub` | `stub` \| `claude_cli` \| `anthropic` (`--llm` overrides) |
| `LHA_CLAUDE_CLI` / `LHA_CLAUDE_MODEL` | `claude` / – | claude CLI path; pin a full model snapshot for reproducible runs |
| `LHA_ANTHROPIC_MODEL_IMPL` / `_ORCH` | opus-4-8 / sonnet-4-6 | Anthropic SDK models |
| `LHA_MAX_STEPS` / `LHA_MAX_REPAIRS` | 20 / 3 | loop budgets |
| `LHA_DEADLINE_S` / `LHA_MAX_LLM_CALLS` | unlimited | wall-clock / LLM-call budgets (0 or unset = unlimited; the run pauses, resumable) |
| `LHA_EXEC_BACKEND` / `LHA_EXEC_IMAGE` | `trusted-local` / `python:3.12-slim` | where target code executes (see [SECURITY.md](SECURITY.md)) |
| `LHA_CODE_BACKEND` | `auto` | code search: `ccc` \| `null` \| `auto` |
| context / misc | — | `LHA_FRESHNESS_MAX_AGE_S` (3600), `LHA_EMBEDDER_MODEL`, `LHA_SKILL_MEMORY` (1), `LHA_PARALLEL_VERIFY` (1), `LHA_DYNAMIC_PLANNING` (0) |
| `LHA_RUNS_DIR` / `LHA_DATA_DIR` | `runs` / `data` | state locations |

## Requirements

- Python 3.11+, [`uv`](https://docs.astral.sh/uv/).
- `uv sync` installs the harness and its dependencies.
- Code search uses [`cocoindex-code`](https://github.com/cocoindex-io/cocoindex-code)
  (`pipx install 'cocoindex-code[full]'`, then `claude mcp add cocoindex-code -- ccc mcp`).
  The harness talks to its MCP `search` tool directly; the bundled demos run without it.

## Status

Everything from the walking skeleton to the durable runtime is implemented and
exercised by `uv run pytest` and `uv run lha eval`. This is a research and portfolio project, not a
production service. Run `lha eval` and the bundled demos from a source checkout — the
wheel ships the harness, while the benchmark fixtures under `data/` live in the repo.
