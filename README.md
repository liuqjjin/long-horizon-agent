# Long-Horizon Research Agent

A **verification-first** long-horizon agent harness for scientific software
engineering — code, papers, and experiments.

The spine of the project is the **verification loop**:

```
context → tool call → execute → verify → repair → checkpoint → repeat
```

with max-steps, checkpoint/resume, and a human-approval gate. Every agent emits a
**structured artifact** (a plan, a context bundle with provenance, a patch, a
verdict) — never free-form chat. A **live-context layer** sits underneath the
spine as infrastructure, reachable only through a small facade (`lha.live_context`)
so the rest of the system never depends on how context is indexed.

## Why "verification-first"

Two objective sources keep the agent honest:

| Family       | Verifiers                                  |
|--------------|--------------------------------------------|
| code         | `pytest`, `ruff` (type-check next)          |
| experiment   | `psnr`, `ssim`, `reproducibility`           |
| context      | `freshness`, `citation`                     |

PSNR is just one verifier among many — the core has no metric-specific assumptions.
The PSNR/SSIM verifiers **independently recompute** the metric from the saved
arrays rather than trusting the experiment's self-reported number, so a fabricated
metric is caught; `reproducibility` re-runs the experiment and checks the numbers
match plus that seed/versions/commit were recorded.

## Architecture

```
src/lha/
  harness/        the spine: loop, state, checkpoint, budget, approval
  live_context/   the facade (the ONLY door to the indexers)
  agents/         Supervisor · ContextEngineer · Implementer · Experimenter · VerifierAgent
  verifiers/      pluggable code / experiment / context families
  llm/            stub | claude_cli | anthropic, behind one interface
  runtime/        opt-in LangGraph durable runtime (SqliteSaver + interrupt)
  memory.py       skill memory (Voyager-lite): record verified successes
  orchestrator.py run many tasks in parallel via process-isolated workers
  eval.py         ResearchAgentBench-Lite (the harness measuring itself)
  tasks/ tools/   task specs; sandbox patch/shell helpers
  cli.py          lha run | resume | batch | eval | index | index-docs | ask | approve | reject
flows/            paper/experiment/skill index flows (behind the facade)
data/             sample repo (toy bug) + paper note + experiment log + experiment script
runs/<run_id>/    per-run artifacts: state.json, ledger.jsonl, plan, patch,
                  verify.json, pr_summary.md, graph.sqlite, workdir/
```

The facade exposes exactly: `search_code`, `search_papers`, `search_experiments`,
`search_skills`, `get_fresh_context`, `reject_stale`. A CI grep enforces that
nothing outside `live_context/` imports the underlying indexers.

## Quickstart

```bash
uv sync                     # install deps
# (one-time) install the code indexer used by search_code:
pipx install 'cocoindex-code[full]'

# Run the toy issue->PR task: finds the bug, fixes it, verifies with REAL pytest
uv run lha run data/tasks/fix_average.yaml

# Run the paper-to-experiment task: runs a real experiment, verifies PSNR/SSIM + reproducibility
uv run lha index-docs            # build paper/experiment context (CocoIndex flows)
uv run lha run data/tasks/run_sr_experiment.yaml

# Answer a query with fresh, cited context
uv run lha index data/sample_repo
uv run lha ask "how is average computed" --root data/sample_repo --kinds code

# Resume a paused run
uv run lha resume <run_id>

# Self-evaluate across all 5 workflows (ResearchAgentBench-Lite)
uv run lha eval                  # -> scorecard + runs/eval_report.json

# Run many tasks in parallel (orchestrator-worker, process-isolated)
uv run lha batch data/tasks/fix_average.yaml data/tasks/run_sr_experiment.yaml

# Durable runtime with an approval gate (LangGraph interrupt + resume)
uv run lha run --runtime langgraph data/tasks/fix_average_approval.yaml   # -> AWAITING_APPROVAL
uv run lha approve <run_id> && uv run lha resume --runtime langgraph <run_id>
```

The walking skeleton runs with a **deterministic stub** implementer — a REAL
`pytest` verifies a REAL fix with no API key and no network. Swap in a real LLM
with `--llm claude_cli` (uses the authenticated `claude` CLI) or `--llm anthropic`.

## Status

- **Week 1-A (done):** verified loop, code search via the facade, pytest/ruff
  verifiers, checkpoint/resume, freshness + stale-context rejection, the toy
  issue→PR task, and a hermetic test suite (`uv run pytest`).
- **Week 1-B (done):** paper/experiment context flows behind the facade.
- **Week 2-3 (done):** `psnr`/`ssim`/`reproducibility` verifiers + the
  paper-to-experiment workflow (run a real experiment → recompute & verify
  metrics → check reproducibility → cite the paper/baseline).
- **Week 4 (done):** multi-agent orchestration (parallel verifiers + a
  process-isolated batch orchestrator), **ResearchAgentBench-Lite** self-eval
  (5/5: issue-to-PR · paper-to-experiment · resume · freshness ·
  verification-ablation), **skill memory** (record verified successes → retrieve
  for future tasks), and an opt-in **LangGraph durable runtime** (SqliteSaver
  checkpoint/resume + `interrupt()` approval gate).
