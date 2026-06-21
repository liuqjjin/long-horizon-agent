# Quickstart

From a clean clone to a verified agent run in a few minutes. Requires
**Python 3.11+** and [`uv`](https://docs.astral.sh/uv/). Run everything from the
repo root.

## 1. Install

```bash
uv sync
```

## 2. Run the toy issue→PR task

The harness finds a planted off-by-one bug, fixes it, and verifies the fix with a
**real `pytest`** run (the default implementer is a deterministic stub, so this
needs no API key and no network):

```bash
uv run lha run data/tasks/fix_average.yaml
```

Expected (abridged):

```
status : DONE
   - pytest: passed=True (2 passed, 0 failed, 0 error)
   - ruff: passed=True (0 violations)
pr     : runs/<run_id>/pr_summary.md
```

Look in `runs/<run_id>/` for the full trail: `plan.json`, `context_bundle.json`
(with provenance), `patch.diff`, `verify.json`, `ledger.jsonl`, and `pr_summary.md`.

## 3. Self-evaluate

```bash
uv run lha eval        # first run downloads a small embedding model (one-time)
```

Expected:

```
# ResearchAgentBench-Lite — 5/5
...
score: 5/5
```

See [BENCHMARKS.md](BENCHMARKS.md) for what each of the five tasks verifies.

## 4. Ask a question with fresh, cited context

```bash
uv run lha index data/sample_repo                                   # build the code index (needs `ccc`)
uv run lha ask "how is average computed" --root data/sample_repo --kinds code
```

You'll get hits with a `[locator]` and a similarity score — provenance for every
answer. (Code search uses `cocoindex-code`; install it with
`pipx install 'cocoindex-code[full]'`. Steps 2–3 work without it.)

## 5. Durable run with a human-approval gate

```bash
uv run lha run --runtime langgraph data/tasks/fix_average_approval.yaml   # -> AWAITING_APPROVAL
uv run lha approve <run_id>
uv run lha resume --runtime langgraph <run_id>                            # -> DONE
```

The run is checkpointed to `runs/<run_id>/graph.sqlite` between the two commands
(LangGraph `interrupt()` + `Command(resume=...)`), so approval survives a restart.

## Next

- [VERIFICATION_FIRST.md](VERIFICATION_FIRST.md) — why the loop is built around an
  objective oracle.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the spine, the facade, the verifier families.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — dev setup and the full gate.
