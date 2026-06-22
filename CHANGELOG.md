# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — verification-first correctness

- **ruff verifier gated on exit code.** A ruff invocation that *failed to run* (config
  error, timeout, non-JSON output) silently passed; it now fails, upholding the
  "a check that can't run must fail" rule (`pytest` already enforced it).
- **LangGraph runtime reverts unverified/rejected patches**, matching the default loop —
  a change that fails verification (or is rejected) no longer survives in the sandbox.
- **Budget bounds the whole run across pause/resume.** `max_steps`/`deadline_s` were
  per-process; cumulative `steps_used`/`elapsed_s` are now persisted and re-seeded on
  resume (check-before-increment, so a pause is exact). `LHA_DEADLINE_S` now reaches the
  loop and rejects non-finite/negative values.
- **Anthropic backend fails closed** on `stop_reason` `max_tokens`/`refusal` instead of
  returning a silently truncated/empty completion; default `max_tokens` 16000 (was a
  truncating 4096) with adaptive thinking + high effort for `claude-opus-4-8`.
- **A step that verified nothing no longer passes** (empty verifier list fails closed).
- **Citations are checked on any provenance-bearing artifact** (e.g. experiments), not
  only patches.
- **Human rejection reverts the change**, and stale approval decisions can no longer be
  misattributed to a later step (decisions carry a `step_id`).

### Fixed — robustness

- The loop **fails closed on an unexpected mid-step fault** (reverts, ledgers, checkpoints)
  instead of wedging at `RUNNING` with a half-applied sandbox.
- The orchestrator **survives a per-task spawn failure** (broadened beyond `TimeoutExpired`)
  so one bad worker can't discard the whole batch; `lha eval` **isolates each case** so one
  crash doesn't zero the report.
- The skill-memory note is **re-indexed after recording**, closing the retrieval loop for
  `issue_to_pr` runs.
- The `claude_cli` backend **pipes the prompt via stdin** (avoids `E2BIG` on large repos).

### Added — observability & deployability

- `lha trace <run_id>` renders a run's ledger timeline; `-v/--verbose` enables logging.
- **Containerization:** multi-stage `Dockerfile` (uv, non-root, layer-cached, HF cache),
  `.dockerignore`, and `docs/DEPLOY.md`.
- Packaging: `py.typed` marker, single-sourced `__version__`, MIT `LICENSE`, and PEP 639
  license/keyword/classifier metadata.
- Pinning tests for all of the above (`tests/test_hardening.py`, `tests/test_deployability.py`).

---

Documentation & engineering-rigor pass (no behavior change unless noted):

### Added
- Top-tier README (differentiator tagline, error-compounding hook, mermaid spine
  diagram, verbatim `lha eval` table) and a `docs/` set: `ARCHITECTURE`, `BENCHMARKS`,
  `VERIFICATION_FIRST` (the thesis), and `demo` (GIF recording script).
- GitHub Actions CI mirroring the local gate (ruff, pyright, facade-isolation grep,
  pytest, `lha eval`) with embedding-model caching.
- `pyright` type-checking of `src/lha` (0 errors) as part of the gate and CI.
- Governance: `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, and issue/PR templates.

### Changed
- Type-only hardening to reach a clean `pyright` run: explicit `None`-guards, Literal
  narrowing, and a few targeted casts for third-party stub gaps. No logic changes;
  `lha eval` remains 5/5.

## 0.1.0 — draft (not yet tagged)

The initial feature-complete harness. *Draft notes for the first release; the tag
is a human decision (see `OVERNIGHT_LOG.md`).*

### Added
- **Verification loop** (the spine): `context → execute → verify → repair →
  checkpoint → repeat`, with max-steps, durable checkpoint/resume, and a
  human-approval gate. State is an atomic `state.json` plus an append-only
  `ledger.jsonl`.
- **Three verifier families behind one interface:** code (`pytest`, `ruff`),
  experiment (`psnr`, `ssim`, `reproducibility` — metrics recomputed from output),
  and context (`freshness`, `citation`). A check that can't run *fails*.
- **Live-context facade** (`search_code` / `search_papers` / `search_experiments` /
  `search_skills` / `get_fresh_context` / `reject_stale`) with CocoIndex and the
  `ccc` code indexer fully isolated behind it (enforced by a grep).
- **Agents** emitting structured artifacts (Supervisor, Context Engineer, Implementer,
  Experimenter, Verifier) and a process-isolated batch **orchestrator** (`lha batch`).
- **Opt-in LangGraph durable runtime** (`--runtime langgraph`): a `StateGraph`
  checkpointed by `SqliteSaver` with `interrupt()`-based human approval.
- **Skill memory** (Voyager-lite): verified successes recorded and retrieved as
  context for future tasks.
- **ResearchAgentBench-Lite** (`lha eval`): self-evaluation across issue-to-PR,
  paper-to-experiment, resume, freshness, and verification-ablation (currently 5/5).
- **CLI:** `lha run|resume|batch|eval|index|index-docs|ask|approve|reject`.
