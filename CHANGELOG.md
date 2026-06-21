# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
