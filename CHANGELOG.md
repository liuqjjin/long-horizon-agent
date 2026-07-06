# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Reworded documentation and code comments for a plainer, more consistent voice, and
  renamed the self-eval suite (previously "ResearchAgentBench-Lite") to "self-eval".
  No behavior change.
- `lha ablate` report output is now a plain table plus a one-line factual summary,
  without the editorial header line or emoji status legend.

### Fixed
- Untracked `.coverage`, a local artifact that had been committed.

## [0.3.0] — 2026-06-28

### Added
- **Verification ablation** (`lha ablate`): measures the project's central claim
  against a real LLM. A paired design draws one first attempt per task and scores the
  same attempt under `trust` (apply and accept), `gate` (apply, run the test gate,
  refuse on failure), and `verify` (gate plus repair loop), reporting claimed vs true
  vs false success. It is leak-free (single-shot implementer with file tools denied,
  shown only non-test source), tamper-proof (patches may edit source only; the test
  oracle and config stay canonical), and excludes transient backend errors (retried,
  then recorded as ERROR, never counted). A weaker implementer `--model` calibrates
  difficulty. See [docs/ABLATION.md](docs/ABLATION.md).
- **Bug-fix benchmark corpus**: 11 small, self-contained Python repos
  (`data/bench/*`, `data/tasks/bench_*.yaml`), each a planted bug with a pytest oracle,
  spanning arithmetic/strings/recursion/stack/search/parsing. Each was screened so the
  bug is real, the oracle catches a naive fix, and the issue stays symptom-level.

### Changed
- **Whole-file rewrites in the LLM implementer.** Real backends now return the full
  corrected file (`file_contents`, a direct write) rather than a unified diff, which
  `git apply` often rejects on minor context drift even when the fix is correct. A
  display diff is still recorded for the artifact.
- The **pytest verifier surfaces the failing assertion messages** (compact, ACI-style)
  so the repair loop fixes the real defect instead of guessing from "tests failed".
- The **implementer no longer dumps test files into the prompt** — a fix is reasoned
  from the issue and structured failure feedback, not transcribed from the oracle.
- `ClaudeCLIClient` gained `--model` and a `no_tools` (single-shot, tools-denied) mode.

### Fixed
- The **pytest verifier clears stale `__pycache__`** before running: a repair that
  rewrites a file to the same byte-size within the same mtime-second could otherwise
  be graded against cached bytecode, so the repair loop could never converge.
- **Path-traversal hardening** in the whole-file parser: a `### ../escape` header (and
  absolute paths) are refused, and a header naming a directory/binary file is skipped
  rather than crashing the step.

### CI
- The CI now **builds the Docker image** (`docker` job), so the Docker build is checked
  on every change.

## [0.2.0] — 2026-06-22

### Added
- **Dynamic LLM planning** (opt-in: `Config.dynamic_planning` / `LHA_DYNAMIC_PLANNING`,
  default off). A real backend can decompose a task into a verifiable plan; the
  candidate is validated (registered verifiers, path-safe step ids) and falls back to
  the deterministic template on any failure, so the stub/`lha eval` path stays
  reproducible.
- **Per-step artifacts** under `runs/<id>/steps/<step_id>/` (verify/patch/experiment/
  context), so a multi-step plan keeps every step's provenance; the PR/experiment
  finalizers report the step that produced the result rather than whichever file wrote
  last.

### Changed
- The **LangGraph runtime now enforces `deadline_s`** in addition to `max_steps`.
- The **default-loop approval gate reuses the exact patch the human approved** on resume
  rather than regenerating it (artifact-binding; see SECURITY.md for the LangGraph caveat).

### Security
- Step ids used as filesystem paths (per-step artifacts, backups) are sanitized to a
  single safe segment, and dynamic plans carrying an unsafe step id are rejected — a
  defense-in-depth guard now that plans can be model-generated.

## [0.1.0] — 2026-06-22

### Fixed — verification correctness

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
- README (tagline, error-compounding intro, mermaid diagram, `lha eval` table) and a
  `docs/` set: `ARCHITECTURE`, `BENCHMARKS`, `VERIFICATION_FIRST` (why verification
  comes first), and `demo` (GIF recording script).
- GitHub Actions CI mirroring the local gate (ruff, pyright, facade-isolation grep,
  pytest, `lha eval`) with embedding-model caching.
- `pyright` type-checking of `src/lha` (0 errors) as part of the gate and CI.
- Governance: `CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`, and issue/PR templates.

### Changed
- Type-only hardening to reach a clean `pyright` run: explicit `None`-guards, Literal
  narrowing, and a few targeted casts for third-party stub gaps. No logic changes;
  `lha eval` remains 5/5.

### The initial feature-complete harness

### Added
- **Verification loop:** `context → execute → verify → repair →
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
- **Skill memory:** verified successes recorded and retrieved as
  context for future tasks.
- **Self-eval** (`lha eval`): self-evaluation across issue-to-PR,
  paper-to-experiment, resume, freshness, and verification-ablation (currently 5/5).
- **CLI:** `lha run|resume|batch|eval|trace|index|index-docs|ask|approve|reject`.
