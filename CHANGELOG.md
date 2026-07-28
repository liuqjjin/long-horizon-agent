# Changelog

Notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `ResolvedPatch` and `PatchTransaction`. The write set now comes from the
  actual patch, and every attempt records `PREPARED`, `APPLIED`, `VERIFIED`, or
  `REVERTED` with manifests and redundant backups.
- Added run-state schema v2 with stable attempt IDs, saved budgets, cumulative
  model usage, run locks, and ledger idempotency keys. Schema-v1 runs remain
  readable but cannot be resumed as schema v2.
- Added five fixed 10-stage repository fixtures covering configuration parsing,
  SQLite migration, concurrency, CLI contracts, and experiment reproduction.
- Added typed repository stages with intent and completion records. Recovery
  does not repeat a stage that may already have produced a side effect.
- Added validated run inspection, self-contained HTML traces, and dry-run-first
  pruning through `lha runs`.
- Added a Terminal-Bench 2.1 adapter for Harbor 0.20.
- Fixed the 20-task evaluation protocol before the scored run. The committed
  schema-v4 package contains its manifests, source and wheel attestation,
  summary, and public trial evidence.
- Added CI checks that rebuild the exact wheel used by Terminal-Bench from its
  recorded Git commit and reject a release that no longer contains that commit.

### Changed

- Codex CLI calls now use temporary homes, workspaces, credential copies, a
  restricted environment, and a separate process group. Malformed or incomplete
  JSONL and unfinished or disallowed tool use fail the call.
- Experiment records now bind arrays by path, shape, data type, digest, and
  input digest. Reproduction runs in a new directory and rejects missing,
  stale, non-finite, or mismatched data.
- Context records now distinguish no result, unavailable backend, failed index,
  stale source, and partial availability.
- Horizon reports now separate paired cells, complete-corpus repetitions, and
  descriptive composition. Cell and episode tests may differ; composition adds
  no samples.
- Boundary proportions use Wilson score intervals. Interior rates continue to
  use a task-cluster bootstrap.
- Packaged context flows moved under `src/lha/live_context/flows/`. Release
  checks install both wheel and source distribution outside the checkout.
- Public documentation was shortened and reorganized around commands,
  implementation, evaluation status, and known limits.
- The application image now bundles a pinned `all-MiniLM-L6-v2` snapshot and
  loads it offline, so container self-eval does not download model files.
- Host, Codex, and Claude subprocesses now bound output, reject invalid timeout
  values, remove process groups on every exit path, and fail when cleanup cannot
  be confirmed.

### Evaluation record

- The schema-v2 ablation files are retained as records of the earlier protocol.
  The scoring boundary, error classification, and evidence format have since
  changed, so those files are not the current project result. New ablation
  numbers require a complete schema-v4 rerun.
- The preregistered Terminal-Bench 2.1 fixed 20-task subset produced 7 `PASS`,
  9 `FAIL`, and 4 `ERROR`; all errors remain in the denominator. This is a
  fixed-subset result, not a full-dataset or leaderboard score.
- No SWE-bench score is published.

## [0.4.1] — 2026-07-25

### Fixed

- Made the repository self-eval independent of whether `ccc` is installed.
  Code-fix and resume cases now declare retrieval optional because their oracle
  is Pytest. A separate case forces the context backend unavailable and passes
  only when the run fails for that reason.
- Added the CI badge and the fail-closed context case.

## [0.4.0] — 2026-07-25

### Added

- Added `lha horizon` for cell, complete-repetition, and composition analysis.
  The first generated report incorrectly required cell and episode McNemar
  values to match; the current report removes that assumption.
- Added `trusted-local` and Docker execution backends for target- or
  model-influenced commands.
- Added a separate ablation scoring path. It applies frozen source changes to new
  canonical repositories and runs original tests through a separate backend.
- Expanded the internal defect corpus from 11 to 17 tasks.
- Added SWE-bench Verified and Terminal-Bench adapters with contract tests.
  This release did not publish public benchmark results.
- Added checksummed checkpoints, append-only ledger validation, crash-injection
  tests, and persistent model-call accounting.
- Bound approvals to the saved patch digest and protected tests, build files,
  and CI configuration from unapproved edits.

### Changed

- Required context now fails when the backend or index is unavailable.
- Repair attempts reload code from the current run workspace.
- Reproducibility checks require an input digest and, in Git repositories, a
  commit identifier.
- Context dependencies moved to the optional `context` extra.
- Ablation results were regenerated with the separate scoring path. These
  schema-v2 results are now retained as historical protocol records.
- Renamed the old self-eval label to `self-eval`.
- Ablation reports now use a plain table and retain explicit `ERROR` cells.

### Fixed

- `Verdict.from_checks` now rejects an empty list.
- Removed a committed local coverage file and corrected stale documentation.

## [0.3.0] — 2026-06-28

### Added

- Added `lha ablate` with paired `trust`, `gate`, and `verify` conditions.
- Added 11 fixed Python defect repositories with Pytest oracles.
- Added whole-file model patches while retaining a display diff.
- Added compact Pytest failure output for repair prompts.

### Fixed

- Cleared stale Python bytecode before verification.
- Rejected absolute paths, parent traversal, directories, and binary targets in
  whole-file patch parsing.
- Added Docker image builds to CI.

## [0.2.0] — 2026-06-22

### Added

- Added optional model-generated planning with typed validation and a
  deterministic fallback.
- Added per-step artifacts under `runs/<id>/steps/`.

### Changed

- Applied the saved deadline across LangGraph resume.
- Reused the exact approved patch after resume instead of generating it again.

### Security

- Restricted step IDs used in filesystem paths to a safe single segment.

## [0.1.0] — 2026-06-22

### Added

- Added the state-machine loop, checkpoint/resume, approval, and bounded repair.
- Added code, experiment, and context verifier families.
- Added the indexed-context facade and isolated CocoIndex and `ccc` behind it.
- Added typed planning and execution components, batch execution, and an
  optional LangGraph runtime.
- Added run tracing, container packaging, type checking, CI, and project
  governance files.

### Fixed

- Made Ruff configuration errors, timeouts, and invalid output fail the check.
- Reverted rejected or unverified patches in both runtimes.
- Persisted step and deadline budgets across resume.
- Rejected truncated and refused Anthropic responses.
- Rejected empty verifier sets.
- Reverted a mid-step failure before saving terminal state.
- Made batch and self-eval isolate per-task failures.
- Piped Claude CLI prompts through standard input to avoid argument limits.
