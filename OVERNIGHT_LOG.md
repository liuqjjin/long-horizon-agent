# Overnight optimization log

Autonomous maintainer working `tools/overnight/BACKLOG.md` top-down. One small,
reversible increment per entry; all gates green before every commit.

**Gate** = `uv run pytest` · `ruff check .` · `uv run pyright src/lha` · facade-isolation grep · `lha eval` (5/5).
Eval policy: run `lha eval` for changes that can affect it (code/flows/verifiers/eval/data/tasks/config)
and at session start/end; pure docs/CI/governance commits run the fast gates (pytest/ruff/grep)
since eval cannot change when no runtime code changed. Last full eval: 5/5 (session start).

Note: `lha eval` can show a transient 4/5 on a heavily-churned `ccc` daemon (many prior runs);
a fresh daemon / clean checkout is reliably 5/5. Tracked under Remaining (harden eval robustness).

---

## Completed
- (baseline) git baseline; gates green: ruff clean, pytest 33, facade clean, `lha eval` 5/5.
  gitignore hygiene (`.DS_Store`/`.claude/`/`.mcp.json`).
- 0.1 README rewrite — differentiator tagline, compounding-error hook, mermaid spine diagram,
  30s quickstart, verbatim `lha eval` 5/5 table, how-it-works, why-different. Substantiable badges only.
- 0.2 docs/demo.md — exact asciinema/agg recording script for the approval→resume demo + eval;
  human records the GIF (headless: no TTY recorder).
- 1.1 docs/BENCHMARKS.md — ResearchAgentBench-Lite: verbatim 5/5 table, per-task objective oracle,
  verification-ablation centerpiece (with/without-verifier contrast on real numbers), reproduce cmd.
- 1.2 docs/ARCHITECTURE.md — spine, facade+isolation grep, 3 verifier families, durable runtime,
  agent team, skill memory, module map, two mermaid diagrams.
- 1.3 docs/VERIFICATION_FIRST.md — thesis: error-compounding math, RLVR link, honest limits, refs.
- 0.1(h) README Documentation section linking the four docs (all targets exist).
- 1.4 clean-checkout reproducibility — verified `uv sync && lha eval` → 5/5 after deleting all
  gitignored generated state (runs/, data/.lha_index/, data/skills/) on a fresh daemon; documented.
- 2.1 GitHub Actions CI (.github/workflows/ci.yml) — mirrors the local gate exactly (ruff,
  facade-isolation grep that fails on a leak, pytest, `lha eval`) with HF model caching. YAML +
  grep logic validated locally; cannot run Actions headless (see Needs human decision for the badge).
- 2.2 type-check (pyright) — drove `pyright src/lha` to 0 errors (None-guards, Literal narrowing,
  targeted casts/ignores for third-party stub gaps; no logic change); added [tool.pyright] config +
  a CI step; pyright is now part of the gate.
- 2.5 governance/community files — CONTRIBUTING (verification-first rule + the full gate),
  SECURITY, CODE_OF_CONDUCT (Contributor Covenant 2.1), issue templates, PR template (gate checklist).

## Remaining (from BACKLOG, top-down)
- Tier 0: 0.3 repo metadata (morning report)
- Tier 2: 2.3 coverage · 2.4 pre-commit · 2.6 CHANGELOG
- Tier 3: 3.1 examples · 3.2 QUICKSTART · 3.3 CLI polish · 3.4 demo task
- Tier 4: 4.1 failure memory · 4.2 eval corpus · 4.3 observability
- Cross-cutting: harden `lha eval` against daemon-state flakiness (reliability of the 5/5 claim)

## Needs human decision
- LICENSE: the repo has no LICENSE file. Without one it is "all rights reserved" and not
  reusable. Recommend **MIT** (simplest, permissive) or **Apache-2.0** (permissive + explicit
  patent grant). Add `LICENSE` and set `[project].license` in pyproject; then a license badge can
  be added. Not chosen automatically (licensing is the author's call).
- CODE_OF_CONDUCT contact: enforcement currently points to "repository contact channels / private
  security advisory". Add a concrete maintainer contact if desired.
- CI badge: after pushing to GitHub and confirming the first CI run is green, add
  `![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)` to the README.
  (Headless agent has no remote/owner slug and cannot run Actions, so no badge added yet —
  a wrong/red badge would be worse than none.) Also watch for HuggingFace anonymous
  rate-limiting on the model download in CI; set an `HF_TOKEN` repo secret if it flakes.

## Iterations
- 2026-06-21 · baseline · gates green (ruff/pytest 33/facade/eval 5/5); gitignore hygiene · PASS
- 2026-06-21 · 0.1 README rewrite · top-tier README, real eval table · PASS (ruff/pytest/grep; eval unchanged)
- 2026-06-21 · 0.2 docs/demo.md · recording script + human NOTE · PASS (docs-only)
- 2026-06-21 · 1.1 docs/BENCHMARKS.md · 5/5 table + ablation centerpiece · PASS (docs; eval 5/5 quoted)
- 2026-06-21 · 1.2 docs/ARCHITECTURE.md · spine+facade+verifiers+runtime, mermaid · PASS (docs-only)
- 2026-06-21 · 1.3 docs/VERIFICATION_FIRST.md · thesis: compounding math + RLVR link + refs · PASS (docs-only)
- 2026-06-21 · 0.1(h) README docs links · added Documentation section (all targets exist) · PASS (docs-only)
- 2026-06-21 · 1.4 clean-checkout repro · deleted gitignored state + fresh daemon → eval 5/5; documented · PASS (eval 5/5)
- 2026-06-21 · 2.1 CI workflow · .github/workflows/ci.yml mirrors local gate; yaml+grep validated · PASS (ci-file)
- 2026-06-21 · 2.2 pyright · fixed 21 type errors → 0; config + CI step; eval 5/5 · PASS
- 2026-06-21 · 2.5 governance · CONTRIBUTING/SECURITY/CoC/issue+PR templates; LICENSE→needs-human · PASS (docs-only)
