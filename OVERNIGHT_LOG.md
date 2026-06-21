# Overnight optimization log

Autonomous maintainer working `tools/overnight/BACKLOG.md` top-down. One small,
reversible increment per entry; all gates green before every commit.

**Gate** = `uv run pytest` · `ruff check .` · facade-isolation grep · `lha eval` (5/5).
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

## Remaining (from BACKLOG, top-down)
- Tier 0: 0.3 repo metadata (morning report)
- Tier 1: 1.1 BENCHMARKS · 1.2 ARCHITECTURE · 1.3 VERIFICATION_FIRST · 1.4 clean-checkout repro
- Tier 2: 2.1 CI · 2.2 type-check · 2.3 coverage · 2.4 pre-commit · 2.5 governance · 2.6 CHANGELOG
- Tier 3: 3.1 examples · 3.2 QUICKSTART · 3.3 CLI polish · 3.4 demo task
- Tier 4: 4.1 failure memory · 4.2 eval corpus · 4.3 observability
- Cross-cutting: harden `lha eval` against daemon-state flakiness (reliability of the 5/5 claim)

## Needs human decision
- (none yet)

## Iterations
- 2026-06-21 · baseline · gates green (ruff/pytest 33/facade/eval 5/5); gitignore hygiene · PASS
- 2026-06-21 · 0.1 README rewrite · top-tier README, real eval table · PASS (ruff/pytest/grep; eval unchanged)
- 2026-06-21 · 0.2 docs/demo.md · recording script + human NOTE · PASS (docs-only)
- 2026-06-21 · 1.1 docs/BENCHMARKS.md · 5/5 table + ablation centerpiece · PASS (docs; eval 5/5 quoted)
