# Overnight optimization log

## ☀️ Morning report (2026-06-21)

A documentation, presentation, and engineering-rigor pass: **17 improvement commits**
(this report is the 18th), each small, reversible, and conventional; every commit
left the gate green.

**Final gate (verbatim):** `ruff check .` clean · `pyright src/lha` 0 errors ·
`pytest` 41 passed · facade-isolation grep clean · `lha eval` **5/5**.

**What changed, by theme**
- *First impression (Tier 0):* rewrote the README around the differentiator
  (objective-oracle verification, not LLM-as-judge) with an error-compounding hook,
  a mermaid spine diagram, a 30s quickstart, and the verbatim `lha eval` 5/5 table;
  added `docs/demo.md` (GIF recording script for a human to run).
- *Differentiator made legible (Tier 1):* `docs/BENCHMARKS.md` (per-task oracle +
  the verification-ablation centerpiece), `docs/ARCHITECTURE.md` (spine/facade/
  verifiers/runtime + diagrams), `docs/VERIFICATION_FIRST.md` (the thesis: error
  compounding `pⁿ`, horizon `≈ ln(1/τ)/ε`, RLVR link, real references), and verified
  clean-checkout reproducibility (`uv sync && lha eval` → 5/5).
- *Engineering rigor (Tier 2):* GitHub Actions CI mirroring the gate; **pyright**
  driven to 0 and added to the gate + CI (type-only fixes, no logic change);
  coverage measured (65%→**69%**) via 8 meaningful unit tests; pre-commit config;
  governance (CONTRIBUTING with the "no claim without a runnable check" rule,
  SECURITY, CODE_OF_CONDUCT, issue/PR templates); CHANGELOG (draft 0.1.0).
- *Polish (Tier 3):* `lha --version` + CLI help examples; `docs/QUICKSTART.md`
  (0→verified-run with real expected output); applied `ruff format` repo-wide.

**Proposed repo metadata (for the human — I can't edit GitHub settings)**
- *Description (≤350):* "Verification-first long-horizon agent harness: every step
  is gated by an objective oracle (real pytest/ruff for code, recomputed PSNR/SSIM
  for experiments, freshness/citation for context), not an LLM judge. Three verifier
  families behind one facade, checkpoint/resume, an opt-in LangGraph durable runtime,
  and a 5/5 self-eval."
- *Topics (~15):* llm-agents, ai-agents, agent-harness, agentic-ai, langgraph, mcp,
  rag, llm-evals, research-agent, verification, cocoindex, durable-execution,
  reproducibility, human-in-the-loop, python.

**Needs human decision** (details under that section below)
- **LICENSE** — none present; pick MIT or Apache-2.0 (blocks reuse + a license badge).
- **CI badge** — add after first green Actions run (needs the GitHub owner/repo slug).
- **Demo GIF** — record via `docs/demo.md` and embed in the README (top star-driver).
- **CODE_OF_CONDUCT contact** — add a concrete maintainer contact if desired.
- **Coverage badge** — wire Codecov (a static badge would rot).

**Top 3 next steps**
1. Add a LICENSE (MIT/Apache-2.0) and set `[project].license` — unblocks everything else.
2. Push to GitHub; confirm CI is green; add the CI badge; record + embed the demo GIF.
3. Backlog depth: `examples/` (runnable self-contained demos, Tier 3.1) and episodic
   failure memory (Reflexion-lite, Tier 4.1); optionally harden `lha eval` against a
   churned-daemon transient miss (the only known flakiness; clean checkout is reliable).

---

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
- 2.3 coverage — pytest-cov + 8 meaningful unit tests (MCP parse/diff/factory); 65%→69%; documented.
- 2.4 pre-commit — `ruff format` repo-wide + .pre-commit-config.yaml (ruff lint/format + hygiene).
- 2.6 CHANGELOG.md — Keep-a-Changelog; draft 0.1.0 (untagged → needs human).
- 3.3 CLI polish — `lha --version` + help epilog examples.
- 3.2 docs/QUICKSTART.md — 0→verified-run tutorial with real expected output; linked from README.
- 0.3 repo metadata — proposed description + topics (see Morning report).

## Remaining (from BACKLOG, top-down)
- Tier 3: 3.1 examples/ (runnable self-contained demos) · 3.4 bundle a one-command demo task
- Tier 4: 4.1 episodic failure memory · 4.2 grow eval corpus · 4.3 observability (`lha trace`)
- Cross-cutting: optional — harden `lha eval` vs a churned-daemon transient miss (clean checkout is reliable)

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
- 2026-06-21 · 2.6 CHANGELOG.md · Keep-a-Changelog; draft 0.1.0 (not tagged → needs-human) · PASS (docs-only)
- 2026-06-21 · 3.3 CLI polish · added `lha --version` + help epilog examples · PASS (parser-only; eval result unaffected, parse verified)
- 2026-06-21 · 2.4(format) ruff format · 26 files reformatted; eval 5/5 · PASS (eval 5/5)
- 2026-06-21 · 2.4 pre-commit · .pre-commit-config.yaml (ruff lint/format + hygiene) + CONTRIBUTING note · PASS (config-only)
- 2026-06-21 · 2.3 coverage · pytest-cov; +8 meaningful tests (MCP parse/diff/factory) 65%→69%; documented · PASS
- 2026-06-21 · 3.2 docs/QUICKSTART.md · 0→running tutorial w/ real expected output; linked from README · PASS (docs-only)
- 2026-06-21 · morning report · summary + repo metadata + needs-human + next steps; reconciled lists · PASS (docs-only)
