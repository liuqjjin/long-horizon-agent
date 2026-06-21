# BACKLOG — overnight optimization (work top-down; each item independently shippable)

## Tier 0 — first impression / README (HIGHEST leverage)
- 0.1 Rewrite README.md, skimmable, in order: (a) one-line tagline naming the differentiator — verification-first reliability via an OBJECTIVE oracle (test-pass + PSNR/SSIM), not LLM-as-judge; (b) 2–3 sentence hook: long-horizon agents fail because errors compound step-by-step; this harness gates every step on an objective oracle to dampen that; (c) a mermaid diagram of the spine (facade → three verifier families → durable runtime); (d) a 30-second Quickstart (`uv sync`; one demo command; `lha eval`); (e) Results section with the REAL `lha eval` table (run it, quote actual output); (f) How it works (spine, facade, CocoIndex isolation, LangGraph durable HITL); (g) Why it's different vs typical agent frameworks; (h) links to docs/; (i) install/requirements. Add only substantiable badges (license, python, ruff, tests); CI badge only after 2.1.
- 0.2 Prepare demo asset: docs/demo.md with the EXACT command sequence to record an asciinema/GIF (`asciinema` + `agg`) of run → AWAITING_APPROVAL → approve → resume → DONE, plus `lha eval` 5/5. Embed a placeholder image link. If no TTY/recorder headless, leave the script + a NOTE for the human to record. (Demo GIF is the #1 star driver.)
- 0.3 Draft repo metadata for the human (can't edit GitHub settings): in the morning report propose a repo description (≤350 chars) and ~15 topics (llm-agents, ai-agents, langgraph, mcp, rag, llm-evals, research-agent, verification, cocoindex, durable-execution, agent-harness, …).

## Tier 1 — make the differentiator LEGIBLE and CREDIBLE
- 1.1 docs/BENCHMARKS.md: table of the 5 ResearchAgentBench-Lite tasks — what each verifies + ACTUAL pass result from `lha eval`. Centerpiece = verification-ablation: the harness CORRECTLY FAILS an unreachable PSNR bar (refuses success it can't verify); if the eval can emit a with/without-verifier contrast, show it. Document the exact reproduce command.
- 1.2 docs/ARCHITECTURE.md: condense the system — verification-loop spine, the facade (six entry points) with CocoIndex isolated behind it, the three verifier families (code/experiment/context), LangGraph durable runtime, multi-agent team, skill memory. Include the mermaid diagram. Adapt existing notes; don't bloat.
- 1.3 docs/VERIFICATION_FIRST.md: the thesis — long-horizon error compounding (p^n; reliable horizon ~ ln(1/τ)/(1−p), so halving step-error doubles horizon), why intrinsic self-correction is unreliable without an external signal, why an objective oracle dampens compounding, and the link to RLVR (verifiable rewards). Rigorous; cite key papers. The intellectual signature.
- 1.4 One-command reproducibility: verify `uv sync && lha eval` reproduces 5/5 from a CLEAN checkout; document it; fix any hidden state/path/CWD assumptions.

## Tier 2 — engineering-rigor signals
- 2.1 GitHub Actions CI (.github/workflows/ci.yml): on push/PR, set up uv + python, run `ruff check .`, `uv run pytest`, the facade-isolation grep, `lha eval`. Mirror the local closing gate EXACTLY. (Green CI badge = major credibility.)
- 2.2 Type-check verifier: integrate pyright (or mypy), add to the gate + CI, fix types incrementally (one module/iteration). Extends the verifier-families story to the harness itself.
- 2.3 Coverage: configure `pytest --cov`, report it, raise coverage on the weakest modules with MEANINGFUL tests (no trivial padding). Add a coverage badge/threshold.
- 2.4 pre-commit config (ruff format+lint, end-of-file-fixer, …); document in CONTRIBUTING.
- 2.5 Governance/community files: CONTRIBUTING.md (uv dev setup; how to run tests + `lha eval`; verification-first contribution rule = no claim without a runnable check), CODE_OF_CONDUCT.md, .github/ISSUE_TEMPLATE/*, PULL_REQUEST_TEMPLATE.md, SECURITY.md. Verify LICENSE exists; if missing, DON'T pick one silently — record under "needs human decision" (recommend MIT or Apache-2.0).
- 2.6 CHANGELOG.md (Keep-a-Changelog) from git history; adopt semver; draft v0.1.0 release notes (do NOT tag/publish — record under "needs human decision").

## Tier 3 — usability / examples / onboarding
- 3.1 examples/ with runnable, self-contained demos (each with README + one-command runner, each exercised by the gate): (a) issue_to_pr/ on a tiny bundled sample repo; (b) paper_to_experiment/ with a small PSNR/SSIM objective-oracle task; (c) resume_hitl/ showing run → AWAITING_APPROVAL → approve → resume → DONE on the LangGraph runtime.
- 3.2 docs/QUICKSTART.md: 0 → running in <5 min from a clean clone, with expected output.
- 3.3 CLI polish: audit `lha --help` and every subcommand for clear help, usage examples, sensible errors; add `lha --version`.
- 3.4 Bundle a tiny demo task/dataset so one `lha` command does something visibly impressive WITH objective verification right after clone.

## Tier 4 — depth features (LOW-RISK only; don't over-engineer unattended)
- 4.1 Episodic failure memory (Reflexion-lite): persist verified FAILURES as lessons (kind="failure"), indexed + retrievable via the existing search_* facade, mirroring skill-memory; gate behind tests.
- 4.2 Grow the eval corpus carefully: add 1–2 tasks whose pass/fail is OBJECTIVELY checkable (e.g. a real small GitHub issue). Never invent expected results.
- 4.3 Observability: document the ledger/trace; optionally add a minimal `lha trace`/summary command or a Langfuse/LangSmith integration DOC (no hard dependency).

## CROSS-CUTTING
After each change, keep README, docs/, the plan, and project memory consistent.