# Benchmarks

Committed snapshots of benchmark runs, so the numbers are inspectable without
re-running (generated output under `runs/` is gitignored).

- **`ablation_report.md`** / **`ablation_report.json`** — the verification ablation:
  a real LLM driven through the harness under `trust` / `gate` / `verify`, measuring
  `claimed` vs `true` vs `false` success. Method, integrity properties, and how to
  reproduce: [`docs/ABLATION.md`](../docs/ABLATION.md). Regenerate with:

  ```bash
  uv run lha --llm claude_cli ablate --model claude-haiku-4-5-20251001 --reps 3
  cp runs/ablation/ablation_report.* benchmarks/
  ```

The deterministic self-eval (`lha eval`) is documented
separately in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md); it runs in CI on a stub,
whereas the ablation needs a real model and so is committed here as a snapshot.
