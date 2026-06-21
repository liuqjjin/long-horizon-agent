# Why verification-first

The thesis behind this project, and why the loop is built around an *objective
oracle* rather than an LLM judging its own work.

## 1. Long-horizon agents fail because errors compound

Model a task as `n` sequential steps, each succeeding independently with
probability `p` (per-step reliability), with no recovery. End-to-end success is

```
P(success) = pⁿ
```

so it decays geometrically in the horizon `n`. Define the **reliable horizon**
`n*(τ)` as the longest task that still succeeds with probability at least `τ`:

```
pⁿ ≥ τ   ⇒   n*(τ) = ln τ / ln p = ln(1/τ) / ln(1/p)
```

For a strong model, `p` is close to 1. Writing the per-step error as `ε = 1 − p`
and using `ln(1/p) = −ln(1 − ε) ≈ ε`, the horizon is approximately

```
n*(τ) ≈ ln(1/τ) / ε
```

Two consequences:

- The horizon scales like **1/ε**. **Halving the per-step error roughly doubles the
  number of steps you can chain** before reliability collapses. Capability at the
  *step* level buys super-linear capability at the *task* level.
- Conversely, even an excellent model (say `p = 0.99`) drifts: at `τ = 0.5`,
  `n* ≈ ln 2 / 0.01 ≈ 69` steps. Real research tasks are longer.

This is a simplified model — steps are not perfectly independent, difficulty
varies, and some errors are silently absorbed. But the qualitative law (geometric
decay; horizon ∝ 1/ε) is the reason long-horizon autonomy is hard, and it tells us
exactly where leverage is: **drive down per-step error**.

## 2. Self-correction alone doesn't break the spiral

The tempting fix is to ask the model to check its own work. But without an external
signal, intrinsic self-correction is unreliable: a model that produced a wrong step
has no independent ground truth to recognize it as wrong, and self-critique can
even degrade correct answers. Huang et al. (ICLR 2024) found LLMs *cannot* reliably
self-correct reasoning without external feedback. So self-reflection raises `p` only
marginally — not enough to change the `1/ε` story.

What *does* change `p` is an **external, objective** check.

## 3. Verification is easier than generation — so gate on it

For many step types there is a checker that is far cheaper and more reliable than
the generator:

- **code** → run the test suite (it passes or it doesn't);
- **experiments** → recompute the metric from the output (PSNR/SSIM with a fixed
  `data_range`), and re-run for reproducibility;
- **context** → is the index behind the source (freshness), and does every claim
  resolve to a real source (citation)?

This generation/verification asymmetry is the same intuition that powers test-driven
software (SWE-bench/SWE-agent use the repo's own tests as the oracle) and
**reinforcement learning from verifiable rewards (RLVR)** — training against
rule-based checkers (unit tests, math answer-keys) rather than a learned reward
model, as used in Tülu 3 and DeepSeek-R1. RLVR applies a verifiable signal at
*training* time; this harness applies the same idea at *inference* time: **gate
every step on a verifiable check, and only advance when it passes.**

A verifier with good recall `r` that triggers a repair turns per-step error `ε`
into something closer to `ε·(1 − r)` (the errors it catches get fixed instead of
propagating). Because the horizon scales like `1/ε`, multiplying `ε` down
multiplies the reliable horizon up. That is the whole bet.

## 4. How this project embodies it

- The loop **never advances on an unverified step** — `context → execute → verify →
  (repair | advance)`. A failed verdict feeds its failures back into a repair
  attempt; an exhausted budget fails the step rather than pretending success.
- Verifiers **recompute** rather than trust. The PSNR/SSIM verifiers recompute the
  metric from the saved arrays, so a fabricated `metrics.json` is caught — not just
  a missing one.
- A verifier that *cannot* run its check returns a **failing** check. "Couldn't
  verify" must never read as "verified."
- The harness **refuses unverifiable success**: the `verification-ablation`
  benchmark sets an unreachable metric bar and the run is reported `FAILED` — see
  [BENCHMARKS.md](BENCHMARKS.md).

## 5. Honest limits

- **Not every step has an objective oracle.** Where one doesn't, this project uses
  the strongest *available* signal (freshness, citation-resolution) — weaker than a
  test suite, and clearly labelled as a context-family check rather than ground truth.
- **The oracle bounds the loop.** A loop is only as reliable as its verifier: a
  flaky test or a too-loose threshold caps the gain. The design response is to make
  verifiers objective, recompute-from-source, and fail-closed.
- **This is a harness, not a model.** It reduces *compounding*; it does not make a
  weak step-policy strong. It pairs best with a capable implementer.

## Key references

- J. Huang, X. Chen, S. Mishra, et al. *Large Language Models Cannot Self-Correct
  Reasoning Yet.* ICLR 2024.
- C. E. Jimenez, J. Yang, et al. *SWE-bench: Can Language Models Resolve Real-World
  GitHub Issues?* ICLR 2024.
- J. Yang, C. E. Jimenez, et al. *SWE-agent: Agent-Computer Interfaces Enable
  Automated Software Engineering.* NeurIPS 2024.
- N. Lambert, J. Morrison, et al. *Tülu 3: Pushing Frontiers in Open Language Model
  Post-Training.* 2024. (introduces RLVR — RL from verifiable rewards)
- DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning.* 2025. (rule-based verifiable rewards)
- N. Shinn, F. Cassano, et al. *Reflexion: Language Agents with Verbal Reinforcement
  Learning.* NeurIPS 2023. (verbal feedback from outcome signals)

*Citations are to real, well-known works; arXiv identifiers are intentionally
omitted to avoid transcription error — search by title.*
