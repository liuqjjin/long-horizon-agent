# Contributing

This project has one rule, which follows from its thesis (see
[docs/VERIFICATION_FIRST.md](docs/VERIFICATION_FIRST.md)):

> **No claim without a runnable check.** Any behavior, number, or benchmark you add
> or assert in docs must be backed by something runnable in this repo. To state a
> result, run it and quote the actual output. This applies to the agent and to contributors.

## Dev setup

```bash
uv sync                 # installs the harness + dev tools (Python 3.11+)
uv run pre-commit install   # optional: auto-run ruff lint+format on commit
```

## The gate (run before every PR)

Every change must keep all of these green — this is exactly what CI runs:

```bash
uv run ruff check .                         # lint
uv run pyright src/lha                       # type-check
uv run pytest -q                             # unit tests (hermetic)
uv run lha eval                              # self-eval — must be 5/5
LHA_DOCKER_TESTS=1 uv run pytest tests/test_sandbox.py -q   # opt-in: real containers (needs a docker daemon)

# facade isolation: CocoIndex must never leak out of live_context (must print nothing)
grep -rnE "^[[:space:]]*(import|from)[[:space:]]+(cocoindex|cocoindex_code)" \
  --include='*.py' src/lha | grep -v "src/lha/live_context/"
```

`lha eval` downloads a small embedding model on first run (one-time). It is
reliably `5/5` from a clean checkout; if it shows a transient code-context miss,
restart the `ccc` daemon (`ccc daemon restart`) and re-run.

### Coverage

```bash
uv run pytest --cov=lha --cov-report=term-missing
```

Line coverage is currently **73%** (65 tests). The uncovered lines are mostly the
network/CLI-bound backends (`ccc` MCP I/O, the `claude_cli`/`anthropic` LLM clients)
that can't be unit-tested hermetically; their pure logic (result parsing, the LLM
factory, diff extraction) *is* tested. New code should come with a meaningful test —
not padding to move the number.

## Ground rules

- **Never weaken a check to pass.** Don't skip, `xfail`, comment out, or delete a
  test, lower a threshold, or loosen a verifier to make the gate green. Fix the real
  cause, or revert the change.
- **Keep CocoIndex behind the facade.** Nothing outside `src/lha/live_context/` may
  import `cocoindex`/`cocoindex_code` or call `ccc`. The grep above enforces it.
- **Prefer small, reversible changes** with a clear conventional-commit message
  (`feat:`/`fix:`/`docs:`/`test:`/`ci:`/`chore:`/`refactor:`).
- **Add a verifier, not a special case.** New objective checks belong in a verifier
  family (`src/lha/verifiers/<family>/`) and are selected per step — see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Where things live

`docs/ARCHITECTURE.md` is the map. In short: the loop spine is `src/lha/harness/`,
the only door to indexed context is `src/lha/live_context/`, verifiers are in
`src/lha/verifiers/`, and `lha eval` is `src/lha/eval.py`.
