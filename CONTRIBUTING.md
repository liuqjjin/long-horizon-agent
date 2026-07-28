# Contributing

Any documented behavior, benchmark number, test count, or coverage value must
come from a command run on the commit that contains the claim.

## Development setup

Use Python 3.11+ and run from the repository root:

```bash
uv sync
uv run pre-commit install  # optional
```

The test suite is hermetic by default: it uses the deterministic LLM stub, a null
code-index backend, temporary data directories, and no network. Keep new unit
tests the same way. Live CLI, Docker, and public-benchmark checks belong in
explicit opt-in tests or release commands.

## Local gate

Every pull request must pass:

```bash
uv run ruff check .
uv run pyright src/lha
uv run pytest -q
LHA_RUNS_DIR=runs/_pr uv run lha eval
```

Facade isolation must print nothing:

```bash
if grep -rnE "^[[:space:]]*(import|from)[[:space:]]+(cocoindex|cocoindex_code)" \
     --include='*.py' src/lha | grep -v "src/lha/live_context/"; then
  echo "CocoIndex import escaped the live_context facade"
  exit 1
fi
```

For a release candidate, also run:

```bash
uv run python -m lha.release_claims
uv run python tools/verify_terminal_source_build.py \
  --root . \
  --evidence benchmarks/terminal_bench_2_1
uv build --clear
docker build -t lha:release .
LHA_DOCKER_TESTS=1 LHA_DOCKER_TEST_IMAGE=lha:release \
  uv run pytest tests/test_sandbox.py -q
docker run --rm lha:release lha --version
docker run --network none --rm lha:release lha eval
```

Follow [docs/DEPLOY.md](docs/DEPLOY.md) to install both the wheel and source
distribution from empty scratch directories and verify that packaged context
flows are present.

Host-side `lha eval` may download the configured embedding model on its first
run. The application image includes a pinned model snapshot and must complete
the command without network access. A download or Docker-daemon failure is not
a successful check.

## Coverage

Generate current coverage when it is useful:

```bash
uv run pytest --cov=lha --cov-report=term-missing
```

Do not copy a previous test count or percentage into a pull request. Include the
actual output from the candidate commit. Add tests for behavior and failure
modes; do not add assertions solely to increase a percentage.

## Change rules

- Never skip, delete, weaken, or mark a check `xfail` to get a green build.
- A verifier that cannot execute must return a failing `Check`.
- New objective checks belong in a verifier family and registry, not a branch in
  the harness loop.
- Boundary data should be a Pydantic model; internal value objects can be
  dataclasses.
- Use `lha.clock.now()` for timestamps.
- Route target or model-influenced subprocesses through `ExecutionBackend`.
- Keep optional dependency imports inside the function that needs them.
- Keep CocoIndex, `cocoindex_code`, and `ccc` calls inside
  `src/lha/live_context/`.
- Comments should explain the failure mode a decision prevents.
- Prefer conventional commit subjects such as `feat:`, `fix:`, `docs:`,
  `test:`, `ci:`, `chore:`, and `refactor:`.

## Recovery changes

Changes to patching, checkpointing, approval, or resume must preserve these
properties:

- the write set comes from `ResolvedPatch`, not declared `touched_files`;
- policy, backup, manifest, apply, approval, and rollback use the same paths;
- `PatchTransaction` transitions remain
  `PREPARED → APPLIED → VERIFIED`, with rollback to `REVERTED`;
- a transaction has durable patch, manifest, journal, and redundant backups;
- state corruption and an invalid ledger event chain fail closed;
- schema-v1 state is not resumed as schema v2;
- concurrent resume is rejected by the run lock;
- the ledger grows logically by event; its implementation validates and
  atomically replaces the complete file rather than using `O_APPEND`;
- ledger attempt IDs and idempotency keys do not duplicate side effects;
- unverified work never survives failure, rejection, or exhausted repair.

Add adversarial tests for the crash windows you change. Relevant suites include:

```bash
uv run pytest \
  tests/test_patch_transactions.py \
  tests/test_crash_injection.py \
  tests/test_approval_binding.py \
  tests/test_review_fixes.py -q
```

## Long-task corpus

`data/long_tasks/` has five pre-fixed repositories with adapter specs, reference
patches, and reference manifests. Their 10-step protocol is exercised by:

```bash
uv run pytest tests/test_long_tasks.py tests/test_long_task_harness.py -q
```

Do not change a repository, oracle, reference patch, or digest after reading
model results. New cases must be authored and frozen before a scored run, and
must demonstrate baseline failure plus reference-patch success.

Repository adapters define setup and check stages. Adding an adapter does not
create a benchmark result; a result also needs a fixed protocol, raw evidence,
provenance, and a committed summary.

## Ablation and statistics

Do not use the internal gate as truth. The final scorer must grade a fresh
canonical repository through a separate backend instance.

Keep these units distinct:

- one paired task/repetition cell;
- one complete-corpus episode per repetition;
- a descriptive horizon composition that adds zero observations.

Cell and episode McNemar p-values may differ. Use Wilson intervals at all-zero or
all-one boundaries; a percentile bootstrap there produces a misleading
zero-width interval.

Changes affecting prompts, patch application, policy, scoring, aggregation, or
runtime provenance must invalidate the ablation cache fingerprint.

Never edit `benchmarks/*.json` or copied result numbers by hand. Regenerate the
report and update all public citations in the same change. Planned repetitions
and unfinished public-benchmark runs are not results.

Formal ablation does not use the exploratory cache and cannot resume. Before it
starts, append and commit a `REGISTERED` event that fixes the source, corpus,
model, CLI and client settings, Docker image, output path, and witness remote.
The runner must create the registered remote witness ref before the first cell.
If preflight or any cell is interrupted, record `ABANDONED`; do not delete the
output and repeat the same outcome-affecting selection.

## Codex changes

The Codex backend must:

- pass a minimal subprocess environment;
- use an attempt-local `HOME`, `CODEX_HOME`, workspace, and temp directory;
- copy authentication without logging it;
- start a separate process group and stop descendants before cleanup;
- fail on malformed or incomplete JSONL, unknown events, and disallowed tool use;
- distinguish deterministic protocol failure from bounded transient retry;
- record secret-free CLI/model/event/outcome provenance.

Add protocol and cleanup tests without requiring live authentication.
Cleanup guarantees apply to normal return, failure, timeout, and handled
interruption. `SIGKILL`, a kernel crash, or power loss can leave a
mode-protected temporary directory and requires manual inspection.

## Documentation and pull requests

Update architecture, security, quickstart, deployment, and changelog text when
behavior changes. Pull requests should include:

- the problem and why the change is needed;
- the invariant or public contract affected;
- exact commands run and their output;
- new adversarial or regression checks;
- generated report provenance if any number changed;
- explicit disclosure of checks not run and why.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map and
[SECURITY.md](SECURITY.md) for execution and credential boundaries.
