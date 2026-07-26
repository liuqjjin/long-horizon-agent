## Problem and scope

<!-- What failed or was missing? Why is this the smallest relevant change? -->

## Contract affected

<!-- Name the state transition, verifier, CLI contract, security boundary, or public claim. -->

## Implementation

<!-- Summarize behavior. Call out migration, rollback, compatibility, and failure handling. -->

## Verification evidence

<!-- Paste the real output or link to the exact CI job. Do not copy counts from an older commit. -->

- [ ] `uv run ruff check .`
- [ ] `uv run pyright src/lha`
- [ ] `uv run pytest -q`
- [ ] `LHA_RUNS_DIR=runs/_pr uv run lha eval`
- [ ] facade-isolation scan prints nothing
- [ ] wheel and source distribution build
- [ ] wheel installs and imports packaged flows from an empty directory
- [ ] source distribution installs and imports packaged flows from an empty directory
- [ ] `LHA_DOCKER_TESTS=1 LHA_DOCKER_TEST_IMAGE=lha:release uv run pytest tests/test_sandbox.py -q`
- [ ] Docker image builds and `lha --version` runs
- [ ] Docker image contains no local credential/config paths checked by CI

Commands not run and why:

<!-- A missing daemon, credential, model, or network is a disclosed gap, not a pass. -->

## Adversarial and recovery cases

<!-- What happens on malformed input, timeout, interruption, duplicate resume, and damaged evidence? -->

- [ ] no test was skipped, deleted, weakened, or marked `xfail` to pass
- [ ] no verifier, threshold, protected-path policy, or denominator was loosened
- [ ] unexecutable checks still fail closed
- [ ] unverified changes are rolled back
- [ ] recovery does not duplicate side effects

## Security and credentials

<!-- Note host/container execution, environment handling, temp credentials, and cleanup. -->

- [ ] no token, authentication file, private path, or secret-bearing log is committed
- [ ] external target code uses an appropriate Docker execution image
- [ ] generated HTML/output was reviewed before sharing

## Data, statistics, and documentation

- [ ] corpus repositories, oracles, and reference patches were not changed after observing results
- [ ] internal gate output is not reused as benchmark truth
- [ ] cell, episode, and composition units are labelled separately
- [ ] all-zero/all-one proportions use Wilson intervals
- [ ] benchmark/report numbers were generated, not hand-edited
- [ ] README, architecture, security, changelog, and method docs match behavior
- [ ] unfinished Terminal-Bench or SWE-bench work is not presented as a score

Generated report provenance, if applicable:

<!-- Git commit/dirty state, source digest, model, effort, CLI version, scorer/image, repetitions, errors. -->
