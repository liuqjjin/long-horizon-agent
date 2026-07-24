# Security Policy

This is a research/portfolio project, not a production service, but reports are
welcome.

## Scope

The harness executes code and shell commands as part of verification (e.g. running
`pytest` and experiment scripts in a per-run sandbox under `runs/<id>/workdir/`).
Run tasks you trust; treat task specs and target repos as code you are about to
execute.

## Execution backends

Everywhere target or model-influenced code runs goes through
`lha.sandbox.ExecutionBackend` (`LHA_EXEC_BACKEND`):

- **`trusted-local`** (default): runs on the host with a scrubbed environment and
  a process-group kill on timeout. It is **not** a sandbox against malicious
  code — it is for this repo's own dev loop and self-eval.
- **`docker`**: runs in a container with `--network none`, an empty environment,
  memory/pids caps, and read-only source mounts (`LHA_EXEC_IMAGE`). Use this for
  external target repos and for scoring model-written patches. The image must
  provide the tools the tasks need — the default `python:3.12-slim` has no
  `pytest`/`ruff`, so code-verification tasks require an image with those
  installed (e.g. `FROM python:3.12-slim` + `pip install pytest
  pytest-json-report ruff`).

## Integrity mechanisms (current state)

- **Approvals bind to artifact bytes** in both runtimes: a decision names the
  step and the SHA-256 of the reviewed `patch.json`; on resume, both the default
  loop and the LangGraph runtime execute only those exact bytes, and a hash
  mismatch reverts the change and fails the run
  (`tests/test_approval_binding.py`).
- **Protected oracle paths**: patches that touch tests, `conftest.py`, or
  build/CI config are refused before they reach the sandbox
  (`lha.tools.policy`), unless the task manifest explicitly allowlists them.
- **Checkpoints refuse damage**: `state.json` is a checksummed envelope; a failed
  integrity check refuses to resume rather than guessing
  (`tests/test_crash_injection.py`).

## Known limitations

- Prompt injection through indexed content is mitigated only by verification —
  a poisoned suggestion still has to pass the objective gate — not prevented.
- `trusted-local` offers no isolation beyond environment scrubbing; choose the
  backend to match how much you trust the target repo.

## Reporting a vulnerability

Please report privately via GitHub's **"Report a vulnerability"** (Security
advisories) on the repository rather than opening a public issue. Include steps to
reproduce and the affected version/commit. We'll acknowledge and respond as soon as
we reasonably can.

Please do not include secrets or credentials in reports.
