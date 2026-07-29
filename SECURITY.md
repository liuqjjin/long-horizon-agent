# Security policy

LHA is maintained for research and engineering evaluation, not as a production
service. Security reports are welcome.

## Scope

The harness runs target code, tests, linters, build commands, and experiment
scripts. A task specification and its target repository should be treated as
code you are about to execute.

The default worktree under `runs/<id>/workdir/` is a path boundary and recovery
boundary. It is not automatically an operating-system sandbox.

## Execution backends

Target or model-influenced commands use `lha.sandbox.ExecutionBackend`.

### `trusted-local`

The default backend:

- runs on the host;
- supplies a small environment rather than inheriting arbitrary secrets;
- creates a process group and terminates it on timeout;
- supports configured resource limits where the host implements them.

It is **not** isolation against malicious code. Target processes still have the
current user's host permissions and may access files or services available to
that user even when environment variables are removed. Use it only for
repositories and commands you trust.

### `docker`

The Docker backend runs a disposable container with network disabled, memory
and process limits, and read-only source mounts where the operation permits. It
does not inherit the host process environment; image-defined variables remain,
and LHA sets `HOME=/tmp`. The working directory and a restricted `/tmp` tmpfs
are writable. Use it for external repositories and the separate ablation
scoring path.

The execution image must contain every command a task needs. The default
`python:3.12-slim` image does not include pytest, `pytest-json-report`, or Ruff;
select a purpose-built `LHA_EXEC_IMAGE` for code verification.

The LHA application image documented in `docs/DEPLOY.md` is not automatically
the execution-backend image.

LHA resolves the Docker client to an absolute path, records its size and
SHA-256, and checks the same bytes before and after backend operations. A
standard macOS Docker Desktop installation may be owned by the logged-in
operator; that fact is recorded but is not treated as a failed boundary. The
host, Docker daemon, and operator account are trusted. A malicious process
already running as the same user could temporarily replace and restore the
client, source checkout, credentials, or report files and is outside this
project's threat model.

## Patch and recovery integrity

- `ResolvedPatch` computes the write set from the actual unified diff or file
  contents. Model-declared `touched_files` is not an authority.
- Protected paths include tests, `conftest.py`, package/build configuration, and
  CI configuration. A task can allow only an exact declared protected path.
- Path resolution rejects absolute paths, `..`, case variants of protected
  names, and writes through symbolic links.
- `PatchTransaction` records `PREPARED`, `APPLIED`, `VERIFIED`, or `REVERTED`.
  Patch, manifest, transaction journal, and redundant backups are persisted
  before the transaction can be replayed.
- Backups retain original bytes, file modes, and patch-created directories.
- Missing, contradictory, or damaged transaction evidence fails closed.
- A late-stage failure rolls back earlier verified patch attempts when the run
  cannot safely deliver the final state.

## State and approval integrity

- `state.json` is a versioned, checksummed envelope written using `fsync` and
  atomic replacement.
- `ledger.jsonl` is logically append-only. The implementation validates the
  existing event chain and atomically replaces the complete file; it does not
  open the file with `O_APPEND`. A torn legacy final line is treated as an
  interruption, while corruption in a complete record is rejected.
- Schema-v1 runs remain inspectable but cannot be resumed as schema v2.
- A per-run lock rejects concurrent resume.
- Stable attempt IDs and idempotency keys prevent duplicate ledger transitions.
- Approval names the step and SHA-256 of the exact persisted `patch.json`.
  Rejection, byte mismatch, or artifact corruption rolls the attempt back.
- LangGraph checkpoints the prepared artifact before its approval interrupt, so
  resume cannot regenerate a different patch after review.

## Codex authentication and process isolation

The Codex CLI backend creates a temporary home, `CODEX_HOME`, workspace, and
temporary directory for each attempt. It copies only the authentication file
required by the CLI and passes an explicit environment allowlist. API keys,
cloud credentials, SSH agent sockets, and arbitrary caller variables are not
forwarded.

The CLI runs in a new process group. On normal return, exception, timeout, or a
handled keyboard interruption, LHA terminates the original process group and
confirms that group is absent before removing the temporary credential
directory.

This is not a guarantee that every descendant process has exited. A tool can
deliberately create another process group or session; after reparenting, the
host backend cannot identify it without a race on both macOS and Linux. The
generated Codex permission profile is checked before credentials are copied and
denies tool processes access to the temporary Codex home, but untrusted tool
execution still requires Docker, a Linux cgroup, or another outer containment
boundary.

That cleanup is cooperative process-exit behavior, not a crash-proof erasure
guarantee. `SIGKILL`, a kernel crash, or power loss can stop the cleanup handler
and leave an attempt-local directory on disk. The directory and credential copy
are created with restrictive modes, but an operator must inspect and remove any
residue before sharing the machine or its storage.

The JSONL protocol fails closed on malformed JSON, unknown events, error events,
incomplete turns, unfinished tool calls, and missing final model output. In
ablation no-tools mode, any tool item invalidates the result.

Reports may record CLI version, model, reasoning effort, event counts, token
usage, status, and timing. They must never record:

- `auth.json` contents;
- API keys, cookies, bearer tokens, or refresh tokens;
- the contents of a user's Codex home;
- credential-bearing environment values.

Do not copy a complete developer home into a container. Benchmark credentials
must be injected into a task-local temporary directory at runtime, never baked
into an image or committed as an artifact.

`LHA_CODEX_SANDBOX=danger-full-access` is rejected unless
`LHA_CODEX_EXTERNAL_SANDBOX=1` explicitly confirms that a disposable outer
sandbox, such as an official benchmark task container, is already in place.

## Reports and deletion

`lha trace --html`, `lha runs list`, `lha runs show`, and `lha runs prune`
validate persisted state before using it. HTML escapes persisted values.

Pruning is a dry run by default. `--apply` can delete only unlocked, validated
`DONE` or `FAILED` runs older than the requested cutoff. Active, paused,
approval-waiting, locked, or corrupt runs are refused.

Run artifacts can contain source code, patches, command output, file paths, and
model prompts. Review them before sharing even though credential bytes should
not be present.

## Known limitations

- `trusted-local` is not a hostile-code sandbox.
- Host process-group cleanup cannot prove that a deliberately detached
  descendant has exited.
- Docker reduces exposure but does not prove that a task image or Docker daemon
  is trustworthy.
- `LHA_DEADLINE_S` is checked when control returns to persisted boundaries. It
  does not asynchronously stop a blocking library call; every such operation
  needs its own timeout.
- Write-once evidence is created directly at its final name. `SIGKILL`, storage
  exhaustion, or power loss during its first write can leave an incomplete
  final file. Recovery rejects inconsistent bytes and may require a new run or
  manual cleanup.
- Atomic replacement can leave a restrictive `.tmp` file if the process cannot
  execute its cleanup handler. Transaction recovery removes only an exact,
  single temporary file whose owner, mode, identity, target, and transition can
  be validated; unknown or multiple files stop recovery.
- Prompt injection through indexed content is mitigated by executable checks,
  not prevented.
- Freshness and citation checks do not prove semantic correctness.
- Tests and Ruff define truth only to the extent that their oracle is complete.
- Public-benchmark adapters are integration code, not evidence of a benchmark
  result.

## Reporting a vulnerability

Use the repository's private GitHub **Report a vulnerability** flow rather than
a public issue. Include:

- the affected version or commit;
- a minimal reproduction;
- the expected and actual security boundary;
- whether credentials, source, or host execution were exposed.

Do not include real secrets in the report. Rotate any credential that may have
been exposed before sending diagnostic artifacts.
