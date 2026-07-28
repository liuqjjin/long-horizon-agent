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

It is **not** isolation against malicious code. Use it only for repositories and
commands you trust.

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
- `ledger.jsonl` is append-only. A torn final write is handled as an interruption;
  corruption in a durable record is rejected.
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

The CLI runs in a new process group. Timeout, exception, or keyboard interruption
terminates the leader and descendants before the temporary credential directory
is removed.

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
- Docker reduces exposure but does not prove that a task image or Docker daemon
  is trustworthy.
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
