# Security Policy

This is a research/portfolio project, not a production service, but reports are
welcome.

## Scope

The harness executes code and shell commands as part of verification (e.g. running
`pytest` and experiment scripts in a per-run sandbox under `runs/<id>/workdir/`).
Run tasks you trust; treat task specs and target repos as code you are about to
execute.

## Known limitations

- **The human-approval gate is not yet content-bound to the approved artifact.**
  On resume, the step's artifact is regenerated. With the default deterministic
  backend (`stub`) the regenerated patch is identical to the one approved; with an
  opt-in non-deterministic backend (`claude_cli`/`anthropic`) it could differ from
  what the human reviewed. Binding the approval to a content hash is planned.

## Reporting a vulnerability

Please report privately via GitHub's **"Report a vulnerability"** (Security
advisories) on the repository rather than opening a public issue. Include steps to
reproduce and the affected version/commit. We'll acknowledge and respond as soon as
we reasonably can.

Please do not include secrets or credentials in reports.
