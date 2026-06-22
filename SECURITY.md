# Security Policy

This is a research/portfolio project, not a production service, but reports are
welcome.

## Scope

The harness executes code and shell commands as part of verification (e.g. running
`pytest` and experiment scripts in a per-run sandbox under `runs/<id>/workdir/`).
Run tasks you trust; treat task specs and target repos as code you are about to
execute.

## Known limitations

- **Approval artifact-binding differs by runtime.** The default loop reuses the
  exact patch the human approved when it resumes — it is not regenerated. The
  opt-in LangGraph runtime re-executes the step node on resume (a property of
  `interrupt()`), so with a non-deterministic backend its resumed artifact could
  differ from what was reviewed; binding it identically is planned.

## Reporting a vulnerability

Please report privately via GitHub's **"Report a vulnerability"** (Security
advisories) on the repository rather than opening a public issue. Include steps to
reproduce and the affected version/commit. We'll acknowledge and respond as soon as
we reasonably can.

Please do not include secrets or credentials in reports.
