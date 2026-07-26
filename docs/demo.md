# 60–90 second terminal demo

This is a recording script, not a claim that a video has already been produced.
Run it on a workstation with a warm `uv` environment, review the output, and
publish media only after every command succeeds.

The recording has two parts:

1. a real approval-gated CLI run that pauses in one process and resumes in
   another;
2. one fixed 10-step long-task integration test whose persisted trace contains
   an objectively rejected first patch, one repair, two approvals, a simulated
   process exit at a safe boundary, and successful recovery.

The reference patch used by the integration test is test-only oracle data; the
production harness does not read it.

## Prerequisites

```bash
uv sync

# macOS; use equivalent packages on Linux
brew install asciinema agg
```

Open a 100×32 terminal. Keep the environment and model cache warm before
recording so installation and downloads are not part of the clip.

## Start recording

```bash
asciinema rec docs/demo.cast \
  --overwrite \
  --cols 100 \
  --rows 32 \
  -c 'env PS1="$ " zsh -f'
```

Run the following commands inside that shell.

## Part 1 — approval survives a process boundary

Use a unique output directory so the demo does not delete or mix with existing
runs:

```bash
DEMO_ROOT="${TMPDIR:-/tmp}/lha-demo-$(date +%Y%m%d-%H%M%S)-$$"
DEMO_START_LOG="$DEMO_ROOT/start.log"
mkdir -p "$DEMO_ROOT"
export LHA_RUNS_DIR="$DEMO_ROOT/manual"

uv run lha run \
  --runtime langgraph \
  data/tasks/fix_average_approval.yaml | tee "$DEMO_START_LOG"

DEMO_RUN_ID="$(awk '$1 == "run_id" {print $3}' "$DEMO_START_LOG")"
uv run lha runs show "$DEMO_RUN_ID"
uv run lha approve "$DEMO_RUN_ID" --note "reviewed the persisted patch"
uv run lha resume --runtime langgraph "$DEMO_RUN_ID"
uv run lha trace "$DEMO_RUN_ID"
```

The first command must show `AWAITING_APPROVAL`; the resume must show `DONE` only
after executable checks pass. Stop the recording if either expectation is not
met.

## Part 2 — rejected patch, repair, interruption, and 10-step resume

Run one parameterized integration case in a retained, unique pytest directory:

```bash
DEMO_PYTEST_ROOT="$DEMO_ROOT/long-task"

uv run pytest \
  'tests/test_long_task_harness.py::test_all_long_tasks_approval_and_safe_crash_match_an_uninterrupted_run[config_parser]' \
  --basetemp "$DEMO_PYTEST_ROOT" \
  -q \
  -rA

DEMO_LONG_RUN="$(find "$DEMO_PYTEST_ROOT" \
  -type d \
  -name 'config_parser-interrupted' \
  -print \
  -quit)"
DEMO_LONG_RUNS="$(dirname "$DEMO_LONG_RUN")"

LHA_RUNS_DIR="$DEMO_LONG_RUNS" \
  uv run lha runs show config_parser-interrupted
LHA_RUNS_DIR="$DEMO_LONG_RUNS" \
  uv run lha trace config_parser-interrupted
LHA_RUNS_DIR="$DEMO_LONG_RUNS" \
  uv run lha trace config_parser-interrupted --html
```

The test itself is the check. It asserts:

- the initial empty patch reaches approval and is rejected by the targeted
  repository check;
- the reference repair reaches a second approval and passes;
- the process exits after step 6 is durably verified and before step 7 begins;
- a fresh `Harness` resumes and completes all 10 steps;
- repaired and uninterrupted worktrees have the same terminal digest;
- approval, repair, and completion idempotency keys are not duplicated.

The final `trace` makes the persisted step and repair history visible. The HTML
path printed by the last command can be opened after recording; do not spend
screen time launching a browser in the terminal clip.

## Finish and render

```bash
exit
agg docs/demo.cast docs/demo.gif --cols 100 --rows 32
```

Review the GIF before adding an embed. Check that:

- no username, credential path, token, unrelated terminal history, or private
  repository path is visible;
- the clip is between 60 and 90 seconds after trimming idle time;
- the `AWAITING_APPROVAL`, `DONE`, pytest pass, repair event, and 10 completed
  steps remain readable;
- no result is described before its command appears.

Do not commit `demo.gif` merely because the script exists. The media is a release
artifact only after the recording has been made and reviewed.
