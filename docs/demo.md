# Recording the demo GIF

A short terminal GIF for the README. This file is the script to record it.

> **NOTE (for a human):** recording needs an interactive TTY and screen-recording
> tools, so it must be run on a workstation — not by the headless autonomous
> maintainer. Follow the steps below, then drop the result at
> `docs/demo.gif` and it will appear in the README via the embed at the bottom.

## What the demo shows

1. The **durable human-approval gate** on the LangGraph runtime:
   `run → AWAITING_APPROVAL → approve → resume → DONE` (the run is checkpointed to
   `graph.sqlite` between the two processes).
2. The harness **self-evaluating** with `lha eval` → `5/5`.

Both are objective: the fix is accepted only after a real `pytest` passes.

## Prerequisites

```bash
# the harness
uv sync

# recording tools (macOS shown; Linux: use your package manager)
brew install asciinema agg          # agg converts .cast -> .gif
```

## Record

```bash
# clean slate so run_ids are fresh
rm -rf runs

asciinema rec docs/demo.cast --overwrite --cols 100 --rows 30 -c bash
# --- inside the recording session, run: ---

# 1) start a run whose edit step needs approval -> pauses, durably checkpointed
uv run lha run --runtime langgraph data/tasks/fix_average_approval.yaml
#    note the printed run_id, e.g. 20260101-120000-fix-average-...-abcd
RID=$(ls -t runs | head -1)

# 2) a human approves the pending edit
uv run lha approve "$RID"

# 3) resume: LangGraph replays from the SqliteSaver checkpoint, verifies, finishes
uv run lha resume --runtime langgraph "$RID"

# 4) the harness grades itself across all five workflows
uv run lha eval

exit   # ends the asciinema recording
```

## Convert to GIF and embed

```bash
agg docs/demo.cast docs/demo.gif --cols 100 --rows 30
rm docs/demo.cast              # keep the repo light; the GIF is the artifact
```

It renders in this doc once `docs/demo.gif` exists (below); add the same line to
the README with the path `docs/demo.gif`:

Once `docs/demo.gif` exists, uncomment the line below (and add the same one to the
README) so it renders here:

<!-- ![demo: approval-gated run, resume, and lha eval 5/5](demo.gif) -->


Tips for a clean recording: keep the window ~100×30, run each command and let its
output settle before the next, and trim dead time afterward with
`asciinema`'s editing or by re-recording. Aim for under ~30 seconds.
