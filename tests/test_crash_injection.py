"""Crash injection: die mid-step, then resume from the checkpoint.

A crash (KeyboardInterrupt here, standing in for SIGKILL/power loss) must leave
a loadable checkpoint; a resume must finish the run without duplicating side
effects; a corrupt checkpoint or ledger must refuse to resume rather than guess.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.agents import ContextEngineer, Implementer, VerifierAgent
from lha.config import Config
from lha.harness import Harness
from lha.harness.checkpoint import load_state, read_ledger
from lha.harness.errors import CheckpointCorrupt


def _cfg(tmp_path, **kw) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
        **kw,
    )


def _crash_once(monkeypatch, cls, method: str) -> None:
    """Raise KeyboardInterrupt on the first call, behave normally afterwards."""
    real = getattr(cls, method)
    calls = {"n": 0}

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cls, method, wrapper)


@pytest.mark.parametrize(
    ("cls", "method"),
    [
        (ContextEngineer, "gather"),
        (Implementer, "implement"),
        (VerifierAgent, "verify"),
    ],
    ids=["during-context", "during-execute", "during-verify"],
)
def test_crash_mid_step_then_resume(tmp_path, monkeypatch, cls, method):
    task = hermetic_task("data/tasks/fix_average.yaml")
    _crash_once(monkeypatch, cls, method)

    with pytest.raises(KeyboardInterrupt):
        Harness(_cfg(tmp_path)).run(task, run_id="crashed")

    # The checkpoint survives the crash: loadable, checksum-valid, not terminal.
    state = load_state(tmp_path / "runs" / "crashed")
    assert not state.is_terminal()

    monkeypatch.undo()  # the "new process" does not crash
    r2 = Harness(_cfg(tmp_path)).resume("crashed")
    assert r2.status == "DONE"
    fixed = (Path(r2.state.workdir) / "mathutils.py").read_text()
    assert "len(values) - 1" not in fixed

    # No duplicated side effects: each step completed exactly once, and every
    # ledger event is unique. (The verify-phase crash also exercises
    # revert-then-reapply of an already-applied patch on resume.)
    records = read_ledger(r2.state.run_dir)
    completes = Counter(r.step_id for r in records if r.phase == "complete")
    assert completes and all(n == 1 for n in completes.values())
    event_ids = [r.event_id for r in records]
    assert len(event_ids) == len(set(event_ids))


def test_resume_after_done_is_a_no_op(tmp_path):
    task = hermetic_task("data/tasks/fix_average.yaml")
    r1 = Harness(_cfg(tmp_path, use_skill_memory=True)).run(task, run_id="done")
    assert r1.status == "DONE"

    skills = sorted((tmp_path / "nodata" / "skills").glob("*.md"))
    assert len(skills) == 1  # the verified success was recorded once
    before = (skills[0].read_text(), len(read_ledger(r1.state.run_dir)))

    r2 = Harness(_cfg(tmp_path, use_skill_memory=True)).resume("done")
    assert r2.status == "DONE"
    assert "terminal" in r2.message
    after_skills = sorted((tmp_path / "nodata" / "skills").glob("*.md"))
    assert [p.name for p in after_skills] == [skills[0].name]
    assert (after_skills[0].read_text(), len(read_ledger(r1.state.run_dir))) == before


# --- corrupt checkpoints refuse to resume -----------------------------------
def _paused_run(tmp_path) -> Path:
    task = hermetic_task("data/tasks/fix_average.yaml")
    r1 = Harness(_cfg(tmp_path, max_steps=1)).run(task, run_id="paused")
    assert r1.status == "PAUSED"
    return tmp_path / "runs" / "paused"


def test_tampered_state_refuses_resume(tmp_path):
    run_dir = _paused_run(tmp_path)
    path = run_dir / "state.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["status"] = "DONE"  # edit without recomputing the checksum
    path.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointCorrupt, match="integrity"):
        Harness(_cfg(tmp_path)).resume("paused")


def test_truncated_state_refuses_resume(tmp_path):
    run_dir = _paused_run(tmp_path)
    path = run_dir / "state.json"
    path.write_text(path.read_text()[: len(path.read_text()) // 2])

    with pytest.raises(CheckpointCorrupt, match="unreadable"):
        Harness(_cfg(tmp_path)).resume("paused")


def test_inconsistent_cursor_refuses_resume(tmp_path):
    # Even with a recomputed (valid) checksum, a cursor outside the plan is
    # damage — semantic validation refuses it.
    run_dir = _paused_run(tmp_path)
    path = run_dir / "state.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["cursor"] = 99
    canonical = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":"))
    envelope["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointCorrupt, match="cursor"):
        Harness(_cfg(tmp_path)).resume("paused")


# --- ledger: torn tail is a crash artifact, mid-file damage is not ----------
def test_ledger_torn_tail_is_dropped(tmp_path):
    run_dir = _paused_run(tmp_path)
    n_before = len(read_ledger(run_dir))
    with open(run_dir / "ledger.jsonl", "a") as f:
        f.write('{"seq": 99, "step_')  # the crash cut this append short

    records = read_ledger(run_dir)  # tolerated, not raised
    assert len(records) == n_before
    assert all(r.seq != 99 for r in records)


def test_ledger_mid_file_corruption_raises(tmp_path):
    run_dir = _paused_run(tmp_path)
    ledger = run_dir / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    assert len(lines) >= 2
    lines[0] = '{"not a record"'
    ledger.write_text("\n".join(lines) + "\n")

    with pytest.raises(CheckpointCorrupt, match="corrupt"):
        read_ledger(run_dir)


def test_complete_corrupt_final_ledger_line_is_damage_not_a_torn_tail(tmp_path):
    run_dir = _paused_run(tmp_path)
    with open(run_dir / "ledger.jsonl", "a") as stream:
        stream.write("{bad}\n")

    with pytest.raises(CheckpointCorrupt, match="corrupt"):
        read_ledger(run_dir)
