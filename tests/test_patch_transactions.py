"""Adversarial checks for the durable patch transaction boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from conftest import hermetic_task

from lha.artifacts import Patch
from lha.config import Config
from lha.harness import Harness
from lha.harness.approval import HumanApprovalGate
from lha.harness.checkpoint import append_ledger, load_state, run_lock
from lha.harness.errors import (
    BudgetExceeded,
    CheckpointCorrupt,
    RunLocked,
    TransactionCorrupt,
)
from lha.harness.loop import _claim_run_dir
from lha.harness.state import LLMUsageState, StepRecord
from lha.harness.transaction import (
    list_transactions,
    read_transaction_events,
    save_transaction,
    state_for_paths,
    transaction_log_path,
    transaction_path,
)
from lha.llm.stub import DeterministicStub
from lha.llm.trace import TracedLLM
from lha.tasks.spec import TaskSpec
from lha.tools import policy
from lha.tools.patch import (
    Backup,
    apply_patch,
    load_backup,
    make_unified_diff,
    resolve_patch,
    revert_patch,
    save_backup,
    snapshot_paths,
)


def _cfg(tmp_path: Path) -> Config:
    return Config(
        llm_backend="stub",
        code_backend="null",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "nodata",
    )


def _paused(tmp_path: Path):
    return Harness(_cfg(tmp_path)).run(
        hermetic_task("data/tasks/fix_average_approval.yaml")
    )


def test_resolved_patch_uses_the_executable_write_set_only():
    patch = Patch(
        step_id="s",
        file_contents={"src/app.py": "answer = 42\n"},
        touched_files=["tests/test_oracle.py"],
    )
    resolved = resolve_patch(patch)
    assert resolved.paths == ["src/app.py"]
    assert policy.check_resolved(resolved) == []


def test_patch_rejects_ambiguous_executable_payloads():
    with pytest.raises(ValueError, match="either unified_diff or file_contents"):
        Patch(
            step_id="s",
            unified_diff=(
                "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+claimed\n"
            ),
            file_contents={"app.py": "actually applied\n"},
        )


def test_resolved_diff_catches_a_protected_path_hidden_by_metadata():
    diff = make_unified_diff("assert False\n", "assert True\n", "tests/test_oracle.py")
    patch = Patch(step_id="s", unified_diff=diff, touched_files=["src/app.py"])
    resolved = resolve_patch(patch)
    assert resolved.paths == ["tests/test_oracle.py"]
    assert policy.check_resolved(resolved) == ["tests/test_oracle.py"]


def test_resolved_diff_decodes_git_quoted_paths():
    diff = (
        'diff --git "a/tests/test quoted.py" "b/tests/test quoted.py"\n'
        '--- "a/tests/test quoted.py"\n'
        '+++ "b/tests/test quoted.py"\n'
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    patch = Patch(step_id="s", unified_diff=diff, touched_files=["src/app.py"])
    resolved = resolve_patch(patch)
    assert resolved.paths == ["tests/test quoted.py"]
    assert policy.check_resolved(resolved) == ["tests/test quoted.py"]


def test_resolved_diff_decodes_git_octal_utf8_and_detects_aliases():
    diff = (
        'diff --git "a/src/\\303\\251.py" "b/src/\\303\\251.py"\n'
        '--- "a/src/\\303\\251.py"\n'
        '+++ "b/src/\\303\\251.py"\n'
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    resolved = resolve_patch(Patch(step_id="s", unified_diff=diff))
    assert resolved.paths == ["src/é.py"]


def test_resolved_diff_matches_git_apply_p1_for_custom_prefixes(tmp_path):
    workdir = tmp_path / "workdir"
    (workdir / "src").mkdir(parents=True)
    target = workdir / "src" / "app.py"
    target.write_text("old\n")
    diff = (
        "diff --git x/src/app.py y/src/app.py\n"
        "--- x/src/app.py\n"
        "+++ y/src/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    patch = Patch(step_id="s", unified_diff=diff)
    resolved = resolve_patch(patch)
    assert resolved.paths == ["src/app.py"]

    _paths, backup = apply_patch(patch, workdir, resolved=resolved)
    assert target.read_text() == "new\n"
    revert_patch(backup, workdir)
    assert target.read_text() == "old\n"


def test_symlink_diff_is_removed_when_apply_fails_closed(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret\n")
    diff = (
        "diff --git a/link b/link\n"
        "new file mode 120000\n"
        "--- /dev/null\n"
        "+++ b/link\n"
        "@@ -0,0 +1 @@\n"
        "+../outside\n"
        "\\ No newline at end of file\n"
    )
    patch = Patch(step_id="s", unified_diff=diff)
    with pytest.raises(ValueError, match="symlink|non-regular"):
        apply_patch(patch, workdir, resolved=resolve_patch(patch))
    assert not (workdir / "link").exists()
    assert not (workdir / "link").is_symlink()
    assert outside.read_text() == "secret\n"


def test_revert_restores_file_mode_and_removes_created_directories(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    script = workdir / "tool.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o644)
    mode_diff = (
        "diff --git a/tool.sh b/tool.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )
    mode_patch = Patch(step_id="mode", unified_diff=mode_diff)
    _paths, mode_backup = apply_patch(mode_patch, workdir)
    assert stat.S_IMODE(script.stat().st_mode) == 0o755
    revert_patch(mode_backup, workdir)
    assert stat.S_IMODE(script.stat().st_mode) == 0o644

    create_patch = Patch(
        step_id="create",
        file_contents={"new/nested/value.txt": "value\n"},
    )
    _paths, create_backup = apply_patch(create_patch, workdir)
    assert (workdir / "new" / "nested" / "value.txt").exists()
    revert_patch(create_backup, workdir)
    assert not (workdir / "new").exists()


def test_backup_rejects_a_checksummed_traversal_path(tmp_path):
    path = tmp_path / "backup.json"
    save_backup(
        Backup(originals={"src/app.py": b"old\n"}, modes={"src/app.py": 0o644}),
        path,
    )
    envelope = json.loads(path.read_text())
    envelope["originals_b64"] = {
        "../victim": base64.b64encode(b"old\n").decode("ascii")
    }
    envelope["modes"] = {"../victim": 0o644}
    payload = json.dumps(
        {
            "schema_version": 4,
            "originals_b64": envelope["originals_b64"],
            "modes": envelope["modes"],
            "created_dirs": envelope["created_dirs"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    envelope["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    path.write_text(json.dumps(envelope))

    with pytest.raises(ValueError, match="unsafe patch path"):
        load_backup(path, required=True)


@pytest.mark.parametrize(
    "original",
    [
        b"value = 1\r\n",
        b"\xef\xbb\xbfvalue = 1\r\n",
    ],
)
def test_rollback_restores_original_bytes_and_mode(tmp_path, original):
    target = tmp_path / "module.py"
    target.write_bytes(original)
    target.chmod(0o640)
    patch = Patch(
        step_id="s",
        file_contents={"module.py": "value = 2\n"},
        touched_files=["module.py"],
    )
    resolved = resolve_patch(patch)
    backup = snapshot_paths(resolved.paths, tmp_path)

    apply_patch(patch, tmp_path, resolved=resolved, backup=backup)
    revert_patch(backup, tmp_path)

    assert target.read_bytes() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_failed_unified_diff_leaves_non_utf8_target_byte_exact(tmp_path):
    target = tmp_path / "binary.py"
    original = b"\xff\xfe\x00unchanged\r\n"
    target.write_bytes(original)
    patch = Patch(
        step_id="s",
        unified_diff=(
            "--- a/binary.py\n"
            "+++ b/binary.py\n"
            "@@ -1 +1 @@\n"
            "-different\n"
            "+changed\n"
        ),
        touched_files=["binary.py"],
    )

    with pytest.raises(RuntimeError, match="git apply failed"):
        apply_patch(patch, tmp_path)
    assert target.read_bytes() == original


def test_transaction_state_detects_permission_only_drift(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("value = 1\n")
    target.chmod(0o644)
    before = state_for_paths(tmp_path, ["module.py"])
    target.chmod(0o755)
    assert state_for_paths(tmp_path, ["module.py"]) != before


@pytest.mark.parametrize("directory_link", [False, True])
def test_prepare_workdir_never_follows_repository_symlinks(
    tmp_path, directory_link
):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / ("outside-dir" if directory_link else "secret.txt")
    if directory_link:
        outside.mkdir()
        (outside / "secret.txt").write_text("HOST_SECRET")
    else:
        outside.write_text("HOST_SECRET")
    (source / "linked").symlink_to(outside, target_is_directory=directory_link)
    task = TaskSpec(
        kind="issue_to_pr",
        title="untrusted repository",
        target_repo=str(source),
        context_requirement="optional",
    )
    workdir = tmp_path / "run" / "workdir"

    with pytest.raises(ValueError, match="symbolic link"):
        Harness(_cfg(tmp_path))._prepare_workdir(task, workdir)
    assert not workdir.exists()


@pytest.mark.parametrize(
    "paths",
    [
        {"../escape.py": "x"},
        {"/tmp/escape.py": "x"},
        {"A.py": "x", "a.py": "y"},
        {"a\\..\\escape.py": "x"},
    ],
)
def test_resolved_patch_rejects_traversal_and_aliases(paths):
    with pytest.raises(ValueError):
        resolve_patch(Patch(step_id="s", file_contents=paths))


def test_prepared_crash_recovery_reapplies_the_same_patch(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    tx = list_transactions(run_dir, "s2-fix")[0]
    assert tx.status == "APPLIED"

    # Crash window: files changed, but neither the APPLIED state nor its event
    # reached disk. Recreate the last genuinely durable PREPARED evidence; an
    # APPLIED -> PREPARED journal transition would be impossible in production.
    events = read_transaction_events(run_dir, tx.step_id, tx.attempt_id)
    prepared = tx.model_copy(
        update={
            "status": "PREPARED",
            "applied_state": {},
            "updated_at": events[0].at,
        }
    )
    payload = prepared.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    assert digest == events[0].transaction_sha256
    transaction_path(run_dir, tx.step_id, tx.attempt_id).write_text(
        json.dumps({"schema_version": 1, "sha256": digest, "payload": payload})
    )
    log_path = transaction_log_path(run_dir, tx.step_id, tx.attempt_id)
    log_path.write_text(log_path.read_text().splitlines()[0] + "\n")
    HumanApprovalGate(run_dir).resolve(approved=True)
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)

    assert done.status == "DONE"
    assert "len(values) - 1" not in (run_dir / "workdir" / "mathutils.py").read_text()
    completed = [
        json.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text().splitlines()
        if json.loads(line).get("phase") == "complete"
        and json.loads(line).get("step_id") == "s2-fix"
    ]
    assert len(completed) == 1


def test_corrupt_primary_backup_restores_from_mirror_and_fails(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    tx = list_transactions(run_dir, "s2-fix")[0]
    (run_dir / tx.backup_ref).write_text("{broken")
    HumanApprovalGate(run_dir).resolve(approved=True)

    failed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert failed.status == "FAILED"
    assert "primary backup is corrupt" in failed.message
    assert "len(values) - 1" in (run_dir / "workdir" / "mathutils.py").read_text()
    assert list_transactions(run_dir, "s2-fix")[0].status == "REVERTED"


def test_worktree_drift_before_approval_fails_and_reverts(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    target = run_dir / "workdir" / "mathutils.py"
    target.write_text("def average(values):\n    return 999\n")
    HumanApprovalGate(run_dir).resolve(approved=True)

    failed = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert failed.status == "FAILED"
    assert "worktree drift" in failed.message
    assert "len(values) - 1" in target.read_text()


def test_run_lock_rejects_a_second_resume(tmp_path):
    paused = _paused(tmp_path)
    with run_lock(paused.state.run_dir):
        with pytest.raises(RunLocked):
            Harness(_cfg(tmp_path)).resume(paused.state.run_id)


def test_run_directory_claim_is_atomic_between_concurrent_creators(tmp_path):
    barrier = threading.Barrier(2)

    def claim() -> str:
        barrier.wait()
        try:
            _claim_run_dir(tmp_path / "runs", "same-run")
        except FileExistsError:
            return "exists"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim) for _ in range(2)]
        outcomes = sorted(future.result() for future in futures)
    assert outcomes == ["created", "exists"]


def test_duplicate_run_id_never_overwrites_existing_bytes(tmp_path):
    task = hermetic_task("data/tasks/fix_average.yaml")
    harness = Harness(_cfg(tmp_path))
    first = harness.run(task, run_id="fixed-run")
    run_dir = Path(first.state.run_dir)
    before = {
        "state": (run_dir / "state.json").read_bytes(),
        "source": (run_dir / "workdir" / "mathutils.py").read_bytes(),
    }

    with pytest.raises(FileExistsError, match="already exists"):
        Harness(_cfg(tmp_path)).run(task, run_id="fixed-run")
    assert (run_dir / "state.json").read_bytes() == before["state"]
    assert (run_dir / "workdir" / "mathutils.py").read_bytes() == before["source"]


def test_run_id_cannot_escape_runs_directory(tmp_path):
    with pytest.raises(ValueError, match="invalid run id"):
        _claim_run_dir(tmp_path / "runs", "../escape")
    assert not (tmp_path / "escape").exists()


def test_schema_one_run_can_be_inspected_but_not_resumed(tmp_path):
    paused = _paused(tmp_path)
    path = Path(paused.state.run_dir) / "state.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["schema_version"] = 1
    canonical = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":"))
    envelope["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointCorrupt, match="schema 2 is required"):
        Harness(_cfg(tmp_path)).resume(paused.state.run_id)


def test_pre_envelope_run_is_not_silently_upgraded_on_load(tmp_path):
    paused = _paused(tmp_path)
    path = Path(paused.state.run_dir) / "state.json"
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    payload.pop("schema_version")
    path.write_text(json.dumps(payload))

    inspected = load_state(paused.state.run_dir)
    assert inspected.schema_version == 1
    with pytest.raises(CheckpointCorrupt, match="schema 2 is required"):
        Harness(_cfg(tmp_path)).resume(paused.state.run_id)


def test_resume_refuses_a_corrupt_middle_ledger_record(tmp_path):
    paused = _paused(tmp_path)
    ledger = Path(paused.state.run_dir) / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    assert len(lines) >= 3
    lines[1] = "{broken"
    ledger.write_text("\n".join(lines) + "\n")
    HumanApprovalGate(paused.state.run_dir).resolve(approved=True)

    with pytest.raises(CheckpointCorrupt, match="ledger"):
        Harness(_cfg(tmp_path)).resume(paused.state.run_id)


def test_resume_refuses_corrupt_transaction_evidence(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    (
        run_dir
        / "transactions"
        / "s2-fix"
        / f"{transaction.attempt_id}.json"
    ).write_text("{broken")
    HumanApprovalGate(run_dir).resolve(approved=True)

    with pytest.raises(TransactionCorrupt, match="recovery evidence"):
        Harness(_cfg(tmp_path)).resume(paused.state.run_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backup_ref", "/tmp/outside.json"),
        ("backup_ref", "../outside.json"),
        ("backup_mirror_ref", "../../outside.json"),
    ],
)
def test_transaction_rejects_external_backup_references(tmp_path, field, value):
    paused = _paused(tmp_path)
    transaction = list_transactions(Path(paused.state.run_dir), "s2-fix")[0]
    payload = transaction.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValueError, match="backup"):
        type(transaction).model_validate(payload)


def test_model_usage_budget_survives_approval_resume(tmp_path):
    config = _cfg(tmp_path).model_copy(update={"max_llm_calls": 1})
    paused = Harness(config).run(hermetic_task("data/tasks/fix_average_approval.yaml"))
    assert paused.status == "AWAITING_APPROVAL"
    assert paused.state.llm_usage.calls == 1

    HumanApprovalGate(paused.state.run_dir).resolve(approved=True)
    done = Harness(config).resume(paused.state.run_id)
    assert done.status == "DONE"
    assert done.state.llm_usage.calls == 1


def test_model_call_write_ahead_checkpoint_prevents_budget_reset(tmp_path):
    first = TracedLLM(DeterministicStub(), max_calls=1).bind(tmp_path)
    first.restore_totals(LLMUsageState())
    first.complete("system", "prompt")

    # Model a hard crash before RunState could copy the in-memory total.
    resumed = TracedLLM(DeterministicStub(), max_calls=1).bind(tmp_path)
    resumed.restore_totals(LLMUsageState())
    assert resumed.totals.calls == 1
    with pytest.raises(BudgetExceeded, match="max_llm_calls=1"):
        resumed.complete("system", "prompt")


def test_model_call_is_reserved_before_a_hard_process_exit(tmp_path):
    script = """
import os
import sys

from lha.llm.base import LLMClient
from lha.llm.trace import TracedLLM


class CrashDuringCall(LLMClient):
    def complete(self, system: str, prompt: str) -> str:
        os._exit(17)


TracedLLM(CrashDuringCall(), max_calls=1).bind(sys.argv[1]).complete("s", "p")
"""
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert child.returncode == 17

    resumed = TracedLLM(DeterministicStub(), max_calls=1).bind(tmp_path)
    resumed.restore_totals(LLMUsageState())
    assert resumed.totals.calls == 1
    with pytest.raises(BudgetExceeded, match="max_llm_calls=1"):
        resumed.complete("system", "prompt")


def test_crash_replay_does_not_consume_a_second_step_budget_unit(
    tmp_path, monkeypatch
):
    from lha.agents.context_engineer import ContextEngineer

    original = ContextEngineer.gather
    crashed = False

    def interrupt_once(self, step, workdir=None):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise KeyboardInterrupt
        return original(self, step, workdir)

    monkeypatch.setattr(ContextEngineer, "gather", interrupt_once)
    task = hermetic_task("data/tasks/fix_average.yaml")
    with pytest.raises(KeyboardInterrupt):
        Harness(_cfg(tmp_path)).run(task, run_id="budget-crash")

    checkpoint = load_state(tmp_path / "runs" / "budget-crash")
    assert checkpoint.steps_used == 1
    assert checkpoint.budgeted_attempts == ["s1-context-r0"]

    resumed = Harness(_cfg(tmp_path)).resume("budget-crash")
    assert resumed.status == "DONE"
    assert resumed.state.steps_used == 2
    assert resumed.state.budgeted_attempts == ["s1-context-r0", "s2-fix-r0"]


def test_corrupt_model_usage_checkpoint_fails_closed(tmp_path):
    (tmp_path / "llm_usage.json").write_text("{broken")
    traced = TracedLLM(DeterministicStub()).bind(tmp_path)
    with pytest.raises(CheckpointCorrupt, match="LLM usage checkpoint"):
        traced.restore_totals(LLMUsageState())


def test_ledger_idempotency_key_prevents_duplicate_complete(tmp_path):
    state = _paused(tmp_path).state
    record = StepRecord(
        seq=state.next_seq(),
        step_id="s2-fix",
        phase="complete",
        attempt_id="s2-fix-r0",
        idempotency_key="s2-fix-r0:complete",
    )
    append_ledger(state, record)
    append_ledger(state, record.model_copy(update={"event_id": "different"}))
    matches = [
        json.loads(line)
        for line in (Path(state.run_dir) / "ledger.jsonl").read_text().splitlines()
        if json.loads(line).get("idempotency_key") == "s2-fix-r0:complete"
    ]
    assert len(matches) == 1


def test_ledger_idempotency_key_cannot_name_a_different_event(tmp_path):
    state = _paused(tmp_path).state
    append_ledger(
        state,
        StepRecord(
            seq=state.next_seq(),
            step_id="s2-fix",
            phase="complete",
            attempt_id="s2-fix-r0",
            idempotency_key="fixed-key",
        ),
    )
    with pytest.raises(CheckpointCorrupt, match="reused for a different event"):
        append_ledger(
            state,
            StepRecord(
                seq=state.next_seq(),
                step_id="s2-fix",
                phase="fail",
                attempt_id="s2-fix-r0",
                idempotency_key="fixed-key",
            ),
        )


@pytest.mark.parametrize("field", ["run_dir", "workdir"])
def test_resume_binds_checkpoint_paths_to_the_requested_run(tmp_path, field):
    paused = _paused(tmp_path)
    state_path = Path(paused.state.run_dir) / "state.json"
    envelope = json.loads(state_path.read_text())
    envelope["payload"][field] = str(tmp_path / "outside")
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":")
    ).encode()
    envelope["sha256"] = hashlib.sha256(canonical).hexdigest()
    state_path.write_text(json.dumps(envelope))

    with pytest.raises(CheckpointCorrupt, match=field):
        Harness(_cfg(tmp_path)).resume(paused.state.run_id)


def test_resume_refuses_a_symlink_ledger_without_touching_its_target(tmp_path):
    paused = _paused(tmp_path)
    ledger = Path(paused.state.run_dir) / "ledger.jsonl"
    victim = tmp_path / "victim.jsonl"
    victim.write_text('{"secret": true}\n')
    ledger.unlink()
    ledger.symlink_to(victim)

    with pytest.raises(CheckpointCorrupt, match="ledger path is unsafe"):
        Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert victim.read_text() == '{"secret": true}\n'


def test_run_lock_refuses_a_symlink_without_overwriting_its_target(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep\n")
    (run_dir / ".run.lock").symlink_to(victim)

    with pytest.raises(CheckpointCorrupt, match="run lock path is unsafe"):
        with run_lock(run_dir):
            pass
    assert victim.read_text() == "keep\n"


def test_repeated_pause_keeps_one_request_and_one_later_decision_event(tmp_path):
    paused = _paused(tmp_path)
    again = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert again.status == "AWAITING_APPROVAL"
    run_dir = Path(paused.state.run_dir)
    requests = list(
        (run_dir / "steps" / "s2-fix" / "attempts").rglob(
            "approval_request.json"
        )
    )
    assert len(requests) == 1
    records = [
        json.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text().splitlines()
    ]
    approvals = [
        record
        for record in records
        if record.get("step_id") == "s2-fix" and record.get("phase") == "approval"
    ]
    assert approvals == []  # a pending request is not a reviewer decision

    HumanApprovalGate(run_dir).resolve(approved=True, note="ok")
    done = Harness(_cfg(tmp_path)).resume(paused.state.run_id)
    assert done.status == "DONE"
    records = [
        json.loads(line)
        for line in (run_dir / "ledger.jsonl").read_text().splitlines()
    ]
    approvals = [
        record
        for record in records
        if record.get("step_id") == "s2-fix" and record.get("phase") == "approval"
    ]
    assert len(approvals) == 1
    assert approvals[0]["idempotency_key"] == "s2-fix-r0:approval"


def test_transaction_writes_one_checksummed_event_per_phase(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    tx = list_transactions(run_dir, "s2-fix")[0]
    events = read_transaction_events(run_dir, tx.step_id, tx.attempt_id)
    assert [event.status for event in events] == ["PREPARED", "APPLIED"]

    # Saving the same durable state is idempotent and does not manufacture an
    # extra phase event.
    save_transaction(run_dir, tx)
    statuses = [
        event.status
        for event in read_transaction_events(run_dir, tx.step_id, tx.attempt_id)
    ]
    assert statuses == [
        "PREPARED",
        "APPLIED",
    ]


def test_transaction_log_corruption_fails_closed(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    tx = list_transactions(run_dir, "s2-fix")[0]
    path = transaction_log_path(run_dir, tx.step_id, tx.attempt_id)
    lines = path.read_text().splitlines()
    lines[0] = '{"payload": {}, "sha256": "bad"}'
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(TransactionCorrupt, match="invalid transaction log"):
        list_transactions(run_dir, tx.step_id)


def test_transaction_log_cannot_drop_its_prepared_prefix(tmp_path):
    paused = _paused(tmp_path)
    run_dir = Path(paused.state.run_dir)
    transaction = list_transactions(run_dir, "s2-fix")[0]
    path = transaction_log_path(
        run_dir, transaction.step_id, transaction.attempt_id
    )
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    path.write_text(lines[-1] + "\n")

    with pytest.raises(TransactionCorrupt, match="start at PREPARED"):
        read_transaction_events(
            run_dir, transaction.step_id, transaction.attempt_id
        )
