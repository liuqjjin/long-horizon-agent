from __future__ import annotations

import hashlib
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest

import lha.bench.terminal_control as terminal_control
from lha.bench.terminal_control import (
    ControlRecordExists,
    ControlStoreError,
    RegisteredAttempt,
    SecureDirectory,
    command_digest,
    evaluation_lock,
    initialize_control_store,
    open_attempt_store,
    terminal_attempt_id,
    terminal_control_root,
    write_model_started,
)

EVALUATION_ID = "a" * 32
PROTOCOL_SHA256 = "b" * 64


def _attempts(
    evaluation_id: str = EVALUATION_ID,
) -> tuple[RegisteredAttempt, ...]:
    values = []
    cases: tuple[tuple[Literal["smoke", "scored"], int], ...] = (
        ("smoke", 3),
        ("scored", 20),
    )
    for run_kind, count in cases:
        for index in range(count):
            instance_id = f"terminal-bench/{run_kind}-{index:02d}"
            values.append(
                RegisteredAttempt(
                    attempt_id=terminal_attempt_id(
                        evaluation_id,
                        run_kind,
                        instance_id,
                    ),
                    run_kind=run_kind,
                    instance_id=instance_id,
                    command_sha256=command_digest(("harbor", "run", instance_id)),
                )
            )
    return tuple(values)


def _initialize(tmp_path: Path):
    output = (tmp_path / "jobs").resolve()
    registration = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )
    return output, registration


def _write_valid_smoke_lifecycle(
    output: Path,
    registration,
) -> None:
    from lha.bench.terminal_bench import HarborExecutionManifest

    smoke_ids = tuple(
        attempt.instance_id
        for attempt in registration.attempts
        if attempt.run_kind == "smoke"
    )
    assert len(smoke_ids) == 3
    digest = "c" * 64
    pinned = f"sha256:{digest}"
    manifest = HarborExecutionManifest(
        dataset_version=pinned,
        run_kind="smoke",
        protocol_sha256=registration.protocol_sha256,
        expected_instance_ids=smoke_ids,
        observed_instance_ids=smoke_ids,
        task_content_digests={item: pinned for item in smoke_ids},
        task_checksums={item: digest for item in smoke_ids},
        task_image_digests={item: pinned for item in smoke_ids},
        codex_events_sha256={item: digest for item in smoke_ids},
        container_image_ids={item: pinned for item in smoke_ids},
        command_started_sha256={item: digest for item in smoke_ids},
        command_envelope_sha256={item: digest for item in smoke_ids},
        terminal_record_sha256={item: digest for item in smoke_ids},
        job_config_sha256={item: digest for item in smoke_ids},
        job_lock_sha256={item: digest for item in smoke_ids},
        job_result_sha256={item: digest for item in smoke_ids},
        trial_result_sha256={item: digest for item in smoke_ids},
        official_status={item: "PASS" for item in smoke_ids},
        protocol_errors={item: None for item in smoke_ids},
        job_dirs=tuple(f"/jobs/{index}" for index, _item in enumerate(smoke_ids)),
    )
    control = terminal_control_root(output, registration.evaluation_id)
    with SecureDirectory(control) as store:
        manifest_sha256 = store.write_json_once("smoke_manifest.json", manifest)
        store.write_json_once(
            "smoke_seal.json",
            terminal_control.SmokeSeal(
                evaluation_id=registration.evaluation_id,
                protocol_sha256=registration.protocol_sha256,
                manifest_sha256=manifest_sha256,
                smoke_instance_ids=(smoke_ids[0], smoke_ids[1], smoke_ids[2]),
                terminal_record_sha256={item: digest for item in smoke_ids},
                sealed_at="2026-07-28T00:00:00+00:00",
            ),
        )


def test_control_root_is_private_sibling_and_attempts_are_fixed(tmp_path):
    output, registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)

    assert output.is_dir()
    assert control.parent == output.parent / ".lha-control"
    assert control not in output.parents and output not in control.parents
    assert control.stat().st_mode & 0o777 == 0o700
    assert len(registration.attempts) == 23
    assert all((control / item.attempt_id).is_dir() for item in registration.attempts)
    assert all(
        (control / item.attempt_id).stat().st_mode & 0o777 == 0o700
        for item in registration.attempts
    )

    resumed = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )
    assert resumed == registration


def test_initialization_recovers_empty_partial_control_store(tmp_path):
    output = (tmp_path / "jobs").resolve()
    output.mkdir(mode=0o700)
    control = terminal_control_root(output, EVALUATION_ID)
    control.parent.mkdir(mode=0o700)
    control.mkdir(mode=0o700)

    registration = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )

    assert (control / "registration.json").is_file()
    assert all((control / item.attempt_id).is_dir() for item in registration.attempts)


def test_initialization_recovers_interrupted_attempt_directory_creation(
    tmp_path, monkeypatch
):
    output = (tmp_path / "jobs").resolve()
    original = SecureDirectory.create_directory
    created = 0

    def interrupt_after_four(self, name):
        nonlocal created
        if len(name) == 64 and created == 4:
            raise RuntimeError("simulated controller crash")
        result = original(self, name)
        if len(name) == 64:
            created += 1
        return result

    monkeypatch.setattr(SecureDirectory, "create_directory", interrupt_after_four)
    with pytest.raises(RuntimeError, match="simulated controller crash"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )
    monkeypatch.setattr(SecureDirectory, "create_directory", original)

    control = terminal_control_root(output, EVALUATION_ID)
    assert not (control / "registration.json").exists()
    assert sum(path.is_dir() for path in control.iterdir()) == 4

    resumed = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )

    assert len(resumed.attempts) == 23
    assert all((control / item.attempt_id).is_dir() for item in resumed.attempts)


def test_initialization_refuses_deleted_committed_attempt_directory(tmp_path):
    output, registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    missing = registration.attempts[-1].attempt_id
    with open_attempt_store(
        output,
        EVALUATION_ID,
        registration.attempts[0].attempt_id,
    ) as store:
        store.write_once("COMMAND_STARTED.json", b"started")
    (control / missing).rmdir()

    with pytest.raises(ControlStoreError, match="unavailable"):
        open_attempt_store(
            output,
            EVALUATION_ID,
            registration.attempts[0].attempt_id,
        )
    with pytest.raises(ControlStoreError, match="unavailable"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )
    assert not (control / missing).exists()


def test_initialization_allows_registered_lifecycle_records(tmp_path):
    output, registration = _initialize(tmp_path)
    with evaluation_lock(output, EVALUATION_ID):
        pass
    _write_valid_smoke_lifecycle(output, registration)

    resumed = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )

    assert resumed == registration


def test_initialization_migrates_complete_legacy_topology_without_replacing_evidence(
    tmp_path,
):
    output, registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    attempt = control / registration.attempts[0].attempt_id
    output_evidence = output / "result.json"
    output_evidence.write_bytes(b'{"reward": 1}\n')
    with SecureDirectory(attempt) as store:
        store.write_once("terminal.json", b'{"outcome": "completed"}\n')
    _write_valid_smoke_lifecycle(output, registration)

    identities = {
        "output": output.stat().st_ino,
        "control": control.stat().st_ino,
        "attempt": attempt.stat().st_ino,
    }
    binding_name, commit_name, _lock_name = terminal_control._anchor_record_names(
        output
    )
    anchor = output.parent / terminal_control._INITIALIZATION_ANCHOR_DIRECTORY
    shutil.rmtree(anchor)

    resumed = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )

    assert resumed == registration
    assert output_evidence.read_bytes() == b'{"reward": 1}\n'
    assert (attempt / "terminal.json").read_bytes() == b'{"outcome": "completed"}\n'
    assert output.stat().st_ino == identities["output"]
    assert control.stat().st_ino == identities["control"]
    assert attempt.stat().st_ino == identities["attempt"]
    assert (anchor / binding_name).is_file()
    assert (anchor / commit_name).is_file()


def test_initialization_recovers_registration_before_external_commit(tmp_path):
    output, registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    registration_bytes = (control / "registration.json").read_bytes()
    _binding_name, commit_name, _lock_name = terminal_control._anchor_record_names(
        output
    )
    commit_path = (
        output.parent
        / terminal_control._INITIALIZATION_ANCHOR_DIRECTORY
        / commit_name
    )
    commit_path.unlink()

    resumed = initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )

    assert resumed == registration
    assert (control / "registration.json").read_bytes() == registration_bytes
    assert commit_path.is_file()


def test_consumer_requires_the_external_registration_commit(tmp_path):
    output, registration = _initialize(tmp_path)
    _binding_name, commit_name, _lock_name = terminal_control._anchor_record_names(
        output
    )
    commit_path = (
        output.parent
        / terminal_control._INITIALIZATION_ANCHOR_DIRECTORY
        / commit_name
    )
    commit_path.unlink()

    with pytest.raises(ControlStoreError, match="commit.json"):
        open_attempt_store(
            output,
            EVALUATION_ID,
            registration.attempts[0].attempt_id,
        )
    assert not commit_path.exists()


def test_initialization_never_recreates_deleted_committed_output(tmp_path):
    output, registration = _initialize(tmp_path)
    with open_attempt_store(
        output,
        EVALUATION_ID,
        registration.attempts[0].attempt_id,
    ) as store:
        store.write_once("COMMAND_STARTED.json", b"started")
    output.rmdir()

    for _ in range(2):
        with pytest.raises(ControlStoreError, match="output root was removed"):
            initialize_control_store(
                evaluation_id=EVALUATION_ID,
                protocol_sha256=PROTOCOL_SHA256,
                output_root=output,
                attempts=_attempts(),
            )
        assert not output.exists()


def test_initialization_never_recreates_output_for_legacy_registration(tmp_path):
    output, _registration = _initialize(tmp_path)
    binding_name, commit_name, _lock_name = terminal_control._anchor_record_names(
        output
    )
    anchor = output.parent / terminal_control._INITIALIZATION_ANCHOR_DIRECTORY
    (anchor / binding_name).unlink()
    (anchor / commit_name).unlink()
    output.rmdir()

    for _ in range(2):
        with pytest.raises(ControlStoreError, match="output root was removed"):
            initialize_control_store(
                evaluation_id=EVALUATION_ID,
                protocol_sha256=PROTOCOL_SHA256,
                output_root=output,
                attempts=_attempts(),
            )
        assert not output.exists()


def test_initialization_never_recreates_deleted_started_control_root(tmp_path):
    output, registration = _initialize(tmp_path)
    with open_attempt_store(
        output,
        EVALUATION_ID,
        registration.attempts[0].attempt_id,
    ) as store:
        store.write_once("COMMAND_STARTED.json", b"started")
    control = terminal_control_root(output, EVALUATION_ID)
    shutil.rmtree(control)

    with pytest.raises(ControlStoreError, match="control root was removed"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )
    assert not control.exists()


def test_initialization_refuses_unknown_top_level_record(tmp_path):
    output, _registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    with SecureDirectory(control) as store:
        store.write_once("unknown.json", b"unknown")

    with pytest.raises(ControlStoreError, match="unexpected records"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )


def test_initialization_is_serialized_across_concurrent_callers(tmp_path):
    output = (tmp_path / "jobs").resolve()

    def initialize():
        return initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(initialize)
        second = pool.submit(initialize)

    assert first.result() == second.result()


def test_output_root_can_belong_to_only_one_evaluation(tmp_path):
    output = (tmp_path / "jobs").resolve()
    initialize_control_store(
        evaluation_id=EVALUATION_ID,
        protocol_sha256=PROTOCOL_SHA256,
        output_root=output,
        attempts=_attempts(),
    )
    other_evaluation = "d" * 32

    with pytest.raises(ControlStoreError, match="another evaluation"):
        initialize_control_store(
            evaluation_id=other_evaluation,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(other_evaluation),
        )

    assert not terminal_control_root(output, other_evaluation).exists()


def test_concurrent_evaluations_cannot_share_one_output_root(tmp_path):
    output = (tmp_path / "jobs").resolve()
    other_evaluation = "d" * 32

    def initialize(evaluation_id):
        try:
            return initialize_control_store(
                evaluation_id=evaluation_id,
                protocol_sha256=PROTOCOL_SHA256,
                output_root=output,
                attempts=_attempts(evaluation_id),
            )
        except ControlStoreError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(initialize, (EVALUATION_ID, other_evaluation))
        )

    registrations = [
        item for item in results if not isinstance(item, ControlStoreError)
    ]
    errors = [item for item in results if isinstance(item, ControlStoreError)]
    assert len(registrations) == 1
    assert len(errors) == 1
    assert "another evaluation" in str(errors[0])


def test_initialization_rejects_an_active_evaluation(tmp_path):
    output, _registration = _initialize(tmp_path)

    with evaluation_lock(output, EVALUATION_ID):
        with pytest.raises(ControlStoreError, match="already active"):
            initialize_control_store(
                evaluation_id=EVALUATION_ID,
                protocol_sha256=PROTOCOL_SHA256,
                output_root=output,
                attempts=_attempts(),
            )


@pytest.mark.parametrize("replaced", ["output", "control", "attempt"])
def test_initialization_detects_directory_replacement_before_return(
    tmp_path, monkeypatch, replaced
):
    output, _registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    attempt = control / _attempts()[0].attempt_id
    original_verify = terminal_control._verify_directory_path
    replaced_once = False

    def replace_then_verify(store, path, *, label):
        nonlocal replaced_once
        if not replaced_once and (
            (replaced == "output" and path == output)
            or (replaced == "control" and path == control)
            or (replaced == "attempt" and path == attempt)
        ):
            replaced_once = True
            renamed = path.with_name(f"{path.name}-original")
            path.rename(renamed)
            path.mkdir(mode=0o700)
        return original_verify(store, path, label=label)

    monkeypatch.setattr(
        terminal_control,
        "_verify_directory_path",
        replace_then_verify,
    )

    with pytest.raises(ControlStoreError, match="replaced during initialization"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )


def test_immutable_marker_prevents_a_second_model_start(tmp_path):
    output, registration = _initialize(tmp_path)
    attempt = registration.attempts[0]
    with open_attempt_store(output, EVALUATION_ID, attempt.attempt_id) as store:
        marker = write_model_started(
            store,
            evaluation_id=EVALUATION_ID,
            attempt_id=attempt.attempt_id,
            protocol_sha256=PROTOCOL_SHA256,
            run_kind=attempt.run_kind,
            instance_id=attempt.instance_id,
            container_id="c" * 64,
        )
        assert marker.instance_id == attempt.instance_id
        assert store.has("MODEL_STARTED.json")
        with pytest.raises(ControlRecordExists, match="already exists"):
            write_model_started(
                store,
                evaluation_id=EVALUATION_ID,
                attempt_id=attempt.attempt_id,
                protocol_sha256=PROTOCOL_SHA256,
                run_kind=attempt.run_kind,
                instance_id=attempt.instance_id,
                container_id="c" * 64,
            )


def test_symlink_and_hardlink_records_never_touch_sentinel(tmp_path):
    store_path = tmp_path / "control"
    store_path.mkdir(mode=0o700)
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"unchanged")
    sentinel.chmod(0o600)

    with SecureDirectory(store_path) as store:
        (store_path / "events.jsonl").symlink_to(sentinel)
        with pytest.raises(ControlRecordExists):
            store.write_once("events.jsonl", b"forged")
        assert sentinel.read_bytes() == b"unchanged"

        os.link(sentinel, store_path / "hardlink.json")
        with pytest.raises(ControlStoreError, match="private regular file"):
            store.read("hardlink.json")
        assert sentinel.read_bytes() == b"unchanged"


def test_write_once_recovers_an_interrupted_pending_write(tmp_path, monkeypatch):
    store_path = tmp_path / "control"
    store_path.mkdir(mode=0o700)
    original_write = terminal_control.os.write
    interrupted = False

    def interrupt_first_write(descriptor, payload):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            original_write(descriptor, payload[: max(1, len(payload) // 2)])
            raise OSError("simulated interrupted write")
        return original_write(descriptor, payload)

    with SecureDirectory(store_path) as store:
        monkeypatch.setattr(terminal_control.os, "write", interrupt_first_write)
        with pytest.raises(ControlStoreError, match="could not be written"):
            store.write_once("record.json", b'{"complete": true}\n')
        monkeypatch.setattr(terminal_control.os, "write", original_write)

        assert not store.has("record.json")
        store.write_once("record.json", b'{"complete": true}\n')
        assert store.read("record.json") == b'{"complete": true}\n'
        assert not any(name.startswith("pending-") for name in os.listdir(store._fd))


def test_write_once_recovers_a_linked_but_unfinalized_record(tmp_path, monkeypatch):
    store_path = tmp_path / "control"
    store_path.mkdir(mode=0o700)
    original_unlink = terminal_control.os.unlink
    interrupted = False

    def interrupt_pending_unlink(path, *args, **kwargs):
        nonlocal interrupted
        if not interrupted and str(path).startswith("pending-"):
            interrupted = True
            raise OSError("simulated crash after publish")
        return original_unlink(path, *args, **kwargs)

    with SecureDirectory(store_path) as store:
        monkeypatch.setattr(terminal_control.os, "unlink", interrupt_pending_unlink)
        with pytest.raises(ControlStoreError, match="could not be finalized"):
            store.write_once("record.json", b'{"complete": true}\n')
        monkeypatch.setattr(terminal_control.os, "unlink", original_unlink)

        assert store.has("record.json")
        assert store.read("record.json") == b'{"complete": true}\n'
        assert not any(name.startswith("pending-") for name in os.listdir(store._fd))


@pytest.mark.parametrize(
    "second_payload",
    [b'{"writer": 1}\n', b'{"writer": 2}\n'],
)
def test_write_once_serializes_live_writers(tmp_path, monkeypatch, second_payload):
    store_path = tmp_path / "control"
    store_path.mkdir(mode=0o700)
    first_payload = b'{"writer": 1}\n'
    first_is_writing = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    original_write = terminal_control.os.write
    paused = False

    def pause_first_write(descriptor, payload):
        nonlocal paused
        if not paused:
            paused = True
            first_is_writing.set()
            assert release_first.wait(timeout=5)
        return original_write(descriptor, payload)

    def write(payload, *, second=False):
        if second:
            second_entered.set()
        try:
            with SecureDirectory(store_path) as store:
                return "ok", store.write_once("record.json", payload)
        except ControlRecordExists:
            return "exists", None

    monkeypatch.setattr(terminal_control.os, "write", pause_first_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(write, first_payload)
        assert first_is_writing.wait(timeout=5)
        second = pool.submit(write, second_payload, second=True)
        assert second_entered.wait(timeout=5)
        release_first.set()
        results = (first.result(timeout=5), second.result(timeout=5))

    final = (store_path / "record.json").read_bytes()
    successes = [digest for outcome, digest in results if outcome == "ok"]
    assert len(successes) == 1
    assert successes[0] == hashlib.sha256(final).hexdigest()
    assert final == first_payload


@pytest.mark.parametrize("record_kind", ["symlink", "directory", "hardlink"])
def test_initialization_rejects_unsafe_lifecycle_record_types(
    tmp_path,
    record_kind,
):
    output, _registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    manifest = control / "smoke_manifest.json"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"{}")
    sentinel.chmod(0o600)
    if record_kind == "symlink":
        manifest.symlink_to(sentinel)
    elif record_kind == "directory":
        manifest.mkdir(mode=0o700)
    else:
        os.link(sentinel, manifest)

    with pytest.raises(ControlStoreError, match="private regular file"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )


def test_initialization_rejects_invalid_lifecycle_content(tmp_path):
    output, registration = _initialize(tmp_path)
    control = terminal_control_root(output, EVALUATION_ID)
    with SecureDirectory(control) as store:
        store.write_once("smoke_manifest.json", b'{"schema_version": 5}\n')
        store.write_once("smoke_seal.json", b'{"schema_version": 2}\n')

    with pytest.raises(ControlStoreError, match="control record is invalid"):
        open_attempt_store(
            output,
            EVALUATION_ID,
            registration.attempts[0].attempt_id,
        )
    with pytest.raises(ControlStoreError, match="control record is invalid"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )


def test_open_directory_fd_survives_parent_path_replacement(tmp_path):
    original = tmp_path / "control"
    original.mkdir(mode=0o700)
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    sentinel = replacement / "record.json"
    sentinel.write_bytes(b"replacement")
    sentinel.chmod(0o600)

    store = SecureDirectory(original)
    renamed = tmp_path / "control-original"
    original.rename(renamed)
    original.symlink_to(replacement, target_is_directory=True)
    try:
        store.write_once("record.json", b"trusted")
    finally:
        store.close()

    assert sentinel.read_bytes() == b"replacement"
    assert (renamed / "record.json").read_bytes() == b"trusted"


def test_unsafe_control_directory_is_rejected(tmp_path):
    directory = tmp_path / "control"
    directory.mkdir(mode=0o755)
    with pytest.raises(ControlStoreError, match="private and owner-only"):
        SecureDirectory(directory)


def test_evaluation_lock_rejects_concurrent_controller(tmp_path):
    output, _registration = _initialize(tmp_path)
    with evaluation_lock(output, EVALUATION_ID) as descriptor:
        assert descriptor >= 0
        with pytest.raises(ControlStoreError, match="already active"):
            with evaluation_lock(output, EVALUATION_ID):
                pass
    with evaluation_lock(output, EVALUATION_ID) as descriptor:
        assert descriptor >= 0


def test_deleted_evaluation_lock_is_not_recreated_or_relocked(tmp_path):
    output, _registration = _initialize(tmp_path)
    lock_path = (
        terminal_control_root(output, EVALUATION_ID) / "evaluation.lock"
    )

    with evaluation_lock(output, EVALUATION_ID):
        lock_path.unlink()
        with pytest.raises(ControlStoreError, match="lock is missing"):
            with evaluation_lock(output, EVALUATION_ID):
                pass
        assert not lock_path.exists()

    with pytest.raises(ControlStoreError, match="lock was removed"):
        initialize_control_store(
            evaluation_id=EVALUATION_ID,
            protocol_sha256=PROTOCOL_SHA256,
            output_root=output,
            attempts=_attempts(),
        )
    assert not lock_path.exists()
