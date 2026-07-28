from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_terminal_bench_2_1.py"
SPEC = importlib.util.spec_from_file_location("lha_terminal_runner_tool", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_terminal_bench_runner_exposes_run_and_validate_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0
    assert "run" in completed.stdout
    assert "validate" in completed.stdout


def test_terminal_bench_runner_validates_committed_evidence() -> None:
    package = ROOT / "benchmarks/terminal_bench_2_1"
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "validate", str(package)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["denominator"] == 20
    assert result["passed"] + result["failed"] + result["errors"] == 20


def test_stage_once_is_idempotent_for_the_same_regular_input(tmp_path) -> None:
    source = tmp_path / "source.whl"
    target = tmp_path / "stage" / "source.whl"
    source.write_bytes(b"wheel-bytes")

    first = runner._stage_once(source, target, mode=0o644)
    second = runner._stage_once(source, target, mode=0o644)

    assert first == target
    assert second == target
    assert target.read_bytes() == b"wheel-bytes"
    assert target.stat().st_nlink == 1
    assert target.stat().st_mode & 0o777 == 0o644
    assert not list(target.parent.glob(f".{target.name}.*"))


@pytest.mark.parametrize(
    ("link_kind", "message"),
    [
        ("symlink", "symlink components"),
        ("hardlink", "regular file with one link"),
    ],
)
def test_stage_once_rejects_linked_sources(tmp_path, link_kind, message) -> None:
    original = tmp_path / "original"
    source = tmp_path / "source"
    original.write_bytes(b"input")
    if link_kind == "symlink":
        source.symlink_to(original)
    else:
        os.link(original, source)

    with pytest.raises(ValueError, match=message):
        runner._stage_once(source, tmp_path / "stage" / "input", mode=0o644)


def test_stage_once_rejects_a_symlinked_source_parent(tmp_path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "input").write_bytes(b"input")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink components"):
        runner._stage_once(
            linked_parent / "input",
            tmp_path / "stage" / "input",
            mode=0o644,
        )


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_stage_once_rejects_linked_existing_targets(tmp_path, link_kind) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"input")
    target = tmp_path / "stage" / "input"
    target.parent.mkdir()
    other = tmp_path / "other"
    other.write_bytes(b"input")
    if link_kind == "symlink":
        target.symlink_to(other)
    else:
        os.link(other, target)

    with pytest.raises(ValueError, match="regular file with one link"):
        runner._stage_once(source, target, mode=0o644)


def test_stage_once_detects_source_change_between_digest_and_copy(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"aaaa")
    target = tmp_path / "stage" / "input"
    original_digest = runner._digest_descriptor
    calls = 0

    def digest_then_change(descriptor, expected, *, label):
        nonlocal calls
        value = original_digest(descriptor, expected, label=label)
        calls += 1
        if calls == 1:
            source.write_bytes(b"bbbb")
        return value

    monkeypatch.setattr(runner, "_digest_descriptor", digest_then_change)

    with pytest.raises(ValueError, match="changed while it was staged"):
        runner._stage_once(source, target, mode=0o644)
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_neutral_root_preflight_rejects_private_parent_and_outside_work(
    tmp_path,
) -> None:
    public_root = tmp_path / "public"
    public_root.mkdir(mode=0o755)
    auth_parent = tmp_path / "auth"
    auth_parent.mkdir(mode=0o700)
    auth = auth_parent / "auth.json"
    auth.write_text("{}")
    valid = SimpleNamespace(
        public_path_root=str(public_root),
        work_dir=str(public_root / "evaluation"),
    )

    work, public = runner._preflight_neutral_paths(valid, auth_path=auth)

    assert work == (public_root / "evaluation").resolve()
    assert public == public_root.resolve()

    private_parent = SimpleNamespace(
        public_path_root=str(tmp_path),
        work_dir=str(tmp_path / "evaluation"),
    )
    with pytest.raises(ValueError, match="overlaps a private path root"):
        runner._preflight_neutral_paths(private_parent, auth_path=auth)

    outside = tmp_path / "outside"
    outside_args = SimpleNamespace(
        public_path_root=str(public_root),
        work_dir=str(outside),
    )
    with pytest.raises(ValueError, match="inside --public-path-root"):
        runner._preflight_neutral_paths(outside_args, auth_path=auth)
    assert not outside.exists()


def test_neutral_root_preflight_rejects_symlink_and_world_writable_root(
    tmp_path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir(mode=0o755)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    auth_parent = tmp_path / "auth"
    auth_parent.mkdir(mode=0o700)
    auth = auth_parent / "auth.json"
    auth.write_text("{}")

    linked = SimpleNamespace(
        public_path_root=str(linked_root),
        work_dir=str(linked_root / "evaluation"),
    )
    with pytest.raises(ValueError, match="owner-controlled"):
        runner._preflight_neutral_paths(linked, auth_path=auth)

    real_root.chmod(0o777)
    writable = SimpleNamespace(
        public_path_root=str(real_root),
        work_dir=str(real_root / "evaluation"),
    )
    with pytest.raises(ValueError, match="owner-controlled"):
        runner._preflight_neutral_paths(writable, auth_path=auth)

    filesystem_root = SimpleNamespace(
        public_path_root="/",
        work_dir=str(tmp_path / "never-created"),
    )
    with pytest.raises(ValueError, match="public path root"):
        runner._preflight_neutral_paths(filesystem_root, auth_path=auth)
    assert not Path(filesystem_root.work_dir).exists()


def test_run_rejects_existing_public_evidence_and_points_to_validate(
    tmp_path, monkeypatch
) -> None:
    package = tmp_path / "public-evidence"
    package.mkdir()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("existing evidence must not touch run inputs or Harbor state")

    monkeypatch.setattr(
        runner,
        "validate_terminal_bench_public_evidence",
        unexpected,
    )
    monkeypatch.setattr(runner, "_preflight_neutral_paths", unexpected)
    monkeypatch.setattr(runner, "_load_or_create_protocol", unexpected)
    monkeypatch.setattr(runner, "initialize_terminal_evaluation", unexpected)
    monkeypatch.setattr(runner, "run_terminal_phase", unexpected)
    args = SimpleNamespace(
        public_out=str(package),
        auth=str(tmp_path / "missing-auth.json"),
    )

    with pytest.raises(ValueError, match="use the validate subcommand"):
        runner._run(args)


def test_resume_revalidates_registration_and_does_not_repeat_sealed_smoke(
    tmp_path, monkeypatch
) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    protocol_path = tmp_path / "protocol.json"
    wheel = tmp_path / "input.whl"
    codex = tmp_path / "codex"
    protocol = SimpleNamespace(
        evaluation_id="a" * 32,
        output_root=str((tmp_path / "jobs").resolve()),
    )
    args = SimpleNamespace(
        public_out=str(tmp_path / "new-public-evidence"),
        auth=str(auth),
        work_dir=str(tmp_path),
        public_path_root=str(tmp_path),
    )
    monkeypatch.setattr(
        runner,
        "_preflight_neutral_paths",
        lambda _args, *, auth_path: (tmp_path, tmp_path),
    )
    monkeypatch.setattr(
        runner,
        "_load_or_create_protocol",
        lambda _args: (protocol, protocol_path, wheel, codex, False),
    )
    monkeypatch.setattr(
        runner,
        "build_harbor_commands",
        lambda _protocol, run_kind, **_kwargs: (f"{run_kind}-command",),
    )
    initialized: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runner,
        "initialize_terminal_evaluation",
        lambda _protocol, commands, **_kwargs: initialized.append(tuple(commands)),
    )
    smoke_seal = object()
    smoke_manifest = object()
    monkeypatch.setattr(
        runner,
        "_load_sealed_smoke",
        lambda _protocol: (smoke_seal, smoke_manifest),
    )
    phases: list[str] = []
    monkeypatch.setattr(
        runner,
        "run_terminal_phase",
        lambda _protocol, run_kind, *_args, **_kwargs: phases.append(run_kind),
    )
    scored_manifest = object()
    monkeypatch.setattr(
        runner,
        "validate_harbor_results",
        lambda *_args, **_kwargs: scored_manifest,
    )
    records = object()
    monkeypatch.setattr(
        runner,
        "derive_terminal_bench_records",
        lambda *_args, **_kwargs: records,
    )

    class _Summary:
        @staticmethod
        def to_markdown():
            return "summary"

    summary = _Summary()
    monkeypatch.setattr(
        runner,
        "summarize_records",
        lambda *_args, **_kwargs: summary,
    )

    class _Validation:
        @staticmethod
        def model_dump(*, mode):
            return {"mode": mode}

    monkeypatch.setattr(
        runner,
        "export_terminal_bench_public_evidence",
        lambda *_args, **_kwargs: _Validation(),
    )

    assert runner._run(args) == 0
    assert initialized == [("smoke-command", "scored-command")]
    assert phases == ["scored"]
