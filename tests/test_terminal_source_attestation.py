from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from lha.bench import terminal_bench as tb
from lha.bench import terminal_public_evidence as tpe

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_COMMITTED_EVIDENCE = _REPOSITORY_ROOT / "benchmarks" / "terminal_bench_2_1"
_EVALUATED_COMMIT = "e63f94620ce8ddd322b19ccb159381183fc31933"
_EVALUATED_TREE = "c5b410a98ebec58d482a8ebc889758bc67662985"
_EVALUATED_WHEEL = "d34e0569943102c73b8ac4d6209bf5a3a061fada285f7e00c6f70c107f10fac0"


def _attestation() -> tpe.SourceAttestation:
    return tpe.SourceAttestation(
        repository_url="https://github.com/liuqjjin/long-horizon-agent",
        commit_sha=_EVALUATED_COMMIT,
        tree_sha=_EVALUATED_TREE,
        package_version="0.4.2.dev0",
        wheel_filename="lha-0.4.2.dev0-py3-none-any.whl",
        wheel_size_bytes=365_480,
        wheel_sha256=_EVALUATED_WHEEL,
        reproducible_build_command="uv build --clear",
    )


def _schema3_package(tmp_path: Path) -> Path:
    package = tmp_path / "schema3"
    shutil.copytree(_COMMITTED_EVIDENCE, package)
    index_path = package / "evidence.json"
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        index_path.read_bytes()
    )
    if index.schema_version == 4:
        (package / "source_attestation.json").unlink()
        summary = tb.TerminalBenchSummary.model_validate_json(
            (package / "summary.json").read_bytes()
        )
        markdown = tpe._summary_markdown_bytes(summary, schema_version=3)
        (package / "summary.md").write_bytes(markdown)
        index = tpe.TerminalBenchPublicEvidenceIndex.model_validate(
            {
                **index.model_dump(
                    mode="json",
                    exclude={"source_attestation_sha256"},
                ),
                "schema_version": 3,
                "summary_markdown_sha256": hashlib.sha256(markdown).hexdigest(),
            }
        )
        index_path.write_bytes(tpe._canonical_model_bytes(index))
    return package


def test_schema3_remains_readable_and_upgrade_binds_evaluated_source(
    tmp_path: Path,
) -> None:
    source = _schema3_package(tmp_path)
    legacy = tpe.validate_terminal_bench_public_evidence(source)
    assert legacy.evaluated_commit_sha is None
    assert legacy.evaluated_tree_sha is None
    assert legacy.evaluated_wheel_filename is None
    assert legacy.evaluated_wheel_size_bytes is None
    assert legacy.evaluated_wheel_sha256 is None
    assert "- 协议错误：4" in (source / "summary.md").read_text()

    target = tmp_path / "schema4"
    upgraded = tpe.upgrade_terminal_bench_public_evidence(
        source,
        target,
        source_attestation=_attestation(),
    )

    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        (target / "evidence.json").read_bytes()
    )
    assert index.schema_version == 4
    assert index.source_attestation_sha256 == tb.sha256_file(
        target / "source_attestation.json"
    )
    assert upgraded == tpe.validate_terminal_bench_public_evidence(target)
    assert upgraded.evaluated_commit_sha == _EVALUATED_COMMIT
    assert upgraded.evaluated_tree_sha == _EVALUATED_TREE
    assert upgraded.evaluated_wheel_filename == "lha-0.4.2.dev0-py3-none-any.whl"
    assert upgraded.evaluated_wheel_size_bytes == 365_480
    assert upgraded.evaluated_wheel_sha256 == _EVALUATED_WHEEL
    assert "- 不可评分 ERROR：4" in (target / "summary.md").read_text()
    assert "协议错误" not in (target / "summary.md").read_text()
    assert "- 协议错误：4" in (source / "summary.md").read_text()


def test_schema4_rejects_attestation_wheel_forgery_even_with_updated_index(
    tmp_path: Path,
) -> None:
    source = _schema3_package(tmp_path)
    target = tmp_path / "schema4"
    tpe.upgrade_terminal_bench_public_evidence(
        source,
        target,
        source_attestation=_attestation(),
    )
    attestation_path = target / "source_attestation.json"
    forged_attestation = _attestation().model_copy(
        update={"wheel_sha256": "0" * 64}
    )
    attestation_path.write_bytes(tpe._canonical_model_bytes(forged_attestation))
    index_path = target / "evidence.json"
    index = tpe.TerminalBenchPublicEvidenceIndex.model_validate_json(
        index_path.read_bytes()
    )
    index_path.write_bytes(
        tpe._canonical_model_bytes(
            index.model_copy(
                update={
                    "source_attestation_sha256": tb.sha256_file(attestation_path),
                }
            )
        )
    )

    with pytest.raises(ValueError, match="wheel does not match"):
        tpe.validate_terminal_bench_public_evidence(target)


def test_schema4_rejects_attestation_byte_tampering(tmp_path: Path) -> None:
    source = _schema3_package(tmp_path)
    target = tmp_path / "schema4"
    tpe.upgrade_terminal_bench_public_evidence(
        source,
        target,
        source_attestation=_attestation(),
    )
    path = target / "source_attestation.json"
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError):
        tpe.validate_terminal_bench_public_evidence(target)


def test_attestation_upgrade_failure_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _schema3_package(tmp_path)
    target = tmp_path / "schema4"
    original_write = tpe._write_new_file
    calls = 0

    def fail_during_copy(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected attestation copy failure")
        original_write(path, payload)

    monkeypatch.setattr(tpe, "_write_new_file", fail_during_copy)
    with pytest.raises(OSError, match="injected"):
        tpe.upgrade_terminal_bench_public_evidence(
            source,
            target,
            source_attestation=_attestation(),
        )

    assert not target.exists()
    assert not (tmp_path / ".schema4.attestation.lock").exists()
    assert not list(tmp_path.glob(".schema4.*"))
    tpe.validate_terminal_bench_public_evidence(source)


def test_attestation_upgrade_rejects_protocol_wheel_mismatch_before_output(
    tmp_path: Path,
) -> None:
    source = _schema3_package(tmp_path)
    target = tmp_path / "schema4"
    mismatched = _attestation().model_copy(update={"wheel_sha256": "f" * 64})

    with pytest.raises(ValueError, match="frozen protocol"):
        tpe.upgrade_terminal_bench_public_evidence(
            source,
            target,
            source_attestation=mismatched,
        )
    assert not target.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "repository_url",
            "https://user@example.com/private",
            "public HTTPS repository",
        ),
        (
            "wheel_filename",
            "../lha-0.4.2.dev0-py3-none-any.whl",
            "wheel filename",
        ),
        (
            "reproducible_build_command",
            "uv build --clear\nprintenv",
            "one non-empty line",
        ),
    ],
)
def test_source_attestation_rejects_unsafe_identity_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tpe.SourceAttestation.model_validate(
            {
                **_attestation().model_dump(mode="json"),
                field: value,
            }
        )


@pytest.mark.parametrize("wheel_size_bytes", [0, 64 * 1024 * 1024 + 1])
def test_source_attestation_rejects_unreasonable_wheel_size(
    wheel_size_bytes: int,
) -> None:
    with pytest.raises(ValueError):
        tpe.SourceAttestation.model_validate(
            {
                **_attestation().model_dump(mode="json"),
                "wheel_size_bytes": wheel_size_bytes,
            }
        )
