from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "todo_cli.cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_json_output_is_one_machine_readable_document(tmp_path):
    source = tmp_path / "todos.json"
    source.write_text(json.dumps([{"id": 1, "title": "ship"}]))
    result = _run("--file", str(source), "--format", "json")
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"items": [{"id": 1, "title": "ship"}]}
    assert result.stderr == ""


def test_missing_item_uses_stderr_and_nonzero_exit(tmp_path):
    source = tmp_path / "todos.json"
    source.write_text(json.dumps([{"id": 1, "title": "ship"}]))
    result = _run("--file", str(source), "--id", "99")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "not found" in result.stderr

