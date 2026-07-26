"""Todo data access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_items(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"todo file does not exist: {source}")
    data = json.loads(source.read_text())
    if not isinstance(data, list):
        raise ValueError("todo file must contain a JSON list")
    return [dict(item) for item in data]


def find_item(items: list[dict[str, Any]], item_id: int) -> dict[str, Any]:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise ValueError(f"todo {item_id} not found")

