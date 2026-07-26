"""Database connection lifecycle."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .migrations import migrate


def open_database(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    migrate(connection)
    return connection

