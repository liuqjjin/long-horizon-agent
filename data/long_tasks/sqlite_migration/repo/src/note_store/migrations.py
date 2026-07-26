"""Schema migrations."""

from __future__ import annotations

import sqlite3


def migrate(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        connection.execute(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)"
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    if version < 2:
        connection.execute("ALTER TABLE notes ADD COLUMN updated_at TEXT")
        connection.commit()

