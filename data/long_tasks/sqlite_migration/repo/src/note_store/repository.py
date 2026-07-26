"""Persistence API."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Note:
    id: int
    body: str
    updated_at: datetime


class NoteRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def add(self, body: str) -> Note:
        cursor = self.connection.execute("INSERT INTO notes(body) VALUES (?)", (body,))
        self.connection.commit()
        return self.get(int(cursor.lastrowid))

    def get(self, note_id: int) -> Note:
        row = self.connection.execute(
            "SELECT id, body, updated_at FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        if row is None:
            raise KeyError(note_id)
        return Note(
            id=int(row["id"]),
            body=str(row["body"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

