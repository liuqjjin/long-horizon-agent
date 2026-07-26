from __future__ import annotations

import sqlite3

from note_store import NoteRepository, open_database


def _create_v1(path):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
    connection.execute("INSERT INTO notes(body) VALUES ('legacy')")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()


def test_v1_database_is_backfilled_and_migration_is_idempotent(tmp_path):
    path = tmp_path / "notes.db"
    _create_v1(path)

    first = open_database(path)
    legacy = NoteRepository(first).get(1)
    assert legacy.body == "legacy"
    assert legacy.updated_at.tzinfo is not None
    first.close()

    second = open_database(path)
    assert second.execute("PRAGMA user_version").fetchone()[0] == 2
    second.close()


def test_new_rows_survive_reopen_with_timestamp(tmp_path):
    path = tmp_path / "notes.db"
    connection = open_database(path)
    created = NoteRepository(connection).add("persist me")
    connection.close()

    reopened = open_database(path)
    loaded = NoteRepository(reopened).get(created.id)
    assert loaded.body == "persist me"
    assert loaded.updated_at == created.updated_at
    reopened.close()

