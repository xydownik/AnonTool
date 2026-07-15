"""SQLite persistence layer for the Token <-> Real value mapping.

Each anonymization run creates a short "session key" (e.g. "KZ-74A1"). Every
entity that gets tokenized is stored as one row: (session_id, token,
original_value, entity_type, created_at). The reverse pass looks the session
up by key and replaces tokens with their original values.

Records older than 24 hours are considered expired and are purged by
`cleanup_old_records()`, which is called once on every app startup.
"""

from __future__ import annotations

import datetime
import os
import secrets
import sqlite3
import string
import threading
from typing import Dict, List, Tuple

DB_PATH = os.environ.get(
    "ANON_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "anon_mapping.db"),
)

# sqlite3 connections are cheap; a single process-wide lock keeps concurrent
# Streamlit sessions/threads from writing at the same time.
_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the database/table (if missing) and purge stale records.

    Safe to call on every app startup / rerun.
    """
    with _lock, get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mappings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                token           TEXT NOT NULL,
                original_value  TEXT NOT NULL,
                entity_type     TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mappings_session_id ON mappings(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mappings_created_at ON mappings(created_at)"
        )
        conn.commit()
    cleanup_old_records()


def cleanup_old_records(max_age_hours: int = 24) -> int:
    """Delete mapping rows older than `max_age_hours`. Returns rows deleted."""
    cutoff = (
        datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)
    ).strftime("%Y-%m-%d %H:%M:%S")
    with _lock, get_connection() as conn:
        cursor = conn.execute("DELETE FROM mappings WHERE created_at < ?", (cutoff,))
        conn.commit()
        return cursor.rowcount


def generate_session_key() -> str:
    alphabet = string.ascii_uppercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(4))
    return f"KZ-{suffix}"


def session_exists(session_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM mappings WHERE session_id = ? LIMIT 1", (session_id,)
        ).fetchone()
    return row is not None


def generate_unique_session_key(max_attempts: int = 10) -> str:
    for _ in range(max_attempts):
        key = generate_session_key()
        if not session_exists(key):
            return key
    raise RuntimeError("Не удалось сгенерировать уникальный ключ сессии, попробуйте ещё раз.")


def save_mapping(session_id: str, records: List[Tuple[str, str, str]]) -> None:
    """Persist mapping rows for a session.

    `records` is a list of (token, original_value, entity_type) tuples.
    """
    if not records:
        return
    with _lock, get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO mappings (session_id, token, original_value, entity_type)
            VALUES (?, ?, ?, ?)
            """,
            [(session_id, token, value, entity_type) for token, value, entity_type in records],
        )
        conn.commit()


def load_mapping(session_id: str) -> Dict[str, str]:
    """Return {token: original_value} for a session.

    Raises KeyError if the session key does not exist (or has expired and was
    already purged by `cleanup_old_records`).
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise KeyError("Ключ восстановления не может быть пустым.")

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT token, original_value FROM mappings WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    if not rows:
        raise KeyError(
            f"Сессия с ключом «{session_id}» не найдена. Возможно, ключ введён с "
            "ошибкой, либо срок хранения (24 часа) уже истёк."
        )
    return {row["token"]: row["original_value"] for row in rows}
