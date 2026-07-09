"""
SQLite persistence for footfall events and processing runs.

Two tables are used:

* ``footfall_events``  — one row per valid line-crossing event.
* ``processing_runs``  — one row per processed sequence (a run summary).

Connections are opened per-call and closed via context managers so the database
is never left locked between operations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from . import config

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS footfall_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_name TEXT,
    frame_index   INTEGER,
    timestamp     TEXT,
    track_id      INTEGER,
    direction     TEXT,
    line_name     TEXT,
    created_at    TEXT
);
"""

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS processing_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_name       TEXT,
    model_name          TEXT,
    tracker_name        TEXT,
    total_in            INTEGER,
    total_out           INTEGER,
    total_unique_tracks INTEGER,
    total_frames        INTEGER,
    fps                 REAL,
    processed_at        TEXT
);
"""


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection, ensuring the parent directory exists."""
    path = Path(db_path) if db_path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _session(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """
    Context manager that commits on success and **always closes** the connection.

    ``with sqlite3.connect() as conn`` only manages the transaction — it leaves
    the connection open, which on Windows keeps the database file locked. This
    wrapper guarantees the handle is released.
    """
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create the tables if they do not already exist."""
    with _session(db_path) as conn:
        conn.execute(_CREATE_EVENTS)
        conn.execute(_CREATE_RUNS)


def reset_db(db_path: str | Path | None = None) -> None:
    """Drop and recreate all tables (destroys existing data)."""
    with _session(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS footfall_events;")
        conn.execute("DROP TABLE IF EXISTS processing_runs;")
        conn.execute(_CREATE_EVENTS)
        conn.execute(_CREATE_RUNS)


def delete_sequence(sequence_name: str, db_path: str | Path | None = None) -> None:
    """
    Remove all events and run summaries for one sequence.

    Called before (re)processing a sequence so a re-run *replaces* its data
    rather than appending duplicate crossing events.
    """
    with _session(db_path) as conn:
        conn.execute(
            "DELETE FROM footfall_events WHERE sequence_name = ?;", (sequence_name,)
        )
        conn.execute(
            "DELETE FROM processing_runs WHERE sequence_name = ?;", (sequence_name,)
        )


# --------------------------------------------------------------------------- #
# Inserts
# --------------------------------------------------------------------------- #
def insert_event(
    sequence_name: str,
    frame_index: int,
    timestamp: str,
    track_id: int,
    direction: str,
    line_name: str,
    db_path: str | Path | None = None,
) -> None:
    """Insert a single crossing event."""
    with _session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO footfall_events
                (sequence_name, frame_index, timestamp, track_id,
                 direction, line_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                sequence_name,
                frame_index,
                timestamp,
                int(track_id),
                direction,
                line_name,
                datetime.now().strftime(config.TIMESTAMP_FORMAT),
            ),
        )


def insert_events_bulk(rows: Iterable[tuple], db_path: str | Path | None = None) -> int:
    """
    Insert many crossing events at once for efficiency.

    Each row must be
    ``(sequence_name, frame_index, timestamp, track_id, direction, line_name)``.
    ``created_at`` is added automatically. Returns the number of rows inserted.
    """
    created_at = datetime.now().strftime(config.TIMESTAMP_FORMAT)
    payload = [(*row, created_at) for row in rows]
    if not payload:
        return 0
    with _session(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO footfall_events
                (sequence_name, frame_index, timestamp, track_id,
                 direction, line_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            payload,
        )
    return len(payload)


def insert_run(
    sequence_name: str,
    model_name: str,
    tracker_name: str,
    total_in: int,
    total_out: int,
    total_unique_tracks: int,
    total_frames: int,
    fps: float,
    db_path: str | Path | None = None,
) -> None:
    """Insert a processing-run summary row."""
    with _session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO processing_runs
                (sequence_name, model_name, tracker_name, total_in, total_out,
                 total_unique_tracks, total_frames, fps, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                sequence_name,
                model_name,
                tracker_name,
                int(total_in),
                int(total_out),
                int(total_unique_tracks),
                int(total_frames),
                float(fps),
                datetime.now().strftime(config.TIMESTAMP_FORMAT),
            ),
        )


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def fetch_events(
    sequence_name: str | None = None, db_path: str | Path | None = None
) -> list[sqlite3.Row]:
    """Return crossing events, optionally filtered to one sequence."""
    with _session(db_path) as conn:
        if sequence_name:
            cur = conn.execute(
                "SELECT * FROM footfall_events WHERE sequence_name = ? "
                "ORDER BY frame_index;",
                (sequence_name,),
            )
        else:
            cur = conn.execute("SELECT * FROM footfall_events ORDER BY frame_index;")
        return cur.fetchall()


def fetch_sequences(db_path: str | Path | None = None) -> list[str]:
    """Return the distinct sequence names that have recorded events."""
    with _session(db_path) as conn:
        cur = conn.execute(
            "SELECT DISTINCT sequence_name FROM footfall_events "
            "ORDER BY sequence_name;"
        )
        return [row["sequence_name"] for row in cur.fetchall()]


def fetch_runs(db_path: str | Path | None = None) -> list[sqlite3.Row]:
    """Return all processing-run summaries, newest first."""
    with _session(db_path) as conn:
        cur = conn.execute("SELECT * FROM processing_runs ORDER BY id DESC;")
        return cur.fetchall()


def fetch_latest_run(
    sequence_name: str, db_path: str | Path | None = None
) -> sqlite3.Row | None:
    """Return the most recent processing-run summary for one sequence."""
    if not table_exists("processing_runs", db_path):
        return None
    with _session(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM processing_runs WHERE sequence_name = ? "
            "ORDER BY id DESC LIMIT 1;",
            (sequence_name,),
        )
        return cur.fetchone()


def table_exists(name: str, db_path: str | Path | None = None) -> bool:
    """Return True if a table with the given name exists."""
    with _session(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (name,),
        )
        return cur.fetchone() is not None
