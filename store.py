"""SQLite checkpoints for Agent Flight Recorder (Dev B).

Public team contract:
    save(run_id, step, state)
    load_last(run_id) -> state | None

Pass the full Step dict (preferred) or an integer step index. Use
load_checkpoint() when the runner needs its exact restart step index.

SQLite commits are durable on disk, but exactly-once external side effects
require idempotency in Dev A's actual agent tools; a database alone cannot
make external APIs exactly-once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, TypedDict


class CheckpointConflictError(ValueError):
    """A step already exists with different data."""


class StaleCheckpointError(ValueError):
    """A new checkpoint is older than the most recently committed step."""


class Checkpoint(TypedDict):
    run_id: str
    step_number: int
    step: dict[str, Any] | None
    state: dict[str, Any]
    saved_at: float


def _db_path(db_path: str | os.PathLike[str] | None) -> Path:
    raw = db_path if db_path is not None else os.environ.get(
        "FLIGHT_RECORDER_DB", "data/flight_recorder.sqlite3"
    )
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(db_path: str | os.PathLike[str] | None) -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path(db_path), timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            run_id TEXT NOT NULL,
            step_number INTEGER NOT NULL,
            step_json TEXT,
            state_json TEXT NOT NULL,
            saved_at REAL NOT NULL,
            PRIMARY KEY (run_id, step_number)
        )
    """)
    connection.commit()
    return connection


def _step_parts(step: int | Mapping[str, Any]) -> tuple[int, dict[str, Any] | None]:
    if isinstance(step, Mapping):
        entry = dict(step)
        index = entry.get("step")
    else:
        entry = None
        index = step
    if type(index) is not int or index < 0:
        raise ValueError("step must be a nonnegative integer or a Step with integer 'step'")
    return index, entry


def _json_encode(value: Any, label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def save(
    run_id: str,
    step: int | Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    db_path: str | os.PathLike[str] | None = None,
) -> None:
    """Atomically commit one step and its resumable state.

    Duplicate identical writes are safe no-ops. Conflicting duplicate writes
    are rejected. A new lower-numbered checkpoint cannot rewind a run.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(state, Mapping):
        raise TypeError("state must be a JSON-serializable mapping")
    number, step_event = _step_parts(step)
    if step_event is not None and step_event.get("run_id") not in (None, run_id):
        raise ValueError("step.run_id must match run_id")
    step_json = None if step_event is None else _json_encode(step_event, "step")
    state_json = _json_encode(dict(state), "state")

    con = _connect(db_path)
    try:
        with con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT step_json, state_json FROM checkpoints "
                "WHERE run_id=? AND step_number=?", (run_id, number)
            ).fetchone()
            if row is not None:
                if row["step_json"] != step_json or row["state_json"] != state_json:
                    raise CheckpointConflictError(
                        f"Conflicting checkpoint for run={run_id!r}, step={number}"
                    )
                return
            row = con.execute(
                "SELECT MAX(step_number) AS latest FROM checkpoints WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row["latest"] is not None and number < row["latest"]:
                raise StaleCheckpointError(
                    f"Cannot append step {number} after step {row['latest']}"
                )
            con.execute(
                "INSERT INTO checkpoints "
                "(run_id, step_number, step_json, state_json, saved_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, number, step_json, state_json, time.time()),
            )
    finally:
        con.close()


def load_checkpoint(
    run_id: str,
    *,
    db_path: str | os.PathLike[str] | None = None,
) -> Checkpoint | None:
    """Return latest durable step + agent state for an exact resume point."""
    con = _connect(db_path)
    try:
        row = con.execute(
            "SELECT run_id, step_number, step_json, state_json, saved_at "
            "FROM checkpoints WHERE run_id=? ORDER BY step_number DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "step_number": row["step_number"],
            "step": json.loads(row["step_json"]) if row["step_json"] is not None else None,
            "state": json.loads(row["state_json"]),
            "saved_at": row["saved_at"],
        }
    finally:
        con.close()


def load_last(
    run_id: str,
    *,
    db_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Return the latest saved state (as required by the team interface)."""
    checkpoint = load_checkpoint(run_id, db_path=db_path)
    return None if checkpoint is None else checkpoint["state"]


def list_steps(
    run_id: str,
    *,
    db_path: str | os.PathLike[str] | None = None,
) -> list[Checkpoint]:
    """Expose committed history for Dev C's incident report generation."""
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT run_id, step_number, step_json, state_json, saved_at "
            "FROM checkpoints WHERE run_id=? ORDER BY step_number ASC", (run_id,)
        ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "step_number": row["step_number"],
                "step": json.loads(row["step_json"]) if row["step_json"] is not None else None,
                "state": json.loads(row["state_json"]),
                "saved_at": row["saved_at"],
            }
            for row in rows
        ]
    finally:
        con.close()
