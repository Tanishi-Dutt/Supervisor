"""Tests for durable, idempotent checkpoints, including real SIGKILL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from store import (
    CheckpointConflictError,
    StaleCheckpointError,
    list_steps,
    load_checkpoint,
    load_last,
    save,
)


def make_step(n: int, run_id: str = "run-1"):
    return {
        "run_id": run_id, "step": n, "action": "lookup",
        "args": {"item": n}, "result": "ok",
        "claimed_status": None, "evidence": {"test_passed": True},
    }


def test_missing_checkpoint_returns_none(tmp_path):
    db = tmp_path / "test.db"
    assert load_last("missing", db_path=db) is None
    assert load_checkpoint("missing", db_path=db) is None


def test_save_then_load_latest_state(tmp_path):
    db = tmp_path / "test.db"
    save("run-1", make_step(1), {"next_step": 2, "job": "A"}, db_path=db)
    save("run-1", make_step(2), {"next_step": 3, "job": "A"}, db_path=db)
    assert load_last("run-1", db_path=db) == {"next_step": 3, "job": "A"}
    c = load_checkpoint("run-1", db_path=db)
    assert c["step_number"] == 2
    assert c["step"]["action"] == "lookup"


def test_insert_with_integer_step_keeps_contract(tmp_path):
    db = tmp_path / "test.db"
    save("run-1", 1, {"cursor": 1}, db_path=db)
    assert load_checkpoint("run-1", db_path=db)["step"] is None


def test_multiple_runs_stay_isolated(tmp_path):
    db = tmp_path / "test.db"
    save("run-1", 1, {"value": "one"}, db_path=db)
    save("run-2", 5, {"value": "two"}, db_path=db)
    assert load_last("run-1", db_path=db)["value"] == "one"
    assert load_last("run-2", db_path=db)["value"] == "two"


def test_identical_duplicate_save_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    event = make_step(1)
    state = {"cursor": 2, "jobs_done": ["a"]}
    save("run-1", event, state, db_path=db)
    saved_at = load_checkpoint("run-1", db_path=db)["saved_at"]
    save("run-1", event, state, db_path=db)
    assert len(list_steps("run-1", db_path=db)) == 1
    assert load_checkpoint("run-1", db_path=db)["saved_at"] == saved_at


def test_conflicting_duplicate_checkpoint_fails(tmp_path):
    db = tmp_path / "test.db"
    save("run-1", 1, {"cursor": 2}, db_path=db)
    with pytest.raises(CheckpointConflictError):
        save("run-1", 1, {"cursor": 3}, db_path=db)
    assert load_last("run-1", db_path=db)["cursor"] == 2


def test_new_older_checkpoint_is_rejected(tmp_path):
    db = tmp_path / "test.db"
    save("run-1", 2, {"cursor": 3}, db_path=db)
    with pytest.raises(StaleCheckpointError):
        save("run-1", 1, {"cursor": 2}, db_path=db)


def test_reject_bad_step_and_state(tmp_path):
    db = tmp_path / "test.db"
    with pytest.raises(ValueError):
        save("run-1", -1, {}, db_path=db)
    with pytest.raises(TypeError):
        save("run-1", 1, ["not mapping"], db_path=db)
    with pytest.raises(ValueError):
        save("run-1", 1, {"bad": object()}, db_path=db)


def test_reject_mismatched_run_ids(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        save("run-1", make_step(1, run_id="run-2"), {}, db_path=tmp_path / "test.db")


def test_list_steps_returns_chronological_history(tmp_path):
    db = tmp_path / "test.db"
    for n in (1, 2, 3):
        save("run-1", make_step(n), {"next_step": n + 1}, db_path=db)
    assert [x["step_number"] for x in list_steps("run-1", db_path=db)] == [1, 2, 3]
    assert len(list_steps("missing", db_path=db)) == 0


def test_env_database_path_can_be_switched(tmp_path, monkeypatch):
    db = tmp_path / "another" / "store.db"
    monkeypatch.setenv("FLIGHT_RECORDER_DB", str(db))
    save("run-1", 1, {"cursor": 2})
    assert db.exists()
    assert load_last("run-1") == {"cursor": 2}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGKILL only")
def test_checkpoint_survives_real_sigkill(tmp_path):
    """Write in a child process, SIGKILL it, and read committed data in parent."""
    db = tmp_path / "kill.db"
    marker = tmp_path / "ready"
    project_root = Path(__file__).resolve().parents[1]
    script = r'''
import os, pathlib, signal, sys
from store import save
path, marker = sys.argv[1:]
save("crash-run", 1, {"next_step": 2, "task_done": "a"}, db_path=path)
save("crash-run", 2, {"next_step": 3, "task_done": "b"}, db_path=path)
pathlib.Path(marker).write_text("ready")
signal.pause()
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(db), str(marker)],
        cwd=str(project_root), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.025)
        assert marker.exists(), f"Child failed: {child.stderr.read() if child.poll() is not None else 'no marker'}"
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if child.stderr:
            child.stderr.close()
    assert load_last("crash-run", db_path=db) == {"next_step": 3, "task_done": "b"}
    assert load_checkpoint("crash-run", db_path=db)["step_number"] == 2
