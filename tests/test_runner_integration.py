"""Regression tests for Dev A's two saves on the same final step."""
import pytest
from store import save, load_last, list_steps, CheckpointConflictError


def completed_record(*, reported=False):
    return {
        "agent": {"step": 3, "done": True, "stopped": False},
        "history": [{"run_id": "r", "step": 3, "action": "finish"}],
        "meta": {"name": "test", "outcome": "recovered", "reported": reported,
                 "events": [], "window_start": 3, "interventions": 1,
                 "vetoes": 0, "first_flag_step": 1},
    }


def test_runner_can_persist_reported_flag_after_writing_report(tmp_path):
    db = tmp_path / "demo.db"
    save("r", 3, completed_record(), db_path=db)
    # Dev C's report uses this persisted pre-report state.
    assert load_last("r", db_path=db)["meta"]["reported"] is False
    save("r", 3, completed_record(reported=True), db_path=db)
    assert load_last("r", db_path=db)["meta"]["reported"] is True
    assert len(list_steps("r", db_path=db)) == 1
    save("r", 3, completed_record(reported=True), db_path=db)  # idempotent


def test_report_update_cannot_modify_agent_state(tmp_path):
    db = tmp_path / "demo.db"
    save("r", 3, completed_record(), db_path=db)
    mutated = completed_record(reported=True)
    mutated["agent"]["step"] = 99
    with pytest.raises(CheckpointConflictError):
        save("r", 3, mutated, db_path=db)
    assert load_last("r", db_path=db)["meta"]["reported"] is False


def test_report_update_cannot_undo_reported_flag(tmp_path):
    db = tmp_path / "demo.db"
    save("r", 3, completed_record(), db_path=db)
    save("r", 3, completed_record(reported=True), db_path=db)
    with pytest.raises(CheckpointConflictError):
        save("r", 3, completed_record(), db_path=db)


def test_only_completed_outcomes_can_update_reported(tmp_path):
    db = tmp_path / "demo.db"
    running = completed_record()
    running["meta"]["outcome"] = "running"
    save("r", 3, running, db_path=db)
    running["meta"]["reported"] = True
    with pytest.raises(CheckpointConflictError):
        save("r", 3, running, db_path=db)


def test_report_update_cannot_change_other_meta_fields(tmp_path):
    """The one-time exception is scoped to flipping `reported`, not a general
    second-write allowance; a runner bug that recomputes other meta during
    the report save must still be caught."""
    db = tmp_path / "demo.db"
    save("r", 3, completed_record(), db_path=db)
    mutated = completed_record(reported=True)
    mutated["meta"]["vetoes"] = 99
    with pytest.raises(CheckpointConflictError):
        save("r", 3, mutated, db_path=db)
    assert load_last("r", db_path=db)["meta"]["reported"] is False


def test_crash_between_the_two_report_saves_leaves_reported_false(tmp_path):
    """If the process dies after the first save (terminal outcome, reported
    still False) but before write_report + the second save ever run, the
    checkpoint on disk must faithfully show reported=False -- store.py's job
    is only to persist that truth, not to guess whether reporting happened.
    (The runner must be the one to notice and retry; see README_DEV_B notes.)"""
    db = tmp_path / "demo.db"
    save("r", 3, completed_record(reported=False), db_path=db)
    state = load_last("r", db_path=db)
    assert state["meta"]["outcome"] == "recovered"
    assert state["meta"]["reported"] is False
    assert len(list_steps("r", db_path=db)) == 1
