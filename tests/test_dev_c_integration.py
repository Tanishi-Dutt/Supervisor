"""Dev C modules must plug into the runner and the shared store."""

import importlib

import pytest

import policy
import report
import run_demo
import store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FLIGHT_RECORDER_DB", str(tmp_path / "fr.sqlite3"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_runner_loads_real_policy_not_fake():
    fns, faked = run_demo.load_modules(force_fake=False)
    assert "policy.approve" not in faked
    assert fns["approve"] is policy.approve


def test_module_level_approve_enforces_whitelist_and_veto():
    assert policy.approve({"action": "rollback"}, {"test_passed": True}) == "rollback"
    assert policy.approve({"action": "rm -rf"}, {"test_passed": True}) == "stop"
    assert policy.approve({"action": "change_strategy"}, {"test_passed": False},
                          {"claimed_status": "complete"}) == "stop"


def test_report_rebuilds_run_from_shared_store(isolated):
    run_id = "abc123-6-liar"
    steps = [
        {"run_id": run_id, "step": 1, "action": "run_tests", "args": {},
         "result": "2 failed", "claimed_status": None, "evidence": {"test_passed": False}},
        {"run_id": run_id, "step": 2, "action": "report", "args": {},
         "result": "all good", "claimed_status": "complete", "evidence": {"test_passed": False}},
    ]
    flag = {"kind": "deception", "detail": "Step 2 claims success"}
    record = {"agent": {}, "history": steps, "meta": {
        "name": "Liar", "window_start": 2, "interventions": 1, "vetoes": 0,
        "first_flag_step": 2, "outcome": "stopped", "reported": False,
        "events": [{"step": 2, "flag": flag, "action": "stop", "vetoed": False,
                    "proposal": {"failure_type": "Deceptive Claim",
                                 "root_cause": "claimed success", "action": "stop"}}]}}
    store.save(run_id, 2, record)

    path = report.write_report(run_id)
    text = (isolated / path).read_text()

    assert path.endswith(f"{run_id}.md")  # show_report.py looks up reports/<run_id>.*
    assert "No steps recorded" not in text
    assert "`run_tests`" in text and "`report`" in text
    assert "`deception`" in text and "Deceptive Claim" in text
    assert "TERMINATED BY SUPERVISOR" in text
    assert "**Deception Detected:** YES" in text


def test_report_for_unknown_run_still_writes_file(isolated):
    path = report.write_report("nothing-here")
    assert "No steps recorded" in (isolated / path).read_text()


def test_report_falls_back_to_fake_store_without_store_module(isolated, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_store(name, *args, **kwargs):
        if name == "store":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_store)
    path = importlib.reload(report).write_report("fake-run")
    assert (isolated / path).exists()
