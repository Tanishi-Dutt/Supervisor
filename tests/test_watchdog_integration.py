"""Regression tests for Dev A's window + interfaces contract."""
import sys
from types import SimpleNamespace
from watchdog import check


def step(n, *, retries=None):
    return {"run_id": "r", "step": n, "action": f"different_{n}",
            "args": {}, "result": "ok", "claimed_status": None,
            "evidence": {} if retries is None else {"retries_used": retries}}


def test_shared_step_budget_applies_to_post_intervention_window(monkeypatch):
    monkeypatch.setitem(sys.modules, "interfaces", SimpleNamespace(STEP_BUDGET=25, RETRY_BUDGET=2))
    assert check([step(24)]) is None
    flag = check([step(25)])
    assert flag["kind"] == "budget"
    assert "25/25" in flag["detail"]


def test_shared_retry_budget_can_be_read_from_evidence(monkeypatch):
    monkeypatch.setitem(sys.modules, "interfaces", SimpleNamespace(STEP_BUDGET=25, RETRY_BUDGET=2))
    assert check([step(4, retries=1)]) is None
    flag = check([step(5, retries=2)])
    assert flag["kind"] == "budget"
    assert "Retry budget" in flag["detail"]


def test_real_interfaces_module_wires_actual_project_budgets():
    """No monkeypatch: exercises the real import of the team's interfaces.py,
    matching exactly what run_demo.py gets when it calls check(window) with
    no explicit limits. Pins STEP_BUDGET/RETRY_BUDGET so a change to
    interfaces.py is caught here instead of only surfacing in the live demo."""
    import interfaces
    assert check([step(interfaces.STEP_BUDGET - 1)]) is None
    flag = check([step(interfaces.STEP_BUDGET)])
    assert flag["kind"] == "budget"
    assert check([step(5, retries=interfaces.RETRY_BUDGET - 1)]) is None
    flag = check([step(5, retries=interfaces.RETRY_BUDGET)])
    assert flag["kind"] == "budget"
    assert "Retry budget" in flag["detail"]


def test_malformed_evidence_does_not_crash_retry_lookup():
    """A Step with evidence explicitly set to None (not a dict) must not raise;
    it simply means no retry count is available from evidence."""
    bad_step = {"run_id": "r", "step": 1, "action": "x", "args": {}, "result": "ok",
                "claimed_status": None, "evidence": None}
    assert check([bad_step]) is None
