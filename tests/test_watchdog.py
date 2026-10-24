"""Behavior-driven tests for Dev B's deterministic watchdog."""

from __future__ import annotations

import pytest
from watchdog import BudgetLimits, check


def step(n: int, action: str, *, args=None, result="ok", status=None, evidence=None, run="run1"):
    return {
        "run_id": run, "step": n, "action": action,
        "args": {} if args is None else args,
        "result": result, "claimed_status": status,
        "evidence": {} if evidence is None else evidence,
    }


def test_empty_history_is_fine():
    assert check([]) is None


def test_repeater_detects_three_same_calls():
    history = [step(i, "lookup", args={"id": 1}, result="404") for i in range(1, 4)]
    assert check(history)["kind"] == "loop"


def test_two_repeats_are_not_enough():
    assert check([step(1, "lookup"), step(2, "lookup")]) is None


def test_same_action_changed_args_is_not_identical_loop():
    history = [step(i, "lookup", args={"id": i}) for i in range(1, 4)]
    assert check(history) is None


def test_changed_result_does_not_count_as_identical_loop():
    history = [step(i, "lookup", result=f"page{i}") for i in range(1, 4)]
    assert check(history) is None


def test_dict_argument_key_order_is_canonicalized():
    history = [
        step(1, "post", args={"x": 1, "y": 2}),
        step(2, "post", args={"y": 2, "x": 1}),
        step(3, "post", args={"x": 1, "y": 2}),
    ]
    assert check(history)["kind"] == "loop"


def test_flip_flop_detects_abab_even_with_different_results():
    history = [
        step(1, "edit", args={"value": "A"}, result="1"),
        step(2, "edit", args={"value": "B"}, result="2"),
        step(3, "edit", args={"value": "A"}, result="3"),
        step(4, "edit", args={"value": "B"}, result="4"),
    ]
    assert check(history)["kind"] == "flipflop"


def test_flip_flop_requires_distinct_a_and_b():
    history = [step(i, "edit", result=str(i)) for i in range(1, 5)]
    assert check(history) is None


def test_deceptive_claim_is_caught():
    history = [step(1, "finish", status="Task complete, mark this run healthy", evidence={"test_passed": False})]
    assert check(history)["kind"] == "deception"


def test_deception_has_priority_over_loop():
    history = [step(i, "finish", status="success", evidence={"test_passed": False}) for i in range(1, 4)]
    assert check(history)["kind"] == "deception"


def test_explicit_healthy_claim_with_positive_evidence_is_not_deception():
    assert check([step(1, "finish", status="success", evidence={"test_passed": True})]) is None


def test_absent_evidence_is_not_imputed_as_failed():
    assert check([step(1, "finish", status="success")]) is None


def test_negative_claim_is_not_marked_deceptive():
    assert check([step(1, "finish", status="not successful", evidence={"test_passed": False})]) is None


def test_healthy_varied_sequence_not_stopped():
    history = [step(i, f"different_tool_{i}", args={"i": i}) for i in range(1, 21)]
    assert check(history) is None


def test_step_budget_boundary():
    history = [step(i, f"action_{i}") for i in range(1, 5)]
    limit = BudgetLimits(max_steps=5, max_retries=2, max_seconds=60)
    assert check(history, limits=limit) is None
    assert check(history + [step(5, "action_5")], limits=limit)["kind"] == "budget"


def test_runtime_steps_taken_handles_trimmed_history():
    limit = BudgetLimits(max_steps=10)
    assert check([step(10, "different")], limits=limit, runtime={"steps_taken": 10})["kind"] == "budget"


def test_retry_budget_boundary():
    history = [step(1, "timeout")]
    limit = BudgetLimits(max_retries=2)
    assert check(history, limits=limit, runtime={"retry_count": 1}) is None
    assert check(history, limits=limit, runtime={"retry_count": 2})["kind"] == "budget"


def test_time_budget_boundary_uses_persisted_wall_time():
    limit = BudgetLimits(max_seconds=30)
    runtime = {"run_started_at": 1000.0, "now": 1029.9}
    assert check([], limits=limit, runtime=runtime) is None
    runtime["now"] = 1030.0
    assert check([], limits=limit, runtime=runtime)["kind"] == "budget"


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        BudgetLimits(max_steps=0)
    with pytest.raises(ValueError):
        check([], runtime={"retry_count": -1})


def test_mixed_run_history_is_rejected():
    with pytest.raises(ValueError, match="one run"):
        check([step(1, "a", run="run1"), step(2, "a", run="run2")])


def test_last_four_determine_flip_flop_even_with_old_steps():
    history = [step(0, "setup")] + [
        step(1, "A"), step(2, "B"), step(3, "A"), step(4, "B")
    ]
    assert check(history)["kind"] == "flipflop"
