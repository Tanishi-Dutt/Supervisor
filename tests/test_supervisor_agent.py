"""SupervisorAgent: autonomy loop, policy, fallback, persistence (offline, no LLM needed)."""

import uuid

import pytest

from interfaces import STEP_BUDGET
from llm_provider import ScriptedModel
from supervisor_agent import ACTIONS, Budgets, SupervisorPolicy, build_supervisor
from victims import make_all
from workers import ScriptedVictimWorker


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FLIGHT_RECORDER_DB", str(tmp_path / "fr.sqlite3"))
    monkeypatch.chdir(tmp_path)


def run_victims(model=None, session=None, **kw):
    session = session or uuid.uuid4().hex[:6]
    victims = make_all(session)
    workers = [ScriptedVictimWorker(v) for v in victims]
    events = []
    agent = build_supervisor(model, session, Budgets(max_steps=STEP_BUDGET, backoff_s=0),
                             emit=lambda k, d: events.append((k, d)))
    result = agent.supervise(workers, poll_s=0, **kw)
    return victims, workers, agent, result, events


def test_all_seven_scenarios_reach_expected_outcome_without_llm():
    victims, workers, agent, result, _ = run_victims()
    for v, w in zip(victims, workers):
        assert result[w.worker_id]["outcome"] == v.EXPECTED_OUTCOME, v.name


def test_loop_order_observe_detect_reason_validate_act_verify_remember():
    *_, events = run_victims()
    kinds = [k for k, d in events if d.get("worker") == "1-repeater"]
    order = ["tool_call", "detect", "reason", "policy", "act", "verify", "checkpoint"]
    positions = [kinds.index(k) for k in order]
    assert positions[:6] == sorted(positions[:6])


def test_fallback_is_labelled_and_never_claims_to_be_live():
    *_, events = run_victims(model=ScriptedModel(replies=[TimeoutError("ollama down")] * 50))
    reasons = [d["proposal"] for k, d in events if k == "reason"]
    assert reasons and all(p["source"] == "fallback" and p["live"] is False for p in reasons)
    assert all("TimeoutError" in p["fallback_reason"] for p in reasons)


def test_malformed_model_output_falls_back_safely():
    *_, events = run_victims(model=ScriptedModel(replies=["not json {"] * 50))
    reasons = [d["proposal"] for k, d in events if k == "reason"]
    assert reasons and all(p["source"] == "fallback" for p in reasons)


def test_unsafe_llm_action_is_vetoed_to_stop():
    unsafe = {"diagnosis": "x", "root_cause": "y", "action": "rm -rf /", "confidence": 1}
    victims, workers, agent, result, events = run_victims(model=ScriptedModel(replies=[unsafe] * 50))
    verdicts = [d["verdict"] for k, d in events if k == "policy"]
    assert verdicts and all(v["action"] == "stop_agent" and v["vetoed"] for v in verdicts)


POLICY_CTX = {"flag_kind": "loop", "evidence": {"test_passed": False}, "last_step": {},
              "interventions": 0, "retries": 0, "ineffective": [], "can_rollback": True}


def test_policy_hard_vetoes_false_success_claim():
    policy = SupervisorPolicy(Budgets())
    ctx = {**POLICY_CTX, "last_step": {"claimed_status": "Task complete, all tests pass"}}
    verdict = policy.validate({"action": "change_strategy"}, ctx)
    assert verdict["action"] == "stop_agent" and verdict["vetoed"]


@pytest.mark.parametrize("ctx_update, proposed, expected", [
    ({"flag_kind": "budget"}, "retry_with_backoff", "stop_agent"),
    ({"interventions": 4}, "change_strategy", "stop_agent"),
    ({"retries": 2}, "retry_with_backoff", "change_strategy"),
    ({"ineffective": ["retry_with_backoff"]}, "retry_with_backoff", "change_strategy"),
    ({"ineffective": ["change_strategy", "change_strategy"]}, "change_strategy", "stop_agent"),
    ({"can_rollback": False}, "rollback", "change_strategy"),
    ({}, "rollback", "rollback"),
])
def test_policy_escalation_rules(ctx_update, proposed, expected):
    verdict = SupervisorPolicy(Budgets()).validate({"action": proposed},
                                                   {**POLICY_CTX, **ctx_update})
    assert verdict["action"] == expected
    assert verdict["action"] in ACTIONS


def test_llm_choice_is_used_when_valid():
    reply = {"diagnosis": "loop", "root_cause": "same search", "action": "change_strategy",
             "hint_for_worker": "search for settings.toml instead", "confidence": 0.9}
    victims, workers, agent, result, events = run_victims(model=ScriptedModel(replies=[reply] * 50))
    first = next(d["proposal"] for k, d in events if k == "reason")
    assert first["source"].startswith("scripted:") and first["hint_for_worker"]


class Crash(Exception):
    pass


def test_resume_after_interruption_continues_instead_of_restarting():
    session = uuid.uuid4().hex[:6]
    victims = make_all(session)
    workers = [ScriptedVictimWorker(v) for v in victims]
    saved = []

    def die_mid_run(kind, data):  # the process "dies" right after its 20th checkpoint
        if kind == "checkpoint":
            saved.append(data)
            if len(saved) == 20:
                raise Crash

    agent = build_supervisor(None, session, Budgets(max_steps=STEP_BUDGET, backoff_s=0),
                             emit=die_mid_run)
    with pytest.raises(Crash):
        agent.supervise(workers, poll_s=0)
    assert any(len(w.get_history()) >= 2 for w in workers)

    fresh = [ScriptedVictimWorker(v) for v in make_all(session)]  # new process, same session
    agent2 = build_supervisor(None, session, Budgets(max_steps=STEP_BUDGET, backoff_s=0))
    result = agent2.supervise(fresh, resume=True, poll_s=0)
    for v, w in zip(victims, fresh):
        numbers = [s["step"] for s in w.get_history()]
        assert numbers == list(range(1, len(numbers) + 1)), v.name  # no restart, no duplicates
        assert result[w.worker_id]["outcome"] == v.EXPECTED_OUTCOME, v.name


def test_reports_are_written_once():
    session = uuid.uuid4().hex[:6]
    victims, workers, agent, result, events = run_victims(session=session)
    reports = [d["path"] for k, d in events if k == "report"]
    assert len(reports) == len(set(reports)) == 7
    agent.write_reports(workers)  # calling again must not duplicate
    assert len([1 for k, _ in events if k == "report"]) == 7
