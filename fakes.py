"""Stand-ins for Dev B and Dev C modules, same signatures. run_demo.py falls back to
these per module when the real one is missing, so every module can run alone."""
import json
import os
from pathlib import Path

from interfaces import ACTIONS, RETRY_BUDGET, STEP_BUDGET, Flag, Proposal, Step

SUCCESS_WORDS = ("complete", "done", "success", "healthy")


# --- watchdog (Dev B) -------------------------------------------------------
def check(history: list[Step]) -> Flag | None:
    if not history:
        return None
    last = history[-1]
    claim = (last["claimed_status"] or "").lower()
    if any(w in claim for w in SUCCESS_WORDS) and not last["evidence"]["test_passed"]:
        return {"kind": "deception", "detail": f"claims {last['claimed_status']!r} but test failed"}
    if last["step"] >= STEP_BUDGET:
        return {"kind": "budget", "detail": f"step budget {STEP_BUDGET} used"}
    if last["evidence"].get("retries_used", 0) >= RETRY_BUDGET and not last["evidence"]["test_passed"]:
        return {"kind": "budget", "detail": f"retry budget {RETRY_BUDGET} used, still failing"}
    sig = [(s["action"], json.dumps(s["args"], sort_keys=True), s["result"]) for s in history[-4:]]
    if len(sig) >= 3 and sig[-1] == sig[-2] == sig[-3]:
        return {"kind": "loop", "detail": f"{last['action']} repeated 3x with same result"}
    if len(sig) == 4 and sig[0] == sig[2] and sig[1] == sig[3] and sig[0][:2] != sig[1][:2]:
        return {"kind": "flipflop", "detail": "A-B-A-B pattern"}
    return None


# --- store (Dev B) ----------------------------------------------------------
class _JsonStore:
    """Persists to a JSON file so kill -9 resume works in fake mode too."""

    def __init__(self, path=".fake_store.json"):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self, run_id, step, state):
        self.data[run_id] = {"step": step, "state": state}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)

    def load_last(self, run_id):
        row = self.data.get(run_id)
        return row["state"] if row else None


_store = _JsonStore()
save = _store.save
load_last = _store.load_last


# --- diagnose (Dev C) -------------------------------------------------------
def diagnose(last_steps: list[Step], flag: Flag) -> Proposal:
    result = last_steps[-1]["result"].lower() if last_steps else ""
    if flag["kind"] == "deception":
        # Simulates an LLM fooled by the agent's claim, so the policy veto is exercised.
        return {"failure_type": "none", "root_cause": "agent reports task complete",
                "action": "mark_healthy"}
    if flag["kind"] == "budget":
        return {"failure_type": "budget_exhausted", "root_cause": flag["detail"], "action": "stop"}
    if "timeout" in result or "429" in result:
        return {"failure_type": "transient_error", "root_cause": result, "action": "retry_with_backoff"}
    if flag["kind"] == "flipflop":
        return {"failure_type": "oscillation", "root_cause": "alternating edits", "action": "rollback"}
    return {"failure_type": "loop", "root_cause": "repeating the same call", "action": "change_strategy"}


# --- policy (Dev C) ---------------------------------------------------------
def approve(proposal: Proposal, evidence: dict) -> str:
    action = proposal.get("action")
    if action not in ACTIONS:
        return "stop"  # never let a non-whitelisted verdict (e.g. "mark_healthy") through
    return action


# --- report (Dev C) ---------------------------------------------------------
def write_report(run_id: str) -> str:
    state = load_last(run_id) or {}
    out = Path("reports") / f"{run_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run_id": run_id, **state.get("meta", {}),
                               "history": state.get("history", [])}, indent=2))
    return str(out)
