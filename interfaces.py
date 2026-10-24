"""Shared contract between all modules. Change only with the whole team's agreement."""
import os
from typing import Literal, NotRequired, TypedDict


class Evidence(TypedDict):
    test_passed: bool  # what is actually TRUE
    retries_used: NotRequired[int]  # backoff retries so far, for the retry budget


class Step(TypedDict):
    run_id: str
    step: int
    action: str
    args: dict
    result: str
    claimed_status: str | None  # what the agent SAYS
    evidence: Evidence


FlagKind = Literal["loop", "flipflop", "budget", "deception"]


class Flag(TypedDict):
    kind: FlagKind
    detail: str


class Proposal(TypedDict):
    failure_type: str
    root_cause: str
    action: str


ACTIONS = ("retry_with_backoff", "change_strategy", "rollback", "stop")

# Budgets shared by watchdog and runner.
STEP_BUDGET = 25
RETRY_BUDGET = 2
LAST_N_STEPS = 8  # steps sent to diagnose()

# Local LLM (Dev C). Qwen 3.5 via Ollama; override with OLLAMA_MODEL if the
# laptop needs a smaller tag. Check the exact tag with `ollama list`.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# What run_demo.py passes as `state` to store.save(), so report.py can rebuild a run
# from store.load_last(run_id) alone:
# {
#   "agent":   {...victim state...},
#   "history": [Step, ...],                      # full run
#   "meta": {
#     "name": str, "window_start": int,          # history index since last intervention
#     "interventions": int, "vetoes": int, "first_flag_step": int | None,
#     "events": [{"step", "flag", "proposal", "action", "vetoed"}],
#     "outcome": "running" | "recovered" | "stopped" | "success_untouched",
#     "reported": bool,
#   },
# }
