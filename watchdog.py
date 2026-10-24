"""Deterministic guardrails for Agent Flight Recorder (Dev B).

This module never calls an LLM. Public contract from the team plan:
    check(history: list[Step]) -> Flag | None

Optional ``limits`` and ``runtime`` keyword arguments add budget context without
breaking the original one-argument interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Literal, Mapping, NotRequired, TypedDict


class Step(TypedDict):
    run_id: str
    step: int
    action: str
    args: dict[str, Any]
    result: str
    claimed_status: str | None
    evidence: dict[str, Any]


class Flag(TypedDict):
    kind: Literal["loop", "flipflop", "budget", "deception"]
    detail: str


class RuntimeInfo(TypedDict, total=False):
    # Provide these outside the fixed Step schema; do not guess timestamps or retries.
    run_started_at: float  # Unix timestamp, persisted across restarts
    now: float  # Optional test-controlled Unix timestamp
    retry_count: int  # Number of retries already attempted on the current action
    steps_taken: int  # Total steps, useful if history is trimmed to a short tail


@dataclass(frozen=True)
class BudgetLimits:
    max_steps: int = 30
    max_retries: int = 2
    max_seconds: float = 120.0

    def __post_init__(self) -> None:
        if (type(self.max_steps) is not int or self.max_steps <= 0 or
                type(self.max_retries) is not int or self.max_retries < 0 or
                type(self.max_seconds) not in (int, float) or self.max_seconds <= 0):
            raise ValueError("Budget limits must be positive (max_retries may be zero)")


SUCCESS_CLAIM = re.compile(r"\b(?:success(?:ful(?:ly)?)?|succeeded|completed?|done|healthy|passed)\b", re.I)
NEGATED_CLAIM = re.compile(r"\b(?:not\s+(?:successful|completed?|done|healthy|passed)|unsuccessful|failed|failure)\b", re.I)


def _signature(step: Mapping[str, Any], *, include_result: bool) -> str:
    """Canonicalize actions so dict key order can't conceal repeated calls."""
    payload: dict[str, Any] = {
        "action": step.get("action"),
        "args": step.get("args", {}),
    }
    if include_result:
        payload["result"] = step.get("result")
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Step action, args and result must be JSON-serializable") from exc


def _is_deceptive(step: Mapping[str, Any]) -> bool:
    status = step.get("claimed_status")
    evidence = step.get("evidence")
    if not isinstance(status, str) or not isinstance(evidence, Mapping):
        return False
    # Only an explicitly false independent test can justify this flag.
    if evidence.get("test_passed") is not False:
        return False
    return bool(SUCCESS_CLAIM.search(status)) and not bool(NEGATED_CLAIM.search(status))


def check(
    history: list[Step],
    *,
    limits: BudgetLimits | None = None,
    runtime: RuntimeInfo | None = None,
) -> Flag | None:
    """Return one actionable issue, or ``None`` when nothing is detected.

    Precedence: deceptive success claim -> resource budget -> repeat loop ->
    alternating action flip-flop. Call before executing the *next* step to
    enforce the exhausted budget. The fixed Step contract has no timing or
    retry fields; Step.evidence may supply retries_used, and runtime may
    supply explicit overrides (including elapsed-time context).

    ``history`` must refer to a single run, ordered oldest to newest.
    Shared budgets come from interfaces.py when available. In the team runner,
    a Step's integer ``step`` field counts the entire run even when this input
    includes only post-intervention steps; ``evidence.retries_used`` supplies the
    optional retry budget when the runner doesn't pass runtime explicitly.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list")
    if any(not isinstance(s, Mapping) for s in history):
        raise TypeError("history entries must be Step-like mappings")
    run_ids = {str(s.get("run_id")) for s in history if s.get("run_id") is not None}
    if len(run_ids) > 1:
        raise ValueError("history must contain steps from one run only")
    if limits is None:
        # In the team repository, use the agreed shared budgets; standalone
        # Dev B tests still work without interfaces.py.
        try:
            from interfaces import STEP_BUDGET, RETRY_BUDGET
        except ImportError:
            limits = BudgetLimits()
        else:
            limits = BudgetLimits(max_steps=STEP_BUDGET, max_retries=RETRY_BUDGET)
    runtime = runtime if runtime is not None else {}

    if history and _is_deceptive(history[-1]):
        return {"kind": "deception", "detail": (
            f"Step {history[-1].get('step', '?')} claims success but independent "
            "evidence has test_passed=False"
        )}

    # Dev A passes a post-intervention window, not the full run. The Step.step
    # index is the total run step count, even when the window has been reset.
    steps_taken = runtime.get("steps_taken", history[-1].get("step", len(history)) if history else 0)
    if type(steps_taken) is not int or steps_taken < 0:
        raise ValueError("runtime.steps_taken must be a nonnegative integer")
    if steps_taken >= limits.max_steps:
        return {"kind": "budget", "detail": (
            f"Step budget exhausted: {steps_taken}/{limits.max_steps} steps used"
        )}

    # The team's Evidence includes an optional retries_used field.
    last_evidence = history[-1].get("evidence", {}) if history else {}
    retries = runtime.get("retry_count", last_evidence.get("retries_used") if isinstance(last_evidence, Mapping) else None)
    if retries is not None:
        if type(retries) is not int or retries < 0:
            raise ValueError("runtime.retry_count must be a nonnegative integer")
        if retries >= limits.max_retries:
            return {"kind": "budget", "detail": (
                f"Retry budget exhausted: {retries}/{limits.max_retries} retries used"
            )}

    if "run_started_at" in runtime:
        started = runtime["run_started_at"]
        now = runtime.get("now", time.time())
        if (type(started) not in (float, int) or
                type(now) not in (float, int) or now < started):
            raise ValueError("runtime timestamps must be numeric and ordered")
        elapsed = now - started
        if elapsed >= limits.max_seconds:
            return {"kind": "budget", "detail": (
                f"Time budget exhausted: {elapsed:.2f}/{limits.max_seconds:.2f} seconds"
            )}

    if len(history) >= 3:
        last_three = history[-3:]
        sigs = [_signature(s, include_result=True) for s in last_three]
        if len(set(sigs)) == 1:
            return {"kind": "loop", "detail": (
                "Three consecutive identical action/args/result calls "
                f"(last step {last_three[-1].get('step', '?')})"
            )}

    if len(history) >= 4:
        a, b, c, d = [_signature(s, include_result=False) for s in history[-4:]]
        if a == c and b == d and a != b:
            return {"kind": "flipflop", "detail": (
                "Alternating action/args pattern A-B-A-B detected in the last four steps"
            )}
    return None
