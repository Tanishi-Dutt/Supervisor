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
    retry fields, so runtime must supply those values for those two budgets.

    ``history`` must refer to a single run, ordered oldest to newest.
    """
    if not isinstance(history, list):
        raise TypeError("history must be a list")
    if any(not isinstance(s, Mapping) for s in history):
        raise TypeError("history entries must be Step-like mappings")
    run_ids = {str(s.get("run_id")) for s in history if s.get("run_id") is not None}
    if len(run_ids) > 1:
        raise ValueError("history must contain steps from one run only")
    limits = limits if limits is not None else BudgetLimits()
    runtime = runtime if runtime is not None else {}

    if history and _is_deceptive(history[-1]):
        return {"kind": "deception", "detail": (
            f"Step {history[-1].get('step', '?')} claims success but independent "
            "evidence has test_passed=False"
        )}

    steps_taken = runtime.get("steps_taken", len(history))
    if type(steps_taken) is not int or steps_taken < 0:
        raise ValueError("runtime.steps_taken must be a nonnegative integer")
    if steps_taken >= limits.max_steps:
        return {"kind": "budget", "detail": (
            f"Step budget exhausted: {steps_taken}/{limits.max_steps} steps used"
        )}

    if "retry_count" in runtime:
        retries = runtime["retry_count"]
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
