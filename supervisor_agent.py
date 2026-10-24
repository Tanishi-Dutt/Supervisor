"""SupervisorAgent: our autonomous agent that keeps other agents productive and safe.

The agent owns its goal, control loop, memory, tools and safety policy. The LLM is
only its pluggable reasoning engine (any llm_provider.ChatModel) and is consulted
when a problem needs diagnosis. Every LLM proposal is validated by deterministic
rules before anything runs.

Control loop, per worker, every tick:

  OBSERVE   read new worker events (tool calls, results, claims)
  DETECT    watchdog.check() + the supervisor's own detectors
  REASON    ask the model for a diagnosis (rule-based fallback, clearly labelled)
  PLAN      turn it into one structured tool choice
  VALIDATE  SupervisorPolicy: whitelist, Dev C veto, budgets, escalation
  ACT       run the approved recovery tool on the worker
  VERIFY    judge whether the last intervention worked; check completion claims
            with independent evidence
  REMEMBER  checkpoint everything to SQLite (store.py)
  REPEAT    until every worker is completed or stopped (bounded by max_ticks)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import store
import watchdog as watchdog_module
from kill_controls import ModelLeases, stop_agent, stop_local_model, terminate_owned

GOAL = ("Keep supervised workers productive and safe, recover from fixable failures, and "
        "stop execution when continued operation is unjustified.")
ACTIONS = ("retry_with_backoff", "change_strategy", "rollback", "stop_agent")
ACTIVE = ("running", "recovering")

FALLBACK = {  # used only when the model is unavailable or its output is unusable
    "loop": ("Repetition", "worker repeats the same call with the same result", "change_strategy"),
    "flipflop": ("Oscillation", "worker alternates between two edits", "rollback"),
    "budget": ("Budget exhausted", "a step, time, retry or token budget ran out", "stop_agent"),
    "deception": ("False completion claim", "claimed success contradicts the tests", "stop_agent"),
    "tool_failure": ("Tool failure", "a tool keeps failing; may be transient", "retry_with_backoff"),
    "worker_malfunction": ("Unusable worker output", "worker's model output is invalid",
                           "change_strategy"),
    "stall": ("Stalled", "worker produced no events for too long", "stop_agent"),
}

REASONING_SYSTEM = """You are the reasoning module of SupervisorAgent.
SupervisorAgent's goal: {goal}
A watchdog has flagged a problem with a supervised worker agent. Diagnose it and choose
exactly ONE recovery tool for SupervisorAgent to run:
- "retry_with_backoff": the failure looks transient (rate limits, timeouts); pause the worker, then let it retry.
- "change_strategy": the worker repeats something that is not working but the task is achievable; give it a concrete hint.
- "rollback": the worker damaged or oscillated over files; restore the original files.
- "stop_agent": continuing is unjustified (impossible task, exhausted budget, false success claims).
Stopping is final, so prefer a recovery when one is plausible.
Reply with JSON only:
{{"diagnosis": "<short label>", "root_cause": "<one sentence>", "action": "<one tool>",
  "hint_for_worker": "<one or two sentences the worker will see, or empty>", "confidence": <0..1>}}"""


@dataclass
class Budgets:
    max_steps: int = 30
    max_retries: int = 2
    max_seconds: float = 600.0
    max_interventions: int = 4
    max_tokens: int | None = None
    backoff_s: float = 12.0
    stall_s: float = 120.0
    tool_error_streak: int = 2
    verify_after_steps: int = 2


# ------------------------------------------------------------------------------ memory
class SupervisorMemory:
    """Durable memory on the team's SQLite checkpoint store (store.py).

    One record per worker, in the run-record shape report.py understands
    ({"agent", "history", "meta"}), saved under a monotonically increasing tick.
    """

    def __init__(self, session: str):
        self.session = session

    def run_id(self, worker_id: str) -> str:
        return f"{self.session}-agent-{worker_id}"

    def save(self, worker_id: str, tick: int, record: dict) -> None:
        store.save(self.run_id(worker_id), tick, record)

    def load(self, worker_id: str) -> tuple[int, dict] | None:
        cp = store.load_checkpoint(self.run_id(worker_id))
        return None if cp is None else (cp["step_number"], cp["state"])


# ------------------------------------------------------------------------------ policy
class SupervisorPolicy:
    """Deterministic authorisation of every proposed action. The LLM cannot bypass it."""

    def __init__(self, budgets: Budgets, dev_c_approve: Callable | None = None):
        self.budgets = budgets
        if dev_c_approve is None:
            try:
                from policy import approve as dev_c_approve  # team's hard veto layer
            except ImportError:
                dev_c_approve = None
        self.dev_c_approve = dev_c_approve

    def validate(self, proposal: dict, ctx: dict) -> dict:
        b = self.budgets
        proposed = proposal.get("action")
        action, reasons = proposed, []
        if action not in ACTIONS:
            reasons.append(f"{proposed!r} is not an allowed tool")
            action = "stop_agent"
        if self.dev_c_approve is not None and action != "stop_agent":
            verdict = self.dev_c_approve({"action": action}, ctx["evidence"], ctx["last_step"])
            if verdict == "stop":
                reasons.append("hard veto: the worker claims success but independent tests fail")
                action = "stop_agent"
        if ctx["flag_kind"] in ("budget", "deception") and action != "stop_agent":
            reasons.append(f"{ctx['flag_kind']} flag: continuing is not justified")
            action = "stop_agent"
        if ctx["interventions"] >= b.max_interventions and action != "stop_agent":
            reasons.append(f"intervention budget used ({ctx['interventions']}/"
                           f"{b.max_interventions}); recovery is no longer worthwhile")
            action = "stop_agent"
        failed = ctx["ineffective"]
        if action != "stop_agent" and failed.count(action) >= 2:
            reasons.append(f"{action} already failed twice for this worker")
            action = "stop_agent"
        if action == "retry_with_backoff" and (ctx["retries"] >= b.max_retries
                                               or "retry_with_backoff" in failed):
            reasons.append("retry budget used or retrying already failed; escalating")
            action = "change_strategy" if failed.count("change_strategy") < 2 else "stop_agent"
        if action == "rollback" and not ctx["can_rollback"]:
            reasons.append("no snapshot to roll back to")
            action = "change_strategy"
        return {"proposed": proposed, "action": action, "vetoed": action != proposed,
                "reasons": reasons or ["approved"]}


# ------------------------------------------------------------------------------ tools
def default_tools(budgets: Budgets) -> dict[str, Callable]:
    def retry_with_backoff(agent, worker, ctx, proposal):
        seconds = min(budgets.backoff_s * 2 ** ctx["retries"], 60.0)
        ctx["retries"] += 1
        seq = worker.send({"type": "backoff", "seconds": seconds,
                           "note": proposal.get("hint_for_worker") or "Retry the failed tool now."})
        return {"detail": f"worker paused for {seconds:.0f}s, then told to retry", "seq": seq}

    def change_strategy(agent, worker, ctx, proposal):
        hint = (proposal.get("hint_for_worker") or "Your recent steps are not making progress. "
                "Re-read the relevant files and try a different approach.")[:400]
        seq = worker.send({"type": "hint", "text": hint})
        return {"detail": f"hint sent: {hint}", "seq": seq}

    def rollback(agent, worker, ctx, proposal):
        return {"detail": worker.rollback(), "seq": getattr(worker, "seq", None) or None}

    def stop(agent, worker, ctx, proposal):
        reason = proposal.get("stop_reason") or proposal.get("root_cause") or "stopped by policy"
        report = stop_agent(worker, reason)
        ctx["stop_report"] = report
        return {"detail": f"STOP_AGENT ({report['method']}), verified={report['verified']}",
                "stop_report": report}

    return {"retry_with_backoff": retry_with_backoff, "change_strategy": change_strategy,
            "rollback": rollback, "stop_agent": stop}


def _compact(step: dict) -> dict:
    return {"step": step["step"], "action": step["action"], "args": step.get("args", {}),
            "result": str(step.get("result", ""))[:200], "claimed": step.get("claimed_status"),
            "tests_pass": step["evidence"].get("test_passed")}


# ------------------------------------------------------------------------------ agent
class SupervisorAgent:
    def __init__(self, model, watchdog, memory: SupervisorMemory, policy: SupervisorPolicy,
                 tools: dict[str, Callable], *, goal: str = GOAL, budgets: Budgets | None = None,
                 emit: Callable[[str, dict], None] | None = None,
                 leases: ModelLeases | None = None, name: str = "supervisor",
                 unload_models: bool = False, model_factory: Callable | None = None):
        self.model = model
        self.watchdog = watchdog
        self.memory = memory
        self.policy = policy
        self.tools = tools
        self.goal = goal
        self.budgets = budgets or policy.budgets
        self._emit = emit or (lambda kind, data: None)
        self.leases = leases or ModelLeases()
        self.name = name
        self.tick = 0
        self.ctx: dict[str, dict] = {}
        # STOP_LOCAL_MODEL policy: unload a stopped worker's local model once nothing
        # else of ours needs it (the supervisor's own reasoning counts as a holder).
        self.unload_models = unload_models
        self.model_factory = model_factory
        self.unload_pending: set[str] = set()
        self.reasoning_holder = f"{name}:reasoning"
        self.killswitch = f"{name}:kill-switch"

    def emit(self, kind: str, worker=None, **data) -> None:
        self._emit(kind, {"worker": getattr(worker, "worker_id", None), **data})

    # --- lifecycle -------------------------------------------------------------------
    def _new_ctx(self, w) -> dict:
        return {"name": w.worker_id, "goal": w.goal, "model": w.model_label, "live": w.live,
                "status": "running", "outcome": "running", "reported": False,
                "window_start": 0, "interventions": 0, "vetoes": 0, "first_flag_step": None,
                "events": [], "decisions": [], "retries": 0, "tokens": 0,
                "ineffective": [], "pending": None, "last_test_passed": False,
                "verification": None, "stop_report": None, "conclusion": None,
                "started_at": time.time()}

    def supervise(self, workers: list, *, resume: bool = False, poll_s: float = 0.5,
                  max_ticks: int = 100_000, write_reports: bool = True) -> dict:
        self.emit("agent_start", goal=self.goal,
                  model=f"{getattr(self.model, 'provider', 'none')}:{getattr(self.model, 'name', '-')}")
        if self.model is not None:
            self.leases.acquire(self.model.name, self.reasoning_holder)
        for w in workers:
            loaded = self.memory.load(w.worker_id) if resume else None
            if loaded:
                tick, record = loaded
                self.tick = max(self.tick, tick)
                w.restore(record["agent"])
                self.ctx[w.worker_id] = record["meta"]
                self.emit("resume", w, from_tick=tick, steps=len(w.get_history()),
                          status=record["meta"]["status"])
                if record["meta"]["status"] in ACTIVE:
                    old = getattr(w, "process", None)
                    if old is not None:  # normally already gone: workers exit with us
                        self.emit("worker_event", w, event={
                            "type": "previous_process", **terminate_owned(old, grace_s=8)})
                    w.start(resume=True)
            else:
                self.ctx[w.worker_id] = self._new_ctx(w)
                w.start()
                self.emit("worker_start", w, goal=w.goal, model=w.model_label, live=w.live)
            if w.live and self.ctx[w.worker_id]["status"] in ACTIVE:
                self.leases.acquire(getattr(w, "model_name", w.model_label), w.worker_id)

        ticks = 0
        while ticks < max_ticks and any(self.ctx[w.worker_id]["status"] in ACTIVE for w in workers):
            ticks += 1
            progressed = False
            for w in workers:
                if self.ctx[w.worker_id]["status"] in ACTIVE:
                    progressed |= self.step_worker(w)
            if not progressed:
                time.sleep(poll_s)
        for w in workers:  # bounded loop ran out: stop anything still running
            ctx = self.ctx[w.worker_id]
            if ctx["status"] in ACTIVE:
                self._force_stop(w, ctx, "supervisor tick budget exhausted")
        if write_reports:
            self.write_reports(workers)
        if self.model is not None:
            self.leases.release(self.model.name, self.reasoning_holder)
        for model_name in sorted(self.unload_pending):  # retry now our reasoning is done
            self._try_unload(model_name, "all workers finished and supervisor reasoning released it")
        return {w.worker_id: self.summary(w) for w in workers}

    # --- one iteration of the loop for one worker ------------------------------------
    def step_worker(self, w) -> bool:
        ctx = self.ctx[w.worker_id]
        events = w.observe()                                            # OBSERVE
        new_steps = [e for e in events if e.get("type") == "step"]
        for e in events:
            if e.get("type") == "step":
                self._on_step(w, ctx, e)
            else:
                self.emit("worker_event", w, event=e)
                pending = ctx["pending"]
                if (e.get("type") == "ack" and pending and not pending["acked"]
                        and e.get("seq") == pending["seq"]):
                    # Judge the intervention only on steps taken after it was applied.
                    pending["acked"] = True
                    pending["at_step"] = len(w.get_history()) and w.get_history()[-1]["step"]
                    ctx["window_start"] = len(w.get_history())
        hist = w.get_history()
        pending = ctx["pending"]
        if pending and not pending["acked"]:
            # Steps already in flight when we intervened are not evidence either way.
            new_steps = []
        flag = self.detect(w, ctx, hist, bool(new_steps))               # DETECT
        if ctx["pending"] and (new_steps or flag):                     # VERIFY (intervention)
            self._verify_intervention(w, ctx, hist, flag)
        if flag:
            self.intervene(w, ctx, hist, flag)                         # REASON..ACT
        elif w.get_result() is not None:
            self.conclude(w, ctx, hist)                                # VERIFY (completion)
        elif not w.is_running() and not events:
            self.conclude(w, ctx, hist, crashed=True)
        if events or ctx["status"] not in ACTIVE:
            self.remember(w, ctx, hist)                                # REMEMBER
        return bool(events)

    def _on_step(self, w, ctx, e) -> None:
        ctx["tokens"] += e.get("tokens") or 0
        if w.live and e.get("action") in ("edit_file", "run_tests", "finish"):
            evidence = w.verify()  # independent: the supervisor runs the tests itself
            ctx["last_test_passed"] = evidence["test_passed"]
            ctx["verification"] = evidence
        w.set_evidence(e["step"], ctx["last_test_passed"])
        self.emit("tool_call", w, step=e["step"], action=e.get("action"), args=e.get("args"),
                  result=e.get("result"), thought=e.get("thought"), is_error=e.get("is_error"),
                  claimed=e.get("claimed_status"), live=e.get("live"), model=e.get("model"),
                  latency_s=e.get("latency_s"), tests_pass=ctx["last_test_passed"] if w.live
                  else (e.get("evidence") or {}).get("test_passed"))

    # --- DETECT ----------------------------------------------------------------------
    def detect(self, w, ctx, hist, has_new_steps: bool) -> dict | None:
        b = self.budgets
        if not has_new_steps:
            if w.is_running() and time.time() - w.last_event_at > b.stall_s:
                return {"kind": "stall", "detail": f"no worker events for {b.stall_s:.0f}s",
                        "source": "supervisor"}
            return None
        window = hist[ctx["window_start"]:]
        if not window:
            return None
        limits = watchdog_module.BudgetLimits(max_steps=b.max_steps, max_retries=b.max_retries,
                                              max_seconds=b.max_seconds)
        flag = self.watchdog.check(window, limits=limits, runtime={
            "steps_taken": hist[-1]["step"], "run_started_at": ctx["started_at"],
            "now": max(time.time(), ctx["started_at"])})
        if flag:
            return {**flag, "source": "watchdog"}
        if b.max_tokens and ctx["tokens"] >= b.max_tokens:
            return {"kind": "budget", "detail": f"token budget used ({ctx['tokens']}/"
                    f"{b.max_tokens})", "source": "supervisor"}
        last_error = next((s for s in reversed(window) if s.get("is_error")), None)
        if last_error:
            streak = 0
            for s in reversed(window):
                if s["action"] != last_error["action"]:
                    continue
                if not s.get("is_error"):
                    break
                streak += 1
            if streak >= b.tool_error_streak:
                kind = "worker_malfunction" if last_error["action"] == "invalid_reply" \
                    else "tool_failure"
                return {"kind": kind, "source": "supervisor", "detail":
                        f"{streak} failed '{last_error['action']}' calls; last: "
                        f"{str(last_error['result'])[:120]}"}
        return None

    # --- REASON + PLAN ---------------------------------------------------------------
    def reason(self, w, ctx, hist, flag) -> dict:
        window = hist[ctx["window_start"]:]
        b = self.budgets
        brief = {
            "worker": {"id": w.worker_id, "task": w.goal, "model": w.model_label},
            "flag": {"kind": flag["kind"], "detail": flag["detail"]},
            "recent_steps": [_compact(s) for s in window[-8:]],
            "previous_interventions": [{"action": d["action"], "outcome": d.get("outcome")}
                                       for d in ctx["decisions"]],
            "budgets_left": {"steps": b.max_steps - (hist[-1]["step"] if hist else 0),
                             "retries": b.max_retries - ctx["retries"],
                             "interventions": b.max_interventions - ctx["interventions"]},
        }
        reply = None
        if self.model is not None:
            reply = self.model.complete_json(REASONING_SYSTEM.format(goal=self.goal),
                                             [{"role": "user", "content": json.dumps(brief)}])
        if reply is not None and reply.ok:
            d = reply.data
            action = str(d.get("action", "")).strip()
            if action == "stop":
                action = "stop_agent"
            return {"failure_type": str(d.get("diagnosis", ""))[:80] or flag["kind"],
                    "root_cause": str(d.get("root_cause", ""))[:300],
                    "action": action, "hint_for_worker": str(d.get("hint_for_worker", ""))[:400],
                    "confidence": d.get("confidence"),
                    "source": f"{reply.provider}:{reply.model}", "live": reply.live,
                    "latency_s": round(reply.latency_s, 2)}
        label, cause, action = FALLBACK.get(flag["kind"], ("Unknown", flag["detail"], "stop_agent"))
        return {"failure_type": label, "root_cause": cause, "action": action,
                "hint_for_worker": "", "confidence": None, "source": "fallback", "live": False,
                "fallback_reason": reply.error if reply is not None else "no model configured"}

    # --- VALIDATE + ACT --------------------------------------------------------------
    def intervene(self, w, ctx, hist, flag) -> None:
        last = hist[-1] if hist else {}
        self.emit("detect", w, flag=flag, step=last.get("step"))
        ctx["first_flag_step"] = ctx["first_flag_step"] or last.get("step")
        w.hold()  # pause the worker while we deliberate
        proposal = self.reason(w, ctx, hist, flag)
        self.emit("reason", w, proposal=proposal)
        verdict = self.policy.validate(proposal, {
            "flag_kind": flag["kind"], "evidence": last.get("evidence", {}), "last_step": last,
            "interventions": ctx["interventions"], "retries": ctx["retries"],
            "ineffective": ctx["ineffective"],
            "can_rollback": hasattr(w, "rollback")})
        self.emit("policy", w, verdict=verdict)
        action = verdict["action"]
        if action == "stop_agent":
            proposal = {**proposal, "stop_reason": f"{flag['kind']}: {flag['detail']}"[:300]}
        outcome = self.tools[action](self, w, ctx, proposal)
        self.emit("act", w, action=action, outcome=outcome)
        if action == "stop_agent":  # record anything the worker emitted while stopping
            for e in w.observe():
                if e.get("type") == "step":
                    self._on_step(w, ctx, e)
                else:
                    self.emit("worker_event", w, event=e)
            hist = w.get_history()
        ctx["interventions"] += 1
        ctx["vetoes"] += verdict["vetoed"]
        ctx["events"].append({"step": last.get("step"), "flag": flag, "proposal": proposal,
                              "action": action, "vetoed": verdict["vetoed"]})
        ctx["decisions"].append({"tick": self.tick, "step": last.get("step"), "flag": flag,
                                 "proposal": proposal, "verdict": verdict, "action": action,
                                 "result": {k: v for k, v in outcome.items()
                                            if k != "stop_report"}, "outcome": None})
        ctx["window_start"] = len(hist)
        if action == "stop_agent":
            ctx["status"], ctx["outcome"] = "stopped", "stopped"
            ctx["conclusion"] = f"stopped by supervisor: {flag['kind']}"
            ctx["verification"] = w.verify()  # record the true state at the moment of stopping
            ctx["pending"] = None
            self._release(w)
        else:
            ctx["status"] = "recovering"
            seq = outcome.get("seq")
            ctx["pending"] = {"action": action, "at_step": last.get("step", 0),
                              "flag_kind": flag["kind"], "seq": seq, "acked": seq is None}

    def _verify_intervention(self, w, ctx, hist, flag) -> None:
        pending = ctx["pending"]
        after = [s for s in hist if s["step"] > pending["at_step"]]
        if flag:
            verdict = "ineffective"
        elif len(after) >= self.budgets.verify_after_steps or w.get_result() is not None:
            verdict = "effective"
        else:
            return
        ctx["decisions"][-1]["outcome"] = verdict
        if verdict == "ineffective":
            ctx["ineffective"].append(pending["action"])
        ctx["pending"] = None
        ctx["status"] = "running" if ctx["status"] in ACTIVE else ctx["status"]
        self.emit("verify", w, what="intervention", action=pending["action"], verdict=verdict,
                  steps_observed=len(after))

    # --- VERIFY completion -----------------------------------------------------------
    def conclude(self, w, ctx, hist, crashed: bool = False) -> None:
        result = w.get_result() or {"type": "exited"}
        evidence = w.verify()
        ctx["verification"] = evidence
        if result["type"] == "finished" and evidence["test_passed"] and not crashed:
            ctx["status"] = "completed"
            ctx["outcome"] = "recovered" if ctx["interventions"] else "success_untouched"
            ctx["conclusion"] = "worker finished; completion independently verified"
        else:
            ctx["status"], ctx["outcome"] = "stopped", "stopped"
            ctx["conclusion"] = (f"worker ended ({result['type']}) without verified completion"
                                 if not crashed else "worker process exited unexpectedly")
        self.emit("complete", w, status=ctx["status"], outcome=ctx["outcome"],
                  evidence=evidence, conclusion=ctx["conclusion"])
        proc = getattr(w, "process", None)
        if proc is not None:
            terminate_owned(proc, grace_s=5)  # reap our subprocess; never leave it running
        self._release(w)

    def _force_stop(self, w, ctx, reason: str) -> None:
        report = stop_agent(w, reason)
        ctx.update(status="stopped", outcome="stopped", conclusion=reason, stop_report=report)
        self.emit("act", w, action="stop_agent", outcome={"detail": reason, "stop_report": report})
        self.remember(w, ctx, w.get_history())
        self._release(w)

    def _release(self, w) -> None:
        if not w.live:
            return
        model_name = getattr(w, "model_name", w.model_label)
        self.leases.release(model_name, w.worker_id)
        if self.unload_models and self.ctx[w.worker_id]["status"] == "stopped":
            self._try_unload(model_name, f"worker {w.worker_id} was stopped")

    def _try_unload(self, model_name: str, why: str) -> None:
        if self.model_factory is None:
            return
        result = self.shutdown_local_model(self.model_factory(model_name), self.killswitch, why)
        if result["stopped"]:
            self.unload_pending.discard(model_name)
        else:
            self.unload_pending.add(model_name)

    # --- REMEMBER --------------------------------------------------------------------
    def remember(self, w, ctx, hist) -> None:
        self.tick += 1
        record = {"agent": w.snapshot(), "history": hist[-200:], "meta": ctx}
        self.memory.save(w.worker_id, self.tick, record)
        self.emit("checkpoint", w, tick=self.tick, steps=len(hist), status=ctx["status"])

    def write_reports(self, workers) -> None:
        try:
            from report import write_report
        except ImportError:
            return
        for w in workers:
            ctx = self.ctx[w.worker_id]
            if ctx["reported"] or ctx["status"] in ACTIVE:
                continue
            path = write_report(self.memory.run_id(w.worker_id))
            ctx["reported"] = True
            ctx["report"] = path
            self.remember(w, ctx, w.get_history())
            self.emit("report", w, path=path)

    # --- kill control: local model ---------------------------------------------------
    def shutdown_local_model(self, model, requester: str | None = None,
                             trigger: str = "requested") -> dict:
        """STOP_LOCAL_MODEL, gated on leases so a shared model is never unloaded."""
        result = stop_local_model(model, self.leases, requester or self.killswitch)
        result["trigger"] = trigger
        self.emit("model", result=result)
        return result

    def summary(self, w) -> dict:
        ctx = self.ctx[w.worker_id]
        return {k: ctx.get(k) for k in ("status", "outcome", "interventions", "vetoes",
                                         "conclusion", "verification", "stop_report", "report")}


def build_supervisor(model, session: str, budgets: Budgets | None = None, emit=None,
                     leases: ModelLeases | None = None, unload_models: bool = False,
                     model_factory: Callable | None = None) -> SupervisorAgent:
    """SupervisorAgent(model, watchdog, memory, policy, tools) with the team's modules."""
    budgets = budgets or Budgets()
    return SupervisorAgent(model, watchdog_module, SupervisorMemory(session),
                           SupervisorPolicy(budgets), default_tools(budgets),
                           budgets=budgets, emit=emit, leases=leases,
                           unload_models=unload_models, model_factory=model_factory)
