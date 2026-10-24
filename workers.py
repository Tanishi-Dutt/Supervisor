"""Worker adapters: one protocol for every supervised agent, whatever model drives it.

SupervisorAgent only sees this interface and the team's Step format
(interfaces.Step), so it does not care whether a worker runs Qwen, Claude, GPT, a
script, or something else. Implemented:

* OllamaProcessWorker - LIVE: a real LLM worker in its own subprocess
                        (worker_process.py) with real tools in a sandbox.
* ScriptedVictimWorker - SIMULATED: wraps one of the seven scripted victims.

A hosted-model worker would implement the same methods; its stop path is
request_cancel() (cancel our in-flight request and loop), not a model unload.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from kill_controls import OwnedProcess, ProcessRegistry, is_alive, new_token
from sandbox_tools import TASK_GOAL, create_task, restore_snapshot

ROOT = Path(__file__).resolve().parent


class WorkerAdapter(Protocol):
    worker_id: str
    model_label: str      # e.g. "ollama:qwen3.5:9b" or "scripted"
    live: bool            # True when a real model drives the worker
    goal: str

    def start(self, resume: bool = False) -> None: ...
    def observe(self) -> list[dict]: ...            # new events since the last call
    def get_history(self) -> list[dict]: ...        # Steps (interfaces.Step)
    def send(self, command: dict) -> None: ...      # recovery commands
    def rollback(self) -> str: ...
    def revoke(self, reason: str) -> None: ...      # reject all future tool calls
    def request_cancel(self, reason: str) -> None: ...
    def is_running(self) -> bool: ...
    def get_result(self) -> dict | None: ...
    def verify(self) -> dict: ...                   # independent completion evidence
    def snapshot(self) -> dict: ...                 # adapter state for checkpoints
    def restore(self, data: dict) -> None: ...


def _to_step(run_id: str, record: dict, test_passed: bool | None) -> dict:
    return {"run_id": run_id, "step": record["step"], "action": record.get("action", "?"),
            "args": record.get("args", {}), "result": record.get("result", ""),
            "claimed_status": record.get("claimed_status"),
            "evidence": {"test_passed": bool(test_passed)},
            # extra fields for the supervisor/UI; ignored by watchdog and report
            "thought": record.get("thought", ""), "is_error": record.get("is_error", False),
            "model": record.get("model"), "live": record.get("live", False),
            "latency_s": record.get("latency_s"), "tokens": record.get("tokens")}


class OllamaProcessWorker:
    live = True

    def __init__(self, worker_id: str, run_dir: Path, model_name: str, registry: ProcessRegistry,
                 run_id: str, goal: str = TASK_GOAL, inject_fault: str = "none",
                 fault_cooldown: float = 10.0):
        self.worker_id = worker_id
        self.run_dir = Path(run_dir)
        self.sandbox = self.run_dir / "sandbox"
        self.snapshot_dir = self.run_dir / "snapshot"
        self.events_path = self.run_dir / "events.jsonl"
        self.control_path = self.run_dir / "control.jsonl"
        self.revoked_path = self.run_dir / "REVOKED"
        self.model_name = model_name
        self.model_label = f"ollama:{model_name}"
        self.registry = registry
        self.run_id = run_id
        self.goal = goal
        self.inject_fault = inject_fault
        self.fault_cooldown = fault_cooldown
        self.process: OwnedProcess | None = None
        self.offset = 0          # bytes of events.jsonl already observed
        self.seq = 0             # last control command number sent
        self.records: list[dict] = []
        self.result: dict | None = None
        self.last_event_at = time.time()

    # --- lifecycle -------------------------------------------------------------------
    def start(self, resume: bool = False) -> None:
        if not self.sandbox.exists():
            create_task(self.sandbox, self.snapshot_dir)
        token = new_token()
        argv = [sys.executable, str(ROOT / "worker_process.py"), "--run-dir", str(self.run_dir),
                "--run-id", self.run_id, "--model", self.model_name, "--goal", self.goal,
                "--inject-fault", self.inject_fault, "--fault-cooldown", str(self.fault_cooldown),
                "--owner-token", token]
        log = open(self.run_dir / "worker.log", "a")
        self.process = self.registry.spawn(self.worker_id, argv, token, cwd=str(ROOT),
                                           stdout=log, stderr=log, env=os.environ.copy())
        self.last_event_at = time.time()

    def is_running(self) -> bool:
        return self.process is not None and is_alive(self.process)

    # --- observation -----------------------------------------------------------------
    def observe(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        with open(self.events_path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        events = []
        for line in chunk[:end].splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "step":
                if any(r["step"] == event["step"] for r in self.records):
                    continue  # re-emitted after a crash
                self.records.append(event)
            elif event.get("type") in ("finished", "exhausted", "cancelled", "revoked", "exited"):
                if self.process is not None and event.get("pid") != self.process.pid:
                    continue  # from a worker process that ended before a restart
                self.result = event
            events.append(event)
        if events:
            self.last_event_at = time.time()
        return events

    def get_history(self) -> list[dict]:
        return [_to_step(self.run_id, r, r.get("test_passed")) for r in self.records]

    def get_result(self) -> dict | None:
        return self.result

    def set_evidence(self, step_number: int, test_passed: bool) -> None:
        for r in self.records:
            if r["step"] == step_number:
                r["test_passed"] = test_passed

    # --- control ---------------------------------------------------------------------
    def hold(self) -> None:
        """Pause the worker at its next loop while the supervisor deliberates."""
        self.send({"type": "hold"})

    def send(self, command: dict) -> int:
        self.seq += 1
        with open(self.control_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**command, "seq": self.seq, "ts": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return self.seq

    def rollback(self) -> str:
        restored = restore_snapshot(self.sandbox, self.snapshot_dir)
        self.send({"type": "notice", "text": "the supervisor rolled back the project files to "
                   f"their original state ({', '.join(restored)}). Re-read them before editing."})
        return f"restored {', '.join(restored)}"

    def revoke(self, reason: str) -> None:
        self.revoked_path.write_text(reason)

    def request_cancel(self, reason: str) -> None:
        self.send({"type": "cancel", "reason": reason})

    # --- evidence --------------------------------------------------------------------
    def verify(self) -> dict:
        from sandbox_tools import run_pytest
        passed, summary = run_pytest(self.sandbox)
        return {"test_passed": passed, "summary": summary, "method": "supervisor-run pytest"}

    # --- persistence -----------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"offset": self.offset, "seq": self.seq, "records": self.records[-200:],
                "result": self.result,
                "pid": self.process.pid if self.process else None,
                "token": self.process.token if self.process else None}

    def restore(self, data: dict) -> None:
        self.offset = data.get("offset", 0)
        self.seq = data.get("seq", 0)
        self.records = data.get("records", [])
        self.result = data.get("result")
        if data.get("pid") and data.get("token"):
            self.process = self.registry.adopt(self.worker_id, data["pid"], data["token"])


class ScriptedVictimWorker:
    """SIMULATED worker: one of the seven deterministic victims from victims.py."""
    live = False
    process = None

    def __init__(self, victim):
        self.victim = victim
        self.worker_id = victim.run_id.split("-", 1)[1]  # e.g. "6-liar", session-free
        self.run_id = victim.run_id
        self.model_label = "scripted (no LLM)"
        self.goal = f"scripted scenario: {victim.name}"
        self.history: list[dict] = []
        self.cancelled = False
        self.result: dict | None = None
        self.last_event_at = time.time()

    def start(self, resume: bool = False) -> None:
        pass

    def observe(self) -> list[dict]:
        if not self.is_running():
            return []
        step = self.victim.next_step(self.victim.state)
        record = {**step, "type": "step", "thought": "", "is_error": False, "live": False}
        self.history.append(step)
        events = [record]
        if self.victim.state["done"]:
            self.result = {"type": "finished", "step": step["step"]}
            events.append(self.result)
        self.last_event_at = time.time()
        return events

    def get_history(self) -> list[dict]:
        return list(self.history)

    def get_result(self) -> dict | None:
        return self.result

    def hold(self) -> None:
        pass  # synchronous: the victim never runs while the supervisor thinks

    def send(self, command: dict) -> int | None:
        mapping = {"backoff": "retry_with_backoff", "hint": "change_strategy",
                   "rollback": "rollback", "cancel": "stop"}
        action = mapping.get(command.get("type"))
        if action:
            self.victim.apply(action)
        return None  # applied immediately, nothing to acknowledge

    def rollback(self) -> str:
        self.victim.apply("rollback")
        return "scripted rollback"

    def revoke(self, reason: str) -> None:
        self.cancelled = True

    def request_cancel(self, reason: str) -> None:
        self.victim.apply("stop")
        self.cancelled = True
        self.result = {"type": "cancelled", "reason": reason}

    def is_running(self) -> bool:
        return not self.cancelled and not self.victim.state["done"]

    def verify(self) -> dict:
        passed = bool(self.history and self.history[-1]["evidence"]["test_passed"])
        return {"test_passed": passed, "summary": "scripted evidence from the victim",
                "method": "simulated"}

    def set_evidence(self, step_number: int, test_passed: bool) -> None:
        pass  # scripted victims carry their own ground truth

    def snapshot(self) -> dict:
        return {"victim_state": self.victim.state, "history": self.history,
                "cancelled": self.cancelled, "result": self.result}

    def restore(self, data: dict) -> None:
        self.victim.state = data["victim_state"]
        self.history = data["history"]
        self.cancelled = data["cancelled"]
        self.result = data["result"]
