"""A real LLM coding worker, run as its own subprocess by SupervisorAgent.

The model picks every tool call itself from the conversation so far; nothing here
replays a fixed sequence. The process talks to the supervisor through files in its
run directory:

  events.jsonl   worker -> supervisor  one JSON event per line (append-only)
  control.jsonl  supervisor -> worker  numbered commands: backoff, hint, notice, cancel
  REVOKED        written by the supervisor's stop_agent; the tool gateway refuses
                 every call once it exists

Progress (conversation, step count, fault state, processed commands) is checkpointed
in SQLite through store.py after every step, so a restarted worker resumes mid-task.
The worker exits by itself if its parent (the supervisor) dies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import store
from llm_provider import OllamaChatModel
from sandbox_tools import (TASK_GOAL, RateLimitFault, ToolGateway, ToolRevoked,
                           content_digest)

SYSTEM = """You are an autonomous coding agent working in a small Python project.
You act by calling exactly one tool per reply. Reply with a JSON object only:
{"thought": "<one short sentence>", "tool": "<tool name>", "args": {...}}

Tools:
- list_files: {}                                    list project files
- read_file: {"path": "<file>"}                     read a file
- edit_file: {"path": "<file>", "content": "<the ENTIRE new file>"}   rewrite a non-test file
- run_tests: {}                                     run the test suite
- finish: {"summary": "<what you did>"}             only after run_tests shows all tests passing

Messages starting with SUPERVISOR come from the system supervising you; follow them."""

MAX_STEPS = 40
HOLD_TIMEOUT_S = 90
KEEP_MESSAGES = 14


def _append(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn final line from a crash; ignore it
    return out


class Worker:
    def __init__(self, args: argparse.Namespace):
        self.run_dir = Path(args.run_dir)
        self.sandbox = self.run_dir / "sandbox"
        self.events = self.run_dir / "events.jsonl"
        self.control = self.run_dir / "control.jsonl"
        self.revoked = self.run_dir / "REVOKED"
        self.run_id = args.run_id
        self.model = OllamaChatModel(args.model)
        self.parent = os.getppid()
        checkpoint = store.load_checkpoint(self.run_id)
        if checkpoint:
            self.state = checkpoint["state"]
        else:
            self.state = {"step": 0, "cmd_seq": 0, "messages": [], "records": [],
                          "fault": None, "finished": False}
            self.state["messages"].append({"role": "user", "content": f"GOAL: {args.goal}"})
            if args.inject_fault == "rate_limit":
                self.state["fault"] = RateLimitFault(args.fault_cooldown).state
        fault = (RateLimitFault(args.fault_cooldown, self.state["fault"])
                 if self.state["fault"] is not None else None)
        self.gateway = ToolGateway(self.sandbox, self.revoked, fault)
        self._reemit_missing()

    # --- helpers ---------------------------------------------------------------------
    def emit(self, kind: str, **fields) -> None:
        _append(self.events, {"type": kind, "ts": time.time(), "pid": os.getpid(), **fields})

    def _reemit_missing(self) -> None:
        """A crash between checkpoint and event write must not lose a step."""
        seen = {e.get("step") for e in _read_jsonl(self.events) if e.get("type") == "step"}
        for record in self.state["records"]:
            if record["step"] not in seen:
                self.emit("step", **record)
        self.emit("worker_started", resumed=bool(self.state["step"]), model=self.model.name,
                  at_step=self.state["step"])

    def save(self) -> None:
        # One checkpoint number per state change (steps*1000 + processed commands), so
        # the store's no-rewind and no-conflicting-rewrite rules hold.
        store.save(self.run_id, self.state["step"] * 1000 + self.state["cmd_seq"], self.state)

    def has_new_commands(self) -> bool:
        return self.revoked.exists() or any(
            c.get("seq", 0) > self.state["cmd_seq"] for c in _read_jsonl(self.control))

    def parent_alive(self) -> bool:
        return os.getppid() == self.parent

    def say(self, text: str) -> None:
        self.state["messages"].append({"role": "user", "content": f"SUPERVISOR: {text}"})

    # --- control channel -------------------------------------------------------------
    def process_commands(self) -> str | None:
        for cmd in _read_jsonl(self.control):
            if cmd.get("seq", 0) <= self.state["cmd_seq"]:
                continue  # already handled (commands may be re-sent after a restart)
            kind = cmd.get("type")
            self.state["cmd_seq"] = cmd["seq"]
            if kind == "cancel":
                self.save()
                self.emit("cancelled", reason=cmd.get("reason", ""), seq=cmd["seq"])
                return "cancel"
            if kind == "hold":
                # The supervisor is deliberating: wait for its next command.
                self.emit("ack", seq=cmd["seq"], command=kind)
                self.save()
                deadline = time.time() + HOLD_TIMEOUT_S
                while time.time() < deadline:
                    if self.revoked.exists() or not self.parent_alive():
                        return "cancel"
                    if any(c.get("seq", 0) > cmd["seq"] for c in _read_jsonl(self.control)):
                        break
                    time.sleep(0.2)
                else:
                    self.emit("hold_timeout", seq=cmd["seq"])
                return self.process_commands()  # handle the command that ended the hold
            if kind == "backoff":
                seconds = float(cmd.get("seconds", 5))
                self.emit("ack", seq=cmd["seq"], command=kind, seconds=seconds)
                deadline = time.time() + seconds
                while time.time() < deadline:
                    if self.revoked.exists() or not self.parent_alive():
                        return "cancel"
                    time.sleep(0.2)
                self.say(f"you were paused for {seconds:.0f}s to back off. {cmd.get('note', '')}"
                         .strip())
            elif kind in ("hint", "notice"):
                self.emit("ack", seq=cmd["seq"], command=kind)
                self.say(str(cmd.get("text", ""))[:500])
            self.save()
        return None

    # --- main loop -------------------------------------------------------------------
    def run(self) -> int:
        while True:
            if not self.parent_alive():
                self.emit("exited", reason="supervisor process is gone")
                return 0
            if self.process_commands() == "cancel":
                return 0
            if self.state["finished"]:
                self.emit("finished", step=self.state["step"])
                return 0
            if self.state["step"] >= MAX_STEPS:
                self.emit("exhausted", step=self.state["step"])
                return 0
            try:
                self.one_step()
            except ToolRevoked as exc:
                self.emit("revoked", reason=str(exc))
                return 0

    def one_step(self) -> None:
        history = self.state["messages"]
        reply = self.model.complete_json(SYSTEM, history[:1] + history[1:][-KEEP_MESSAGES:])
        step = self.state["step"] + 1
        record = {"step": step, "model": reply.model, "provider": reply.provider,
                  "live": reply.live, "latency_s": round(reply.latency_s, 2),
                  "tokens": reply.tokens, "claimed_status": None}
        if not reply.ok:
            record.update(action="invalid_reply", args={}, thought="",
                          result=f"model reply unusable: {reply.error}", is_error=True)
            history.append({"role": "user", "content": "Your last reply was not a valid JSON "
                            "tool call. Reply with one JSON object as instructed."})
            time.sleep(1.0)  # don't spin if the model server is down
        elif self.has_new_commands():
            # The supervisor spoke while we were thinking: its command (hold, backoff,
            # cancel...) takes effect before this tool call runs. Drop the call.
            self.emit("preempted", tool=str(reply.data.get("tool", "")), at_step=step)
            return
        else:
            data = reply.data
            tool = str(data.get("tool", ""))
            args = data.get("args") if isinstance(data.get("args"), dict) else {}
            self.gateway._guard()  # raises ToolRevoked before anything runs
            result, is_error = self.gateway.call(tool, args)
            shown_args = {k: v for k, v in args.items() if k != "content"}
            if tool == "edit_file" and isinstance(args.get("content"), str):
                shown_args["sha"] = content_digest(args["content"])
                shown_args["bytes"] = len(args["content"].encode())
            record.update(action=tool, args=shown_args, thought=str(data.get("thought", ""))[:300],
                          result=result[:600], is_error=is_error)
            if tool == "finish" and not is_error:
                record["claimed_status"] = str(args.get("summary") or "task complete")[:200]
                self.state["finished"] = True
            history.append({"role": "assistant", "content": reply.text[:6000]})
            history.append({"role": "user", "content": f"TOOL RESULT ({tool}):\n{result[:3000]}"})
        self.state["step"] = step
        self.state["records"] = (self.state["records"] + [record])[-200:]
        self.state["messages"] = history[:1] + history[1:][-KEEP_MESSAGES:]
        self.save()                 # checkpoint first ...
        self.emit("step", **record)  # ... then publish; _reemit_missing covers the gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--goal", default=TASK_GOAL)
    ap.add_argument("--inject-fault", choices=["none", "rate_limit"], default="none")
    ap.add_argument("--fault-cooldown", type=float, default=10.0)
    ap.add_argument("--owner-token", required=True)  # lets the supervisor verify ownership
    args = ap.parse_args()
    return Worker(args).run()


if __name__ == "__main__":
    sys.exit(main())
