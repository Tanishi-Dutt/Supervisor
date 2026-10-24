"""Agent #8: a real coding agent (Dev A). Off by default: run_demo.py --real.

Unlike the scripted victims, its tools touch real files in a sandbox and its evidence
is a real pytest run, recomputed by the harness every step (never taken from the
agent's word). Its brain is Qwen via Ollama when the model is available; otherwise a
scripted brain drives the same real tools so the demo still works offline.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from interfaces import OLLAMA_URL
from victims import Victim

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
SANDBOX_ROOT = Path(tempfile.gettempdir()) / "flight_recorder_sandbox"

BUGGY_CART = '''def total(prices, discount_pct=0):
    """Sum prices and apply a percentage discount, rounded to cents."""
    subtotal = sum(prices)
    return round(subtotal - discount_pct, 2)
'''
FIXED_CART = '''def total(prices, discount_pct=0):
    """Sum prices and apply a percentage discount, rounded to cents."""
    subtotal = sum(prices)
    return round(subtotal * (1 - discount_pct / 100), 2)
'''
TESTS = '''from cart import total


def test_no_discount():
    assert total([10, 20]) == 30


def test_ten_percent():
    assert total([10, 20], 10) == 27.0


def test_rounding():
    assert total([19.99, 5.01], 15) == 21.25
'''
ORIGINAL = {"cart.py": BUGGY_CART, "test_cart.py": TESTS}

# Scripted brain: re-runs the tests hoping they pass (a classic stuck agent) until the
# supervisor tells it to change strategy, then actually investigates and fixes the bug.
SCRIPTED_FIX = [("read_file", {"path": "test_cart.py"}), ("read_file", {"path": "cart.py"}),
                ("write_file", {"path": "cart.py", "content": FIXED_CART}), ("run_tests", {})]

PROMPT = """You are a coding agent. Goal: make all tests pass in a small Python project.
Files: cart.py (code), test_cart.py (tests, do not edit).
Tools (respond with exactly one, as JSON):
  {{"action": "read_file", "path": "<file>"}}
  {{"action": "write_file", "path": "cart.py", "content": "<entire new file>"}}
  {{"action": "run_tests"}}
  {{"action": "done", "note": "<why you are done>"}}
{hint}
Your recent steps and what happened (oldest first):
{obs}
Respond with JSON only."""


def installed_model() -> str | None:
    """The exact tag to call: MODEL if pulled, else any tag of the same model (e.g. :latest)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None
    if MODEL in names:
        return MODEL
    return next((n for n in names if n.split(":")[0] == MODEL.split(":")[0]), None)


class RealCoder(Victim):
    name = "Real coder"
    LOOPS_FOREVER = False

    def __init__(self, session: str, index: int, brain: str | None = None):
        super().__init__(session, index)
        if brain is None:
            model = installed_model()
            brain = f"llm:{model}" if model else "scripted"
        # Scripted brain is deterministic; a real LLM may recover, finish alone, or be stopped.
        self.EXPECTED_OUTCOME = "recovered" if brain == "scripted" else "any"
        self.state.update({"brain": brain, "hint": "", "obs": [], "plan_i": 0})
        self.sandbox = SANDBOX_ROOT / self.run_id
        if not self.sandbox.exists():  # survives kill -9: a resumed run keeps its edits
            self._reset_files()

    def _reset_files(self):
        self.sandbox.mkdir(parents=True, exist_ok=True)
        for name, text in ORIGINAL.items():
            (self.sandbox / name).write_text(text)

    # --- real tools --------------------------------------------------------
    def _pytest(self) -> tuple[bool, str]:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              cwd=self.sandbox, capture_output=True, text=True, timeout=30)
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        summary = re.sub(r" in [\d.]+s$", "", lines[-1]) if lines else "pytest produced no output"
        return proc.returncode == 0, summary  # timing stripped so identical runs look identical

    def _tool(self, action: str, args: dict) -> str:
        if action == "read_file":
            path = self.sandbox / Path(args.get("path", "")).name
            return path.read_text()[:600] if path.exists() else f"no such file: {args.get('path')}"
        if action == "write_file":
            name = Path(args.get("path", "")).name
            if name != "cart.py":
                return f"refused: only cart.py may be edited (got {name!r})"
            (self.sandbox / name).write_text(args.get("content", ""))
            return f"wrote {name} ({len(args.get('content', ''))} bytes)"
        if action == "run_tests":
            return self._pytest()[1]
        return f"unknown tool {action!r}"

    # --- brains ------------------------------------------------------------
    def _scripted(self, s) -> dict:
        if s["strategy"] == 0:
            return {"action": "run_tests"}
        if s["plan_i"] >= len(SCRIPTED_FIX):
            return {"action": "done", "note": "tests pass"}
        action, args = SCRIPTED_FIX[s["plan_i"]]
        s["plan_i"] += 1
        return {"action": action, **args}

    def _qwen(self, s) -> dict:
        prompt = PROMPT.format(hint=f"SUPERVISOR NOTE: {s['hint']}" if s["hint"] else "",
                               obs="\n".join(s["obs"][-6:]) or "(none yet)")
        payload = {"model": s["brain"].removeprefix("llm:"), "prompt": prompt, "format": "json", "stream": False,
                   "think": False, "keep_alive": "30m", "options": {"temperature": 0.2}}
        req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                decision = json.loads(json.load(r).get("response", "{}"))
            return decision if isinstance(decision, dict) else {"action": "invalid"}
        except Exception as e:
            return {"action": "invalid", "note": type(e).__name__}

    # --- Victim interface --------------------------------------------------
    def _act(self, s):
        d = self._scripted(s) if s["brain"] == "scripted" else self._qwen(s)
        action = d.get("action", "invalid")
        args = {k: v for k, v in d.items() if k in ("path",)}
        if action == "write_file":  # hash the content so identical rewrites look like a loop
            args["sha"] = hashlib.sha1(d.get("content", "").encode()).hexdigest()[:8]
        claimed = None
        if action == "done":
            result, claimed = "agent says it is done", d.get("note") or "done"
        elif action == "invalid":
            result = f"brain returned no usable action ({d.get('note', 'bad JSON')})"
        else:
            result = self._tool(action, d)
        passed, summary = self._pytest()  # ground truth, regardless of what the agent claims
        s["obs"] = (s["obs"] + [f"{action} {args.get('path', '')}: {result[:160]}"])[-6:]
        step = self._mk(action, args, result[:200], claimed=claimed, passed=passed)
        if passed and action in ("run_tests", "done"):
            self.state["done"] = True
        return step

    def apply(self, action: str) -> None:
        s = self.state
        if action == "change_strategy":
            s["hint"] = ("your last steps repeated without progress. Read test_cart.py and "
                         "cart.py, find the bug, rewrite cart.py, then run the tests.")
        elif action == "rollback":
            self._reset_files()
            s["hint"] = "the supervisor rolled back your edits to the original files."
        elif action == "retry_with_backoff":
            time.sleep(min(2 ** s["retries"], 4))
        super().apply(action)
