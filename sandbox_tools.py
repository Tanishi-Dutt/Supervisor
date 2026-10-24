"""Disposable coding sandbox and the only tools an LLM worker may use.

Safety boundaries (enforced here, not by the model):
* Every path must resolve to a regular file directly inside the sandbox directory.
* edit_file may only rewrite non-test *.py files; test files are read-only.
* Edited code is parsed first; only imports from a small stdlib allowlist and no
  calls to exec/eval/open/__import__/compile are accepted.
* run_tests runs pytest inside the sandbox with a timeout and a minimal environment.
* After the supervisor revokes a worker, every tool call raises ToolRevoked.

This is a guardrail, not an OS-level sandbox: code that passes the checks runs as
the current user. See README_SUPERVISOR_AGENT.md "Known limitations".
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK_GOAL = ("Make every test in test_inventory.py pass by fixing the bugs in inventory.py. "
             "Do not edit the tests. Verify with run_tests before you finish.")

BUGGY_INVENTORY = '''"""Tiny inventory helpers for a stationery shop."""


def restock_needed(stock, threshold):
    """Names of items whose quantity is below `threshold`, sorted alphabetically."""
    return [name for name, qty in stock.items() if qty > threshold]


def total_value(stock, prices):
    """Total value of the stock: quantity times unit price, summed over all items."""
    return sum(stock[name] + prices[name] for name in stock)
'''

INVENTORY_TESTS = '''from inventory import restock_needed, total_value


def test_restock_needed_lists_low_items_sorted():
    assert restock_needed({"pens": 3, "paper": 50, "ink": 1}, 5) == ["ink", "pens"]


def test_restock_needed_empty_when_all_stocked():
    assert restock_needed({"pens": 10}, 5) == []


def test_total_value():
    assert total_value({"pens": 3, "ink": 2}, {"pens": 1.5, "ink": 10.0}) == 24.5
'''

TASK_FILES = {"inventory.py": BUGGY_INVENTORY, "test_inventory.py": INVENTORY_TESTS}
ALLOWED_IMPORTS = {"math", "decimal", "fractions", "typing", "dataclasses", "functools",
                   "itertools", "collections", "operator", "statistics", "inventory"}
FORBIDDEN_CALLS = {"exec", "eval", "open", "compile", "__import__", "globals", "locals",
                   "getattr", "setattr", "delattr", "input", "breakpoint"}
MAX_FILE_BYTES = 20_000
TEST_TIMEOUT_S = 30


class ToolRevoked(PermissionError):
    """The supervisor stopped this worker; no further tool calls are allowed."""


class ToolError(ValueError):
    """A tool call was rejected by the sandbox rules."""


def create_task(sandbox: Path, snapshot: Path) -> None:
    """Write the task files and a pristine snapshot used for rollback."""
    sandbox.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, text in TASK_FILES.items():
        (sandbox / name).write_text(text)
        (snapshot / name).write_text(text)


def restore_snapshot(sandbox: Path, snapshot: Path) -> list[str]:
    restored = []
    for src in sorted(snapshot.iterdir()):
        if src.is_file():
            shutil.copyfile(src, sandbox / src.name)
            restored.append(src.name)
    return restored


def run_pytest(sandbox: Path) -> tuple[bool, str]:
    """Run the sandbox's tests. Used by the worker's tool and the supervisor's verifier."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": str(sandbox)}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header",
             "-o", "addopts=", "--rootdir", str(sandbox)],
            cwd=sandbox, capture_output=True, text=True, timeout=TEST_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return False, f"tests timed out after {TEST_TIMEOUT_S}s"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    summary = re.sub(r" in [\d.]+s", "", lines[-1]) if lines else "pytest produced no output"
    failures = [line.strip() for line in lines if line.startswith(("E ", "FAILED"))][:6]
    detail = "\n".join(failures)
    return proc.returncode == 0, summary + (f"\n{detail}" if detail else "")


class RateLimitFault:
    """INJECTED, clearly labelled failure: run_tests is rate limited (HTTP 429).

    The first call fails, and every call within `cooldown_s` of the previous attempt
    fails. The worker has no way to wait on its own; only a supervisor-applied
    backoff (a pause of at least `cooldown_s`) clears it. After one successful call
    the fault switches itself off.
    """

    def __init__(self, cooldown_s: float = 10.0, state: dict | None = None):
        self.cooldown_s = cooldown_s
        self.state = state if state is not None else {"active": True, "last_attempt": None,
                                                      "rejections": 0}

    def check(self, now: float | None = None) -> str | None:
        if not self.state["active"]:
            return None
        now = time.time() if now is None else now
        last = self.state["last_attempt"]
        self.state["last_attempt"] = now
        if last is None or now - last < self.cooldown_s:
            self.state["rejections"] += 1
            return ("ERROR 429 Too Many Requests: test runner rate limited "
                    f"[INJECTED FAULT #{self.state['rejections']}]")
        self.state["active"] = False
        return None


def _check_code(content: str) -> None:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise ToolError(f"refused: code does not parse ({exc.msg} on line {exc.lineno})")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            names = []
        bad = [n for n in names if n not in ALLOWED_IMPORTS]
        if bad:
            raise ToolError(f"refused: import of {bad[0]!r} is not allowed in the sandbox")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in FORBIDDEN_CALLS:
            raise ToolError(f"refused: call to {node.func.id}() is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") \
                and node.attr not in ("__init__", "__name__"):
            raise ToolError(f"refused: dunder attribute {node.attr!r} is not allowed")


class ToolGateway:
    """Executes allowlisted tools for one worker inside one sandbox."""

    TOOLS = ("list_files", "read_file", "edit_file", "run_tests", "finish")

    def __init__(self, sandbox: Path, revoked_marker: Path, fault: RateLimitFault | None = None):
        self.sandbox = sandbox.resolve()
        self.revoked_marker = revoked_marker
        self.fault = fault

    def _guard(self) -> None:
        if self.revoked_marker.exists():
            reason = self.revoked_marker.read_text().strip() or "stopped by supervisor"
            raise ToolRevoked(f"tool call rejected: worker was stopped ({reason})")

    def _path(self, raw: object) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("refused: 'path' must be a file name")
        path = (self.sandbox / raw).resolve()
        if path.parent != self.sandbox or path.is_symlink():
            raise ToolError(f"refused: {raw!r} is outside the sandbox")
        return path

    def call(self, tool: str, args: dict) -> tuple[str, bool]:
        """Return (result_text, is_error). Raises ToolRevoked after a stop."""
        self._guard()
        if not isinstance(args, dict):
            args = {}
        try:
            if tool == "list_files":
                return ", ".join(sorted(p.name for p in self.sandbox.iterdir()
                                        if p.is_file() and not p.name.startswith("."))), False
            if tool == "read_file":
                path = self._path(args.get("path"))
                if not path.is_file():
                    return f"no such file: {args.get('path')}", True
                return path.read_text()[:4000], False
            if tool == "edit_file":
                path = self._path(args.get("path"))
                content = args.get("content")
                if not path.name.endswith(".py") or path.name.startswith("test_"):
                    raise ToolError(f"refused: {path.name} is read-only (tests cannot be edited)")
                if not isinstance(content, str) or not content.strip():
                    raise ToolError("refused: 'content' must be the full new file text")
                if len(content.encode()) > MAX_FILE_BYTES:
                    raise ToolError("refused: file too large")
                _check_code(content)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(content)
                os.replace(tmp, path)
                return f"wrote {path.name} ({len(content.encode())} bytes)", False
            if tool == "run_tests":
                if self.fault is not None:
                    injected = self.fault.check()
                    if injected:
                        return injected, True
                passed, summary = run_pytest(self.sandbox)
                return summary, False  # failing tests are information, not a tool error
            if tool == "finish":
                return "worker reports the task is complete", False
            return f"unknown tool {tool!r}; allowed: {', '.join(self.TOOLS)}", True
        except ToolError as exc:
            return str(exc), True


def content_digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]
