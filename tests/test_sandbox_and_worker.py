"""Sandbox tool rules, the injected fault, and the real worker subprocess (offline)."""

import json
import time
import uuid

import pytest

from kill_controls import ProcessRegistry, is_alive, stop_agent
from sandbox_tools import (RateLimitFault, ToolGateway, ToolRevoked, create_task,
                           restore_snapshot, run_pytest)
from workers import OllamaProcessWorker


@pytest.fixture
def gateway(tmp_path):
    create_task(tmp_path / "sandbox", tmp_path / "snapshot")
    return ToolGateway(tmp_path / "sandbox", tmp_path / "REVOKED")


FIXED = '''def restock_needed(stock, threshold):
    return sorted(name for name, qty in stock.items() if qty < threshold)


def total_value(stock, prices):
    return sum(stock[name] * prices[name] for name in stock)
'''


def test_task_starts_failing_and_fix_passes(gateway):
    assert run_pytest(gateway.sandbox)[0] is False
    assert gateway.call("edit_file", {"path": "inventory.py", "content": FIXED}) == \
        (f"wrote inventory.py ({len(FIXED.encode())} bytes)", False)
    passed, summary = run_pytest(gateway.sandbox)
    assert passed and "3 passed" in summary


@pytest.mark.parametrize("args, message", [
    ({"path": "../outside.py", "content": "x = 1"}, "outside the sandbox"),
    ({"path": "/etc/passwd", "content": "x = 1"}, "outside the sandbox"),
    ({"path": "test_inventory.py", "content": "def test(): pass"}, "read-only"),
    ({"path": "inventory.py", "content": "import os\nos.system('echo hi')"}, "import of 'os'"),
    ({"path": "inventory.py", "content": "x = eval('1')"}, "eval()"),
    ({"path": "inventory.py", "content": "x = ().__class__"}, "dunder"),
    ({"path": "inventory.py", "content": "def broken(:"}, "does not parse"),
])
def test_edit_file_rules(gateway, args, message):
    result, is_error = gateway.call("edit_file", args)
    assert is_error and message in result


def test_unknown_tool_is_an_error(gateway):
    assert gateway.call("shell", {"cmd": "ls"})[1] is True


def test_revoked_worker_cannot_call_tools(gateway):
    gateway.revoked_marker.write_text("budget exhausted")
    with pytest.raises(ToolRevoked):
        gateway.call("read_file", {"path": "inventory.py"})


def test_injected_fault_needs_a_pause_then_clears():
    fault = RateLimitFault(cooldown_s=10)
    assert "INJECTED FAULT #1" in fault.check(now=100)
    assert "INJECTED FAULT #2" in fault.check(now=103)   # retrying too fast
    assert fault.check(now=114) is None                  # after a >=10s backoff
    assert fault.check(now=114.1) is None                # cleared for good


def test_rollback_restores_original_files(gateway, tmp_path):
    gateway.call("edit_file", {"path": "inventory.py", "content": FIXED})
    restore_snapshot(gateway.sandbox, tmp_path / "snapshot")
    assert "qty > threshold" in (gateway.sandbox / "inventory.py").read_text()


def test_real_worker_subprocess_is_observed_and_stopped(tmp_path, monkeypatch):
    """Worker process with an unreachable model: steps are recorded, then STOP_AGENT."""
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")   # nothing listens here
    monkeypatch.setenv("FLIGHT_RECORDER_DB", str(tmp_path / "fr.sqlite3"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    worker = OllamaProcessWorker("coder", run_dir, "qwen3.5:9b", ProcessRegistry(),
                                 run_id=f"t-{uuid.uuid4().hex[:6]}")
    worker.start()
    deadline = time.time() + 20
    while time.time() < deadline and len(worker.get_history()) < 2:
        worker.observe()
        time.sleep(0.2)
    history = worker.get_history()
    assert len(history) >= 2 and history[0]["action"] == "invalid_reply"
    assert history[0]["live"] is False  # an unreachable model is never reported as live
    report = stop_agent(worker, "test stop", grace_s=5, term_s=3)
    assert report["verified"] and not is_alive(worker.process)
    assert (run_dir / "REVOKED").read_text() == "test stop"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(e["type"] in ("cancelled", "revoked") for e in events) or \
        report["method"] in ("sigterm", "sigkill")
