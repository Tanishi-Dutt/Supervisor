"""STOP_AGENT and STOP_LOCAL_MODEL within their safety limits."""

import os
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from kill_controls import (ModelLeases, OwnedProcess, ProcessRegistry, is_alive, new_token,
                           stop_agent, stop_local_model, terminate_owned, verify_ownership)

STUBBORN = textwrap.dedent("""
    import signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)   # ignores polite termination
    while True:
        time.sleep(0.1)
""")
POLITE = "import time\nwhile True: time.sleep(0.1)\n"


def spawn(code, registry=None):
    registry = registry or ProcessRegistry()
    token = new_token()
    proc = registry.spawn("w", [sys.executable, "-c", code, token], token)
    time.sleep(0.3)
    return proc


class FakeWorker:
    def __init__(self, proc):
        self.worker_id, self.process, self.live = "w", proc, False
        self.revoked = self.cancelled = None

    def revoke(self, reason):
        self.revoked = reason

    def request_cancel(self, reason):
        self.cancelled = reason

    def is_running(self):
        return is_alive(self.process)


def test_stop_agent_escalates_to_sigkill_and_verifies():
    proc = spawn(STUBBORN)
    report = stop_agent(FakeWorker(proc), "test", grace_s=0.5, term_s=0.5)
    assert report["method"] == "sigkill" and report["verified"] is True
    assert not is_alive(proc)


def test_stop_agent_uses_sigterm_when_enough():
    proc = spawn(POLITE)
    worker = FakeWorker(proc)
    report = stop_agent(worker, "budget exhausted", grace_s=0.3, term_s=3)
    assert report["method"] == "sigterm" and report["verified"]
    assert worker.revoked == worker.cancelled == "budget exhausted"


def test_refuses_to_signal_own_or_parent_process():
    for pid in (os.getpid(), os.getppid(), 1):
        ok, why = verify_ownership(OwnedProcess("x", pid, new_token()))
        assert not ok and "protected" in why


def test_refuses_pid_without_owner_token():
    proc = spawn(POLITE)
    impostor = OwnedProcess("x", proc.pid, new_token())  # right pid, wrong token
    report = terminate_owned(impostor, grace_s=0.2, term_s=0.2)
    assert report["method"] == "refused" and is_alive(proc)
    terminate_owned(proc, grace_s=0.1, term_s=2)


def test_spawn_requires_token_in_argv():
    with pytest.raises(ValueError):
        ProcessRegistry().spawn("w", [sys.executable, "-c", POLITE], new_token())


class FakeModel:
    def __init__(self, local=True, loaded=True):
        self.name, self.provider, self.local = "qwen3.5:9b", "ollama", local
        self._loaded, self.unloads = loaded, 0

    def loaded(self):
        return self._loaded

    def unload(self):
        self.unloads += 1
        self._loaded = False


def test_stop_local_model_refuses_when_shared():
    leases, model = ModelLeases(), FakeModel()
    leases.acquire(model.name, "supervisor:reasoning")
    result = stop_local_model(model, leases, "supervisor:kill-switch")
    assert not result["stopped"] and "shared" in result["reason"] and model.unloads == 0


def test_stop_local_model_unloads_when_exclusive():
    model = FakeModel()
    result = stop_local_model(model, ModelLeases(), "supervisor:kill-switch")
    assert result["stopped"] and model.unloads == 1 and result["model_files"] == "untouched"


def test_stop_local_model_never_claims_to_stop_hosted_models():
    result = stop_local_model(FakeModel(local=False), ModelLeases(), "k")
    assert not result["stopped"] and "hosted" in result["reason"]


def test_stop_local_model_noop_when_not_loaded():
    model = FakeModel(loaded=False)
    result = stop_local_model(model, ModelLeases(), "k")
    assert result["action"] == "none" and model.unloads == 0
