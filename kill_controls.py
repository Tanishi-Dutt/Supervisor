"""STOP_AGENT and STOP_LOCAL_MODEL: the supervisor's two kill controls.

STOP_AGENT acts only on processes this supervisor launched itself. Each worker is
spawned with a unique --owner-token; before any signal is sent we check that the
PID still belongs to a process whose command line carries that token, and that it
is not this process, its parent, or PID 1. There is no name matching and no pkill.

STOP_LOCAL_MODEL unloads a model from a local Ollama server through the documented
keep_alive=0 request. It refuses while any other holder (another worker, or the
supervisor's own reasoning) still leases the model, never deletes model files, and
never touches the Ollama server process itself. Hosted providers are not unloaded:
for them we can only cancel our own worker, which STOP_AGENT already does.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- processes
@dataclass
class OwnedProcess:
    label: str
    pid: int
    token: str
    popen: subprocess.Popen | None = None  # None when re-adopted after a restart


def new_token() -> str:
    return f"fr-{uuid.uuid4().hex}"


def _cmdline(pid: int) -> str | None:
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5)
    except Exception:
        return None
    return out.stdout.strip() or None


def is_alive(proc: OwnedProcess) -> bool:
    if proc.popen is not None:
        return proc.popen.poll() is None
    try:
        os.kill(proc.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmd = _cmdline(proc.pid)
    return bool(cmd) and proc.token in cmd  # a recycled PID is not "our" process


def verify_ownership(proc: OwnedProcess) -> tuple[bool, str]:
    if proc.pid in (os.getpid(), os.getppid(), 0, 1):
        return False, f"refusing to signal protected pid {proc.pid}"
    if proc.popen is not None and proc.popen.pid != proc.pid:
        return False, "pid does not match the process we spawned"
    cmd = _cmdline(proc.pid)
    if cmd is None:
        return False, "process not found"
    if proc.token not in cmd:
        return False, "pid is not running our worker (owner token missing)"
    return True, "owned"


class ProcessRegistry:
    """Every subprocess the supervisor owns, keyed by label."""

    def __init__(self) -> None:
        self.procs: dict[str, OwnedProcess] = {}

    def spawn(self, label: str, argv: list[str], token: str, **popen_kwargs) -> OwnedProcess:
        if token not in argv:
            raise ValueError("argv must carry the owner token so ownership can be verified")
        popen = subprocess.Popen(argv, **popen_kwargs)
        proc = OwnedProcess(label, popen.pid, token, popen)
        self.procs[label] = proc
        return proc

    def adopt(self, label: str, pid: int, token: str) -> OwnedProcess:
        """Track a process from before a restart (verified by token before any signal)."""
        proc = OwnedProcess(label, pid, token, None)
        self.procs[label] = proc
        return proc


def _wait_dead(proc: OwnedProcess, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_alive(proc):
            return True
        time.sleep(0.1)
    return not is_alive(proc)


def terminate_owned(proc: OwnedProcess, grace_s: float = 5.0, term_s: float = 3.0) -> dict:
    """Wait for a graceful exit, then SIGTERM, then SIGKILL - each only if still needed."""
    started = time.time()
    if proc.popen is None:  # re-adopted after a restart: prove it is still ours first
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            return {"verified": True, "method": "already exited", "pid": proc.pid}
        except PermissionError:
            pass
        owned, why = verify_ownership(proc)
        if not owned:
            return {"verified": False, "method": "refused", "detail": why, "pid": proc.pid}
    if _wait_dead(proc, grace_s):
        method = "graceful"
    else:
        owned, why = verify_ownership(proc)
        if not owned:
            return {"verified": False, "method": "refused", "detail": why, "pid": proc.pid}
        os.kill(proc.pid, signal.SIGTERM)
        if _wait_dead(proc, term_s):
            method = "sigterm"
        else:
            owned, why = verify_ownership(proc)
            if not owned:
                return {"verified": False, "method": "refused", "detail": why, "pid": proc.pid}
            os.kill(proc.pid, signal.SIGKILL)
            _wait_dead(proc, 3.0)
            method = "sigkill"
    if proc.popen is not None:
        try:
            proc.popen.wait(timeout=1)  # reap so no zombie is left behind
        except subprocess.TimeoutExpired:
            pass
    dead = not is_alive(proc)
    return {"verified": dead, "method": method, "pid": proc.pid,
            "exit_code": proc.popen.returncode if proc.popen is not None else None,
            "elapsed_s": round(time.time() - started, 2)}


def stop_agent(worker, reason: str, grace_s: float = 5.0, term_s: float = 3.0) -> dict:
    """STOP_AGENT for any WorkerAdapter, whatever model powers it.

    1. revoke: future tool calls from the worker are rejected
    2. cancel: ask the worker's loop to stop
    3. escalate only if needed (owned processes only), then verify it is gone
    """
    worker.revoke(reason)
    worker.request_cancel(reason)
    proc = getattr(worker, "process", None)
    if proc is None:  # in-process worker (scripted, or a hosted-model worker we run)
        return {"worker_id": worker.worker_id, "reason": reason, "method": "in-process cancel",
                "verified": not worker.is_running(), "pid": None}
    report = terminate_owned(proc, grace_s=grace_s, term_s=term_s)
    return {"worker_id": worker.worker_id, "reason": reason, **report}


# --------------------------------------------------------------------------- models
@dataclass
class ModelLeases:
    """Which of our components currently need which model."""
    holders: dict[str, set[str]] = field(default_factory=dict)

    def acquire(self, model_name: str, holder: str) -> None:
        self.holders.setdefault(model_name, set()).add(holder)

    def release(self, model_name: str, holder: str) -> None:
        self.holders.get(model_name, set()).discard(holder)

    def holders_of(self, model_name: str) -> set[str]:
        return set(self.holders.get(model_name, set()))


def stop_local_model(model, leases: ModelLeases, requester: str,
                     verify_timeout_s: float = 15.0) -> dict:
    """STOP_LOCAL_MODEL: unload one local model if nothing else of ours still uses it."""
    base = {"model": model.name, "provider": model.provider, "requester": requester}
    if not getattr(model, "local", False):
        return {**base, "stopped": False, "action": "refused",
                "reason": "hosted provider: we can cancel our own requests and stop our worker, "
                          "but we do not control the provider's model servers"}
    others = leases.holders_of(model.name) - {requester}
    if others:
        return {**base, "stopped": False, "action": "refused",
                "reason": f"model is shared with active holder(s): {', '.join(sorted(others))}"}
    if not model.loaded():
        return {**base, "stopped": True, "action": "none", "reason": "model was not loaded"}
    model.unload()
    deadline = time.time() + verify_timeout_s
    while time.time() < deadline and model.loaded():
        time.sleep(0.3)
    unloaded = not model.loaded()
    return {**base, "stopped": unloaded, "action": "unloaded via keep_alive=0",
            "reason": "no other holders" if unloaded else "still loaded after timeout",
            "model_files": "untouched"}
