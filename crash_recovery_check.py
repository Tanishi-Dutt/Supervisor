"""Genuine crash-and-resume check for SupervisorAgent with a LIVE worker.

1. Launch `agent_demo.py live` (supervisor + real Qwen worker) as a subprocess.
2. Wait until SQLite holds at least --min-steps worker steps.
3. SIGKILL exactly that supervisor process (the PID we launched; nothing else).
4. Confirm its worker subprocess exits by itself (it watches its parent).
5. Relaunch with --resume and wait for it to finish.
6. Verify: resumed from the checkpoint (not step 1), contiguous history with no
   duplicated steps, verified outcome, exactly one report.

    python crash_recovery_check.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import store
from agent_demo import RUNS
from kill_controls import OwnedProcess, is_alive

ROOT = Path(__file__).resolve().parent


def wait_for(cond, timeout: float, every: float = 0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cond()
        if value:
            return value
        time.sleep(every)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()
    session = f"crash-{uuid.uuid4().hex[:6]}"
    sup_run, worker_run = f"{session}-agent-coder", f"{session}-worker-coder"
    RUNS.mkdir(parents=True, exist_ok=True)
    log1, log2 = RUNS / f"{session}-first.log", RUNS / f"{session}-resume.log"
    env = {**os.environ, "COLUMNS": "150", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(ROOT / "agent_demo.py"), "live", "--session", session]

    print(f"[1] launching supervisor for session {session}")
    with open(log1, "w") as out:
        first = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)

    def progress():
        cp = store.load_checkpoint(sup_run)
        return cp if cp and len(cp["state"]["history"]) >= args.min_steps else None

    cp = wait_for(progress, args.timeout / 2)
    if cp is None:
        first.kill()
        print("FAIL: worker never reached the minimum number of steps")
        return 1
    steps_before = [s["step"] for s in cp["state"]["history"]]
    worker = cp["state"]["agent"]
    print(f"[2] supervisor checkpoint has worker steps {steps_before}; worker pid {worker['pid']}")

    print(f"[3] SIGKILL supervisor pid {first.pid} (the exact process this script launched)")
    os.kill(first.pid, signal.SIGKILL)
    first.wait()
    orphan = OwnedProcess("coder", worker["pid"], worker["token"])
    gone = wait_for(lambda: not is_alive(orphan), 30)
    print(f"[4] worker subprocess exited on its own after supervisor death: {bool(gone)}")
    if not gone:
        print("FAIL: worker outlived the supervisor")
        return 1
    worker_cp = store.load_checkpoint(worker_run)
    print(f"    worker's own checkpoint: step {worker_cp['state']['step']}")

    print("[5] relaunching with --resume")
    with open(log2, "w") as out:
        second = subprocess.run(cmd + ["--resume"], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                env=env, timeout=args.timeout)
    resumed_log = log2.read_text()
    final = store.load_checkpoint(sup_run)
    meta, history = final["state"]["meta"], final["state"]["history"]
    numbers = [s["step"] for s in history]
    reports = sorted((ROOT / "reports").glob(f"{session}-*"))
    worker_log = (RUNS / session / "events.jsonl").read_text()
    restarts = worker_log.count('"type": "worker_started"')
    checks = {
        "resume exit code 0": second.returncode == 0,
        "restored from SQLite checkpoint": "RESTORED FROM SQLITE CHECKPOINT" in resumed_log,
        "worker resumed mid-task (not step 1)": f'"resumed": true' in worker_log,
        "history contiguous, no duplicate steps": numbers == list(range(1, len(numbers) + 1)),
        "pre-crash steps preserved": numbers[:len(steps_before)] == steps_before,
        "continued past the crash point": len(numbers) > len(steps_before),
        "completed with independent verification": meta["status"] == "completed"
        and (meta.get("verification") or {}).get("test_passed") is True,
        "exactly one report": len(reports) == 1,
    }
    print(f"[6] final status={meta['status']} outcome={meta['outcome']} steps={numbers} "
          f"worker starts={restarts} verification={meta.get('verification')}")
    for name, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {name}")
    print(f"    logs: {log1} , {log2}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
