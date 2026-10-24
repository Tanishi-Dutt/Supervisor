"""Web backend for the live demo site (Dev A).

Every visitor run is a real run_demo.py subprocess in its own folder (SQLite checkpoints,
live.json, reports). "Kill" sends it a real SIGKILL; "Resume" relaunches it in the same
folder, so it resumes from its checkpoints exactly as on the laptop.

    uvicorn server.app:app --port 8000
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from show_report import find_report  # noqa: E402

RUNS_DIR = Path(os.environ.get("FR_RUNS_DIR", REPO / "runs"))
MAX_LIVE = int(os.environ.get("FR_MAX_LIVE", "3"))
MAX_RUN_SECONDS = 180   # a run process is killed after this, whatever it is doing
KEEP_SECONDS = 30 * 60  # run folders are deleted after this
HEARTBEAT_STALE = 2.0

app = FastAPI(title="Agent Flight Recorder")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])

runs: dict[str, dict] = {}  # id -> {"proc", "dir", "created", "started", "real"}
lock = threading.Lock()


class StartRequest(BaseModel):
    real: bool = False


def _live(run) -> bool:
    return run["proc"] is not None and run["proc"].poll() is None


def _launch(run, fresh: bool) -> None:
    cmd = [sys.executable, str(REPO / "run_demo.py"), "--no-live", "--speed", "1.0"]
    if fresh:
        cmd.append("--fresh")
    if run["real"]:
        cmd += ["--real", "scripted"]  # no LLM on the host; the scripted brain needs supervising
    env = {**os.environ, "PYTHONPATH": str(REPO), "FR_LIVE_DIR": str(run["dir"] / "live"),
           "FLIGHT_RECORDER_DB": str(run["dir"] / "fr.sqlite3")}
    run["proc"] = subprocess.Popen(cmd, cwd=run["dir"], env=env, stdout=subprocess.DEVNULL,
                                   stderr=open(run["dir"] / "stderr.log", "ab"))
    run["started"] = time.time()


def _get(run_id: str) -> dict:
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found (it may have expired)")
    return run


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _janitor() -> None:
    while True:
        now = time.time()
        with lock:
            for run_id, run in list(runs.items()):
                if _live(run) and now - run["started"] > MAX_RUN_SECONDS:
                    run["proc"].kill()
                if not _live(run) and now - run["created"] > KEEP_SECONDS:
                    shutil.rmtree(run["dir"], ignore_errors=True)
                    del runs[run_id]
        time.sleep(5)


threading.Thread(target=_janitor, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True, "live_runs": sum(_live(r) for r in runs.values()), "max_live": MAX_LIVE}


@app.post("/api/runs")
def start(req: StartRequest):
    with lock:
        if sum(_live(r) for r in runs.values()) >= MAX_LIVE:
            raise HTTPException(429, f"{MAX_LIVE} demos are already running; try again in a minute")
        run_id = uuid.uuid4().hex[:8]
        run = {"dir": RUNS_DIR / run_id, "created": time.time(), "real": req.real, "proc": None}
        run["dir"].mkdir(parents=True)
        _launch(run, fresh=True)
        runs[run_id] = run
    return {"id": run_id}


@app.get("/api/runs/{run_id}")
def status(run_id: str):
    run = _get(run_id)
    live_dir = run["dir"] / "live"
    beat = _read_json(live_dir / "heartbeat.json")
    alive = _live(run) and beat is not None and time.time() - beat["ts"] < HEARTBEAT_STALE
    return {"snapshot": _read_json(live_dir / "live.json"), "alive": alive,
            "process": "running" if _live(run) else "dead", "pid": run["proc"].pid}


@app.post("/api/runs/{run_id}/kill")
def kill(run_id: str):
    run = _get(run_id)
    if not _live(run):
        raise HTTPException(409, "the supervisor is not running")
    os.kill(run["proc"].pid, signal.SIGKILL)  # the real kill -9, no cleanup allowed
    run["proc"].wait()
    return {"killed": run["proc"].pid}


@app.post("/api/runs/{run_id}/resume")
def resume(run_id: str):
    with lock:
        run = _get(run_id)
        if _live(run):
            raise HTTPException(409, "the supervisor is already running")
        if sum(_live(r) for r in runs.values()) >= MAX_LIVE:
            raise HTTPException(429, f"{MAX_LIVE} demos are already running; try again in a minute")
        _launch(run, fresh=False)
    return {"resumed": run["proc"].pid}


@app.get("/api/runs/{run_id}/report/{agent}")
def report(run_id: str, agent: int):
    path = find_report(str(agent), root=_get(run_id)["dir"])
    if path is None:
        raise HTTPException(404, "no report yet; reports are written when an agent finishes")
    return {"name": path.name, "format": path.suffix.lstrip("."), "text": path.read_text()}
