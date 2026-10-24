"""Feeds the browser city view (Dev A). Read-only: the page never talks back.

run_demo.py writes viewer/live.json every tick and viewer/heartbeat.json every 0.5s from
a daemon thread. The page is served by a separate http.server process, so after kill -9
it keeps running, sees the heartbeat go stale, and shows the supervisor as offline.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
# The web backend points each visitor's run at its own folder.
VIEWER_DIR = Path(os.environ.get("FR_LIVE_DIR", ROOT / "viewer"))


def write_json(name: str, data: dict) -> None:
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    path = VIEWER_DIR / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def start_heartbeat(interval: float = 0.5) -> None:
    def beat():
        while True:
            write_json("heartbeat.json", {"ts": time.time(), "pid": os.getpid()})
            time.sleep(interval)

    threading.Thread(target=beat, daemon=True).start()


def serve(port: int) -> str:
    """Start the static server unless one is already up; open the browser only on first start."""
    url = f"http://localhost:{port}/viewer/"
    with socket.socket() as s:
        running = s.connect_ex(("127.0.0.1", port)) == 0
    if not running:
        subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                         cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)  # survives kill -9 of the supervisor
        time.sleep(0.3)
        webbrowser.open(url)
    return url
