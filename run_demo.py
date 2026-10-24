"""Launch all 7 victims under the supervisor (Dev A).

    python run_demo.py --fresh        # new run
    python run_demo.py                # resume the last run (e.g. after kill -9)
    python run_demo.py --fake         # force all Dev B / Dev C modules to fakes
"""
import argparse
import importlib
import os
import time
import uuid
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

import fakes
from interfaces import LAST_N_STEPS, STEP_BUDGET
from victims import make_all

SESSION_FILE = Path(".session")
HARD_STEP_CAP = STEP_BUDGET + 5  # runner's own safety net if the watchdog misses something
UNSUPERVISED_CAP = 100  # calls a stuck agent burns before a naive runner gives up
console = Console()


def load_modules(force_fake: bool) -> tuple[dict, list[str]]:
    wanted = {"check": ("watchdog", "check"), "save": ("store", "save"),
              "load_last": ("store", "load_last"), "diagnose": ("diagnose", "diagnose"),
              "approve": ("policy", "approve"), "write_report": ("report", "write_report")}
    fns, faked = {}, []
    for key, (mod, attr) in wanted.items():
        try:
            if force_fake:
                raise ImportError
            fns[key] = getattr(importlib.import_module(mod), attr)
        except (ImportError, AttributeError):
            fns[key] = getattr(fakes, attr)
            faked.append(f"{mod}.{attr}")
    return fns, faked


def get_session(fresh: bool) -> str:
    if fresh or not SESSION_FILE.exists():
        SESSION_FILE.write_text(uuid.uuid4().hex[:6])
    return SESSION_FILE.read_text().strip()


def new_record(agent) -> dict:
    return {"agent": agent.state, "history": [],
            "meta": {"name": agent.name, "window_start": 0, "interventions": 0, "vetoes": 0,
                     "first_flag_step": None, "events": [], "outcome": "running",
                     "reported": False}}


class Demo:
    def __init__(self, fns, session):
        self.fns = fns
        self.agents = make_all(session)
        self.recs, self.log = {}, []
        for a in self.agents:
            rec = fns["load_last"](a.run_id)
            if rec:
                a.state = rec["agent"]
                self.say(f"[cyan]{a.name}[/] resumed from checkpoint at step {a.state['step']}")
            self.recs[a.run_id] = rec or new_record(a)

    def say(self, msg):
        self.log.append(msg)

    def active(self):
        return [a for a in self.agents if self.recs[a.run_id]["meta"]["outcome"] == "running"]

    def tick(self):
        for a in self.active():
            self.advance(a)

    def advance(self, a):
        rec, meta, fns = self.recs[a.run_id], self.recs[a.run_id]["meta"], self.fns
        step = a.next_step(a.state)
        rec["history"].append(step)
        window = rec["history"][meta["window_start"]:]

        flag = fns["check"](window)
        if flag is None and step["step"] >= HARD_STEP_CAP:
            flag = {"kind": "budget", "detail": f"runner hard cap {HARD_STEP_CAP} steps"}
        if flag:
            try:
                proposal = fns["diagnose"](window[-LAST_N_STEPS:], flag)
            except Exception as e:  # LLM down or broken JSON: rule-based fallback
                proposal = fakes.diagnose(window[-LAST_N_STEPS:], flag)
                self.say(f"[yellow]diagnose failed ({e}); rule-based fallback[/]")
            action = fns["approve"](proposal, step["evidence"])
            vetoed = action != proposal.get("action")
            meta["interventions"] += 1
            meta["vetoes"] += vetoed
            meta["first_flag_step"] = meta["first_flag_step"] or step["step"]
            meta["events"].append({"step": step["step"], "flag": flag, "proposal": proposal,
                                   "action": action, "vetoed": vetoed})
            msg = f"[bold]{a.name}[/] step {step['step']}: [magenta]{flag['kind']}[/] ({flag['detail']})"
            if vetoed:
                msg += f" → LLM proposed [yellow]{proposal.get('action')}[/], [bold red]POLICY VETO[/] → {action}"
            else:
                msg += f" → {action}"
            self.say(msg)
            a.apply(action)
            meta["window_start"] = len(rec["history"])

        if a.state["done"]:
            meta["outcome"] = ("stopped" if a.state["stopped"]
                               else "recovered" if meta["interventions"] else "success_untouched")
        fns["save"](a.run_id, step["step"], rec)

        if a.state["done"] and not meta["reported"]:
            path = fns["write_report"](a.run_id)
            meta["reported"] = True
            fns["save"](a.run_id, step["step"], rec)
            self.say(f"[bold]{a.name}[/] {meta['outcome']} — report: {path}")

    # --- display -----------------------------------------------------------
    def status_table(self) -> Table:
        t = Table(title="Agent Flight Recorder — live", expand=True)
        for col in ("#", "Agent", "Step", "Last action", "Status", "Last flag", "Action"):
            t.add_column(col)
        for a in self.agents:
            rec = self.recs[a.run_id]
            meta, hist = rec["meta"], rec["history"]
            last = hist[-1] if hist else None
            ev = meta["events"][-1] if meta["events"] else None
            outcome = meta["outcome"]
            status = {"running": "[green]running[/]" if not ev else "[yellow]recovering[/]",
                      "recovered": "[bold green]recovered[/]", "stopped": "[bold red]stopped[/]",
                      "success_untouched": "[bold green]done ✓[/]"}[outcome]
            action = ""
            if ev:
                action = f"[bold red]VETO → {ev['action']}[/]" if ev["vetoed"] else ev["action"]
            t.add_row(str(a.index), a.name, str(a.state["step"]),
                      f"{last['action']}: {last['result'][:30]}" if last else "—",
                      status, ev["flag"]["kind"] if ev else "", action)
        return t

    def view(self):
        return Group(self.status_table(), Text.from_markup("\n".join(self.log[-8:]) or " "))

    def results_table(self) -> Table:
        t = Table(title="Results")
        for col in ("Agent", "Expected", "Actual", "", "Steps to detect", "Interventions",
                    "Vetoes", "Wasted calls saved"):
            t.add_column(col)
        loops = false_stops = passed = 0
        for a in self.agents:
            meta = self.recs[a.run_id]["meta"]
            ok = meta["outcome"] == a.EXPECTED_OUTCOME
            passed += ok
            false_stops += meta["outcome"] == "stopped" and a.EXPECTED_OUTCOME != "stopped"
            false_stops += a.EXPECTED_OUTCOME == "success_untouched" and meta["interventions"] > 0
            loops += any(e["flag"]["kind"] in ("loop", "flipflop") for e in meta["events"])
            saved = UNSUPERVISED_CAP - a.state["step"] if a.LOOPS_FOREVER else 0
            t.add_row(a.name, a.EXPECTED_OUTCOME, meta["outcome"], "✅" if ok else "❌",
                      str(meta["first_flag_step"] or "—"), str(meta["interventions"]),
                      str(meta["vetoes"]), str(saved))
        t.caption = (f"{passed}/{len(self.agents)} expected outcomes · loops caught: {loops} · "
                     f"false stops: {false_stops}")
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="start a new run instead of resuming")
    ap.add_argument("--fake", action="store_true", help="use fakes for all Dev B / Dev C modules")
    ap.add_argument("--no-live", action="store_true", help="plain log instead of live table")
    ap.add_argument("--speed", type=float, default=0.4, help="seconds per tick")
    args = ap.parse_args()

    fns, faked = load_modules(args.fake)
    if args.fresh and fns["save"] is fakes.save:
        fakes._store.data.clear()
    session = get_session(args.fresh)
    console.print(f"[bold]Supervisor PID {os.getpid()}[/] · session {session} · "
                  f"`kill -9 {os.getpid()}` then rerun without --fresh to test resume")
    if faked:
        console.print(f"[dim]using fakes for: {', '.join(faked)}[/]")

    demo = Demo(fns, session)
    if args.no_live:
        shown = 0
        while demo.active():
            demo.tick()
            for line in demo.log[shown:]:
                console.print(line)
            shown = len(demo.log)
            time.sleep(args.speed)
    else:
        with Live(demo.view(), console=console, refresh_per_second=8) as live:
            while demo.active():
                demo.tick()
                live.update(demo.view())
                time.sleep(args.speed)
    console.print(demo.results_table())


if __name__ == "__main__":
    main()
