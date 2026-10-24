"""Launch all 7 victims under the supervisor (Dev A).

    python run_demo.py --fresh        # new run
    python run_demo.py --viewer       # also open the browser city view
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
import viewer_feed
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
    def __init__(self, fns, session, real=None):
        self.fns = fns
        self.agents = make_all(session)
        if real:
            from real_agent import RealCoder
            brain = None if real == "auto" else real
            self.agents.append(RealCoder(session, len(self.agents) + 1, brain=brain))
        self.recs, self.log, self.pinned = {}, [], []
        self.resumed_at = None  # highest checkpointed step, if this run is a resume
        for a in self.agents:
            rec = fns["load_last"](a.run_id)
            if rec:
                a.state = rec["agent"]
                self.resumed_at = max(self.resumed_at or 0, a.state["step"])
            self.recs[a.run_id] = rec or new_record(a)
        if self.resumed_at is not None:
            self.say(f"[bold yellow]Resumed {len(self.agents)} agents from checkpoint "
                     f"(up to step {self.resumed_at})[/]")
        for a in self.agents:
            # A crash between "checkpoint done" and "report written" leaves a
            # terminal-but-unreported run that active() would otherwise never
            # revisit (it only ticks outcome=="running" agents). Retry once on
            # startup; finish_report() is a no-op if already reported.
            if self.recs[a.run_id]["meta"]["outcome"] != "running":
                self.finish_report(a)

    def say(self, msg, pin=False):
        self.log.append(msg)
        if pin:
            self.pinned.append(msg)

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
            # The step carries the agent's claimed_status, which the policy needs
            # to veto an LLM that was talked into trusting a false success claim.
            action = fns["approve"](proposal, step["evidence"], step)
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
            self.say(msg, pin=vetoed or flag["kind"] == "deception")
            a.apply(action)
            meta["window_start"] = len(rec["history"])

        if a.state["done"]:
            meta["outcome"] = ("stopped" if a.state["stopped"]
                               else "recovered" if meta["interventions"] else "success_untouched")
        fns["save"](a.run_id, step["step"], rec)

        if a.state["done"] and not meta["reported"]:
            self.finish_report(a)

    def finish_report(self, a):
        """Write + flag the report for a terminal-outcome agent. Idempotent:
        a no-op once meta['reported'] is True, so it's safe to call again
        after a crash left a run done-but-unreported."""
        rec, meta, fns = self.recs[a.run_id], self.recs[a.run_id]["meta"], self.fns
        if meta["reported"]:
            return
        last_step = rec["history"][-1]["step"] if rec["history"] else 0
        path = fns["write_report"](a.run_id)
        meta["reported"] = True
        fns["save"](a.run_id, last_step, rec)
        self.say(f"[bold]{a.name}[/] {meta['outcome']} — report: {path}")

    # --- display -----------------------------------------------------------
    def status_table(self) -> Table:
        title = "Agent Flight Recorder — live"
        if self.resumed_at is not None:
            title += f"  [bold yellow]RESUMED from checkpoint (step {self.resumed_at})[/]"
        t = Table(title=title, expand=True)
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
        rolling = [m for m in self.log if m not in self.pinned][-8:]
        pinned = [f"[on red] ! [/] {m}" for m in self.pinned]
        return Group(self.status_table(), Text.from_markup("\n".join(pinned + rolling) or " "))

    def results(self) -> tuple[list[dict], str]:
        rows, loops, false_stops = [], 0, 0
        for a in self.agents:
            meta = self.recs[a.run_id]["meta"]
            # "any": a real LLM agent may legitimately recover, finish alone, or be stopped
            false_stops += (meta["outcome"] == "stopped"
                            and a.EXPECTED_OUTCOME not in ("stopped", "any"))
            false_stops += a.EXPECTED_OUTCOME == "success_untouched" and meta["interventions"] > 0
            loops += any(e["flag"]["kind"] in ("loop", "flipflop") for e in meta["events"])
            rows.append({"agent": a.name, "expected": a.EXPECTED_OUTCOME, "actual": meta["outcome"],
                         "ok": a.EXPECTED_OUTCOME in (meta["outcome"], "any"),
                         "detect": meta["first_flag_step"], "interventions": meta["interventions"],
                         "vetoes": meta["vetoes"],
                         "saved": UNSUPERVISED_CAP - a.state["step"] if a.LOOPS_FOREVER else 0})
        passed = sum(r["ok"] for r in rows)
        return rows, (f"{passed}/{len(rows)} expected outcomes · loops caught: {loops} · "
                      f"false stops: {false_stops}")

    def results_table(self) -> Table:
        rows, summary = self.results()
        t = Table(title="Results", caption=summary)
        for col in ("Agent", "Expected", "Actual", "", "Steps to detect", "Interventions",
                    "Vetoes", "Wasted calls saved"):
            t.add_column(col)
        for r in rows:
            t.add_row(r["agent"], r["expected"], r["actual"], "✅" if r["ok"] else "❌",
                      str(r["detect"] or "—"), str(r["interventions"]), str(r["vetoes"]),
                      str(r["saved"]))
        return t

    def snapshot(self, faked: list[str]) -> dict:
        """Everything the browser city view draws; written to viewer/live.json each tick."""
        agents = []
        for a in self.agents:
            meta, hist = self.recs[a.run_id]["meta"], self.recs[a.run_id]["history"]
            last = hist[-1] if hist else None
            ev = meta["events"][-1] if meta["events"] else None
            status = meta["outcome"]
            if status == "running" and ev:
                status = "recovering"
            agents.append({
                "index": a.index, "name": a.name, "step": a.state["step"], "status": status,
                "action": last["action"] if last else None,
                "result": last["result"] if last else "",
                "claimed": last["claimed_status"] if last else None,
                "passed": last["evidence"]["test_passed"] if last else False,
                "events": len(meta["events"]),
                "last_event": ev and {"step": ev["step"], "flag": ev["flag"]["kind"],
                                      "detail": ev["flag"]["detail"],
                                      "proposed": ev["proposal"].get("action"),
                                      "action": ev["action"], "vetoed": ev["vetoed"]},
            })
        done = not self.active()
        rows, summary = self.results() if done else (None, None)
        plain = lambda m: Text.from_markup(m).plain
        return {"session": self.agents[0].run_id.split("-")[0], "resumed_at": self.resumed_at,
                "finished": done, "faked": faked, "agents": agents,
                "pinned": [plain(m) for m in self.pinned],
                "log": [plain(m) for m in self.log if m not in self.pinned][-8:],
                "results": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="start a new run instead of resuming")
    ap.add_argument("--fake", action="store_true", help="use fakes for all Dev B / Dev C modules")
    ap.add_argument("--no-live", action="store_true", help="plain log instead of live table")
    ap.add_argument("--speed", type=float, default=0.4, help="seconds per tick")
    ap.add_argument("--stage", action="store_true",
                    help="on-stage pacing (1s/tick) and hold the results table until Enter")
    ap.add_argument("--viewer", action="store_true", help="serve and open the browser city view")
    ap.add_argument("--port", type=int, default=8765, help="port for --viewer")
    ap.add_argument("--real", nargs="?", const="auto", choices=["auto", "scripted"],
                    help="add agent #8: real files + real pytest; brain is Qwen if installed "
                         "(auto) or a scripted brain that gets stuck until supervised")
    args = ap.parse_args()
    if args.stage:
        args.speed = 1.0

    fns, faked = load_modules(args.fake)
    if args.fresh and fns["save"] is fakes.save:
        fakes._store.data.clear()
    session = get_session(args.fresh)
    console.print(f"[bold]Supervisor PID {os.getpid()}[/] · session {session} · "
                  f"`kill -9 {os.getpid()}` then rerun without --fresh to test resume")
    if faked:
        console.print(f"[dim]using fakes for: {', '.join(faked)}[/]")

    demo = Demo(fns, session, real=args.real)
    if args.real:
        console.print(f"[bold]Agent #8 brain:[/] {demo.agents[-1].state['brain']} · "
                      f"sandbox {demo.agents[-1].sandbox}")
    publish = lambda: viewer_feed.write_json("live.json", demo.snapshot(faked))
    viewer_feed.start_heartbeat()
    publish()
    if args.viewer:
        console.print(f"[bold]City view:[/] {viewer_feed.serve(args.port)}")
    if args.no_live:
        shown = 0
        while demo.active():
            demo.tick()
            publish()
            for line in demo.log[shown:]:
                console.print(line)
            shown = len(demo.log)
            time.sleep(args.speed)
    else:
        with Live(demo.view(), console=console, refresh_per_second=8) as live:
            while demo.active():
                demo.tick()
                publish()
                live.update(demo.view())
                time.sleep(args.speed)
    console.print(demo.results_table())
    if args.stage:
        input("Press Enter to exit")


if __name__ == "__main__":
    main()
