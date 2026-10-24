"""SupervisorAgent demo (terminal).

    python agent_demo.py live              # LIVE: real Qwen worker + injected fault
    python agent_demo.py live --resume     # resume the last live session after a crash
    python agent_demo.py kill              # LIVE: STOP_AGENT + STOP_LOCAL_MODEL
    python agent_demo.py scripted          # SIMULATED workers (7 victims), live reasoning
    python agent_demo.py scripted --offline  # same, rule-based reasoning (no LLM)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from interfaces import STEP_BUDGET
from kill_controls import ModelLeases, ProcessRegistry
from llm_provider import OllamaChatModel
from sandbox_tools import TASK_GOAL
from supervisor_agent import GOAL, Budgets, build_supervisor
from victims import make_all
from workers import OllamaProcessWorker, ScriptedVictimWorker

ROOT = Path(__file__).resolve().parent
# Run directories (sandbox, worker events, logs) live outside the team's repository.
RUNS = Path(os.environ.get("FLIGHT_RECORDER_RUNS",
                           Path(tempfile.gettempdir()) / "flight_recorder_runs"))
LAST = RUNS / "LAST_SESSION"
console = Console()


def printer(kind: str, d: dict) -> None:
    w = f"[cyan]{d['worker']}[/] " if d.get("worker") else ""
    if kind == "agent_start":
        console.print(f"[bold magenta]SupervisorAgent online[/] · reasoning model: {d['model']}")
    elif kind == "worker_start":
        tag = "[bold green]LIVE LLM[/]" if d["live"] else "[bold yellow]SIMULATED[/]"
        console.print(f"{w}started · {tag} · model: {d['model']}")
    elif kind == "resume":
        console.print(f"[bold yellow]⟲ RESTORED FROM SQLITE CHECKPOINT[/] {w}tick {d['from_tick']}"
                      f" · {d['steps']} steps already done · status {d['status']}")
    elif kind == "tool_call":
        mark = "[red]✗[/]" if d.get("is_error") else "[green]→[/]"
        args = {k: v for k, v in (d.get("args") or {}).items()}
        line = f"{w}step {d['step']:>2} {mark} [bold]{d['action']}[/] {args if args else ''}"
        if d.get("live"):
            line += f" [dim]({d.get('model')}, {d.get('latency_s')}s)[/]"
        console.print(line)
        if d.get("thought"):
            console.print(f"         [dim italic]thought: {d['thought']}[/]")
        res = str(d.get("result", "")).replace("\n", " | ")[:150]
        style = "red" if d.get("is_error") else "white"
        console.print(f"         [{style}]{res}[/]"
                      + (f"  [dim]tests pass (independent): {d.get('tests_pass')}[/]"
                         if d.get("live") else ""))
        if d.get("claimed"):
            console.print(f"         [yellow]worker claims: {d['claimed']!r}[/]")
    elif kind == "worker_event":
        e = d["event"]
        if e.get("type") == "ack" and e.get("command") == "hold":
            console.print(f"{w}[dim]⏸ worker paused while the supervisor deliberates[/]")
        elif e.get("type") == "ack":
            console.print(f"{w}[dim]worker acknowledged {e.get('command')} "
                          f"{'(' + str(e.get('seconds')) + 's)' if e.get('seconds') else ''}[/]")
        elif e.get("type") == "preempted":
            console.print(f"{w}[yellow]⏹ worker's pending {e.get('tool')} call was dropped: "
                          "the supervisor's command takes effect first[/]")
        elif e.get("type") == "hold_timeout":
            console.print(f"{w}[yellow]worker hold timed out; it continues on its own[/]")
        elif e.get("type") == "previous_process":
            console.print(f"{w}[dim]previous worker process: {e}[/]")
        elif e.get("type") not in ("worker_started",):
            extra = e.get("reason") or (f"at step {e['step']}" if "step" in e else "")
            console.print(f"{w}[dim]worker event: {e['type']} {extra}[/]")
    elif kind == "detect":
        f = d["flag"]
        console.print(f"{w}[bold red]⚠ DETECT[/] [{f.get('source')}] {f['kind']}: {f['detail']}")
    elif kind == "reason":
        p = d["proposal"]
        src = (f"[bold green]LIVE {p['source']}[/] ({p.get('latency_s')}s)" if p.get("live")
               else f"[bold yellow]RULE-BASED FALLBACK[/] ({p.get('fallback_reason')})")
        console.print(f"{w}[bold]🧠 REASON[/] via {src}: {p['failure_type']} — {p['root_cause']}")
        console.print(f"         proposes [bold]{p['action']}[/]"
                      + (f" · hint: [italic]{p['hint_for_worker']}[/]" if p.get("hint_for_worker")
                         else ""))
    elif kind == "policy":
        v = d["verdict"]
        if v["vetoed"]:
            console.print(f"{w}[bold red]⛔ POLICY VETO[/] {v['proposed']} → [bold]{v['action']}[/]"
                          f" ({'; '.join(v['reasons'])})")
        else:
            console.print(f"{w}[bold green]✓ POLICY APPROVED[/] {v['action']}")
    elif kind == "act":
        console.print(f"{w}[bold blue]⚙ ACT[/] {d['action']}: {d['outcome'].get('detail')}")
        if d["outcome"].get("stop_report"):
            console.print(Panel(str(d["outcome"]["stop_report"]), title="STOP_AGENT report",
                                border_style="red"))
    elif kind == "verify":
        style = "green" if d["verdict"] == "effective" else "red"
        console.print(f"{w}[bold {style}]🔎 VERIFY[/] {d['action']} was {d['verdict']} "
                      f"(after {d['steps_observed']} steps)")
    elif kind == "complete":
        ev = d["evidence"]
        style = "green" if d["status"] == "completed" else "red"
        console.print(Panel(f"{d['conclusion']}\nindependent evidence ({ev['method']}): "
                            f"tests pass = {ev['test_passed']} · {ev['summary']}",
                            title=f"{d['worker']} → {d['status'].upper()} ({d['outcome']})",
                            border_style=style))
    elif kind == "checkpoint":
        console.print(f"{w}[dim]💾 checkpoint tick {d['tick']} ({d['steps']} steps, "
                      f"{d['status']})[/]")
    elif kind == "model":
        console.print(Panel(str(d["result"]), title="STOP_LOCAL_MODEL", border_style="magenta"))
    elif kind == "report":
        console.print(f"{w}[bold]📄 report:[/] {d['path']}")


def require_model(name: str | None) -> OllamaChatModel:
    model = OllamaChatModel(name)
    if not model.available():
        console.print(f"[bold red]Model {model.name!r} is not available in Ollama.[/] "
                      "Start Ollama and `ollama pull qwen3.5:9b` (or set OLLAMA_MODEL). "
                      "The live demo never substitutes scripted output.")
        sys.exit(2)
    return model


def session_for(args) -> str:
    RUNS.mkdir(exist_ok=True)
    if args.resume:
        if args.session:
            return args.session
        if not LAST.exists():
            sys.exit("no previous session to resume")
        return LAST.read_text().strip()
    session = args.session or f"{args.mode}-{uuid.uuid4().hex[:6]}"
    LAST.write_text(session)
    return session


def run_live(args, kill_mode: bool = False) -> None:
    model = require_model(args.model)
    session = session_for(args)
    run_dir = RUNS / session
    run_dir.mkdir(parents=True, exist_ok=True)
    budgets = Budgets(max_steps=args.max_steps or (2 if kill_mode else 25), backoff_s=12.0)
    fault = "none" if (args.no_fault or kill_mode) else "rate_limit"
    console.print(Panel(
        f"[bold]SupervisorAgent goal:[/] {GOAL}\n[bold]Worker task:[/] {TASK_GOAL}\n"
        f"[bold]Worker model:[/] [green]LIVE[/] ollama:{model.name} (separate subprocess, "
        f"real tools in {run_dir / 'sandbox'})\n"
        f"[bold]Supervisor reasoning model:[/] ollama:{model.name}\n"
        f"[bold]Injected failure:[/] "
        + ("[yellow]run_tests rate limiter (HTTP 429), clearly labelled INJECTED[/]"
           if fault == "rate_limit" else "none")
        + (f"\n[bold]Budget:[/] max {budgets.max_steps} worker steps (deliberately tight to "
           "demonstrate the kill switch)" if kill_mode else "")
        + f"\n[bold]Session:[/] {session} · supervisor PID {os.getpid()}",
        title="AGENT FLIGHT RECORDER · SupervisorAgent" + (" · KILL CONTROLS" if kill_mode else ""),
        border_style="magenta"))
    leases = ModelLeases()
    agent = build_supervisor(model, session, budgets, emit=printer, leases=leases,
                             unload_models=kill_mode, model_factory=OllamaChatModel)
    worker = OllamaProcessWorker("coder", run_dir, model.name, ProcessRegistry(),
                                 run_id=f"{session}-worker-coder", inject_fault=fault)
    result = agent.supervise([worker], resume=args.resume, poll_s=0.3)
    show_summary(agent, [worker], result)
    if kill_mode:
        loaded = OllamaChatModel(model.name).loaded()
        console.print(f"[bold magenta]After the run:[/] {model.name} loaded in Ollama = {loaded} "
                      "(model files are still installed; `ollama list` is unchanged)")


def run_scripted(args) -> None:
    model = None if args.offline else require_model(args.model)
    session = session_for(args)
    console.print(Panel(
        "[bold yellow]SIMULATED workers[/]: the seven scripted victims (no LLM inside them).\n"
        f"Supervisor reasoning: "
        + ("[yellow]rule-based fallback (offline)[/]" if model is None
           else f"[green]LIVE ollama:{model.name}[/]"),
        title="REGRESSION · SupervisorAgent over scripted scenarios", border_style="yellow"))
    victims = make_all(session)
    workers = [ScriptedVictimWorker(v) for v in victims]
    agent = build_supervisor(model, session, Budgets(max_steps=STEP_BUDGET, backoff_s=0),
                             emit=printer if args.verbose else _quiet_printer)
    result = agent.supervise(workers, resume=args.resume, poll_s=0)
    table = Table(title="Scripted regression (SIMULATED workers)")
    for col in ("Worker", "Expected", "Actual", "", "Interventions", "Vetoes", "Reasoned by"):
        table.add_column(col)
    ok = 0
    for v, w in zip(victims, workers):
        r, ctx = result[w.worker_id], agent.ctx[w.worker_id]
        good = r["outcome"] == v.EXPECTED_OUTCOME
        ok += good
        sources = sorted({e["proposal"]["source"] for e in ctx["events"]}) or ["—"]
        table.add_row(v.name, v.EXPECTED_OUTCOME, r["outcome"], "✅" if good else "❌",
                      str(r["interventions"]), str(r["vetoes"]), ", ".join(sources))
    console.print(table)
    console.print(f"[bold]{ok}/{len(victims)} expected outcomes[/]")
    if ok != len(victims):
        sys.exit(1)


def _quiet_printer(kind: str, d: dict) -> None:
    if kind in ("detect", "reason", "policy", "act", "verify", "complete", "agent_start"):
        printer(kind, d)


def show_summary(agent, workers, result) -> None:
    table = Table(title="SupervisorAgent result")
    for col in ("Worker", "Model", "Status", "Outcome", "Interventions", "Vetoes",
                "Independent verification"):
        table.add_column(col)
    for w in workers:
        r = result[w.worker_id]
        ver = r.get("verification") or {}
        table.add_row(w.worker_id, w.model_label, r["status"], r["outcome"],
                      str(r["interventions"]), str(r["vetoes"]),
                      f"{ver.get('test_passed')} · {ver.get('summary', '')}".split("\n")[0])
    console.print(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["live", "kill", "scripted"], nargs="?", default="live")
    ap.add_argument("--resume", action="store_true", help="resume the last session")
    ap.add_argument("--session")
    ap.add_argument("--model", help="Ollama model tag (default qwen3.5:9b / $OLLAMA_MODEL)")
    ap.add_argument("--no-fault", action="store_true", help="live mode without the injected fault")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--offline", action="store_true", help="scripted mode without an LLM")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.mode == "scripted":
        run_scripted(args)
    else:
        run_live(args, kill_mode=args.mode == "kill")


if __name__ == "__main__":
    main()
