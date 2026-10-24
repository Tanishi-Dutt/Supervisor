"""Put any agent loop under the supervisor in a few lines (Dev A).

    from flight_recorder import FlightRecorder

    with FlightRecorder("checkout-bot-42") as fr:
        while not agent.finished:
            action, args = agent.decide()
            result = agent.run(action, args)
            verdict = fr.step(action, args, result,
                              claimed=agent.status, test_passed=run_my_tests())
            if verdict == "stop":
                break
            if verdict != "continue":
                agent.recover(verdict)  # retry_with_backoff | change_strategy | rollback

Same watchdog, LLM diagnosis, policy veto, SQLite checkpoints and incident report as
run_demo.py. Re-opening the same run_id after a crash resumes its history.
"""
from types import SimpleNamespace

import fakes
from interfaces import LAST_N_STEPS
from run_demo import load_modules, new_record

TERMINAL = ("recovered", "stopped", "success_untouched")


class FlightRecorder:
    def __init__(self, run_id: str, fake: bool = False):
        self.run_id = run_id
        self.fns, self.faked = load_modules(fake)
        self.rec = self.fns["load_last"](run_id) or new_record(SimpleNamespace(state={}, name=run_id))
        self.meta = self.rec["meta"]
        self.report_path = None

    def step(self, action: str, args: dict, result, claimed: str | None = None,
             test_passed: bool = False) -> str:
        """Record one agent step. Returns "continue" or the policy-approved action."""
        hist, meta = self.rec["history"], self.meta
        retries = sum(e["action"] == "retry_with_backoff" for e in meta["events"])
        step = {"run_id": self.run_id, "step": len(hist) + 1, "action": action, "args": args,
                "result": str(result)[:500], "claimed_status": claimed,
                "evidence": {"test_passed": bool(test_passed), "retries_used": retries}}
        hist.append(step)
        window = hist[meta["window_start"]:]

        verdict = "continue"
        flag = self.fns["check"](window)
        if flag:
            try:
                proposal = self.fns["diagnose"](window[-LAST_N_STEPS:], flag)
            except Exception:  # LLM down or broken JSON: rule-based fallback
                proposal = fakes.diagnose(window[-LAST_N_STEPS:], flag)
            verdict = self.fns["approve"](proposal, step["evidence"], step)
            meta["interventions"] += 1
            meta["vetoes"] += verdict != proposal.get("action")
            meta["first_flag_step"] = meta["first_flag_step"] or step["step"]
            meta["events"].append({"step": step["step"], "flag": flag, "proposal": proposal,
                                   "action": verdict, "vetoed": verdict != proposal.get("action")})
            meta["window_start"] = len(hist)
            if verdict == "stop":
                meta["outcome"] = "stopped"
        self.fns["save"](self.run_id, step["step"], self.rec)
        return verdict

    def close(self) -> str | None:
        """Settle the outcome and write the incident report. Called by `with`."""
        hist, meta = self.rec["history"], self.meta
        if not hist or meta["reported"]:
            return self.report_path
        last = len(hist)
        if meta["outcome"] not in TERMINAL:
            passed = hist[-1]["evidence"]["test_passed"]
            meta["outcome"] = ("stopped" if not passed
                               else "recovered" if meta["interventions"] else "success_untouched")
            last += 1  # closing checkpoint: the store never rewrites an earlier step's outcome
            self.fns["save"](self.run_id, last, self.rec)
        self.report_path = self.fns["write_report"](self.run_id)
        meta["reported"] = True
        self.fns["save"](self.run_id, last, self.rec)
        return self.report_path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
