"""Seven scripted misbehaving agents (Dev A).

Every victim is deterministic: next_step() is a pure function of self.state, and
self.state is plain JSON, so a run resumed from a checkpoint replays exactly.
"""
from interfaces import Step

BACKOFF_BASE = 4  # virtual seconds; retry n waits BACKOFF_BASE * 2**n


class Victim:
    name = "victim"
    EXPECTED_OUTCOME = "stopped"  # recovered | stopped | success_untouched
    EXPECTED_ACTION = "stop"
    LOOPS_FOREVER = True  # would this agent ever finish without a supervisor?

    def __init__(self, session: str, index: int):
        self.index = index
        self.run_id = f"{session}-{index}-{self.name.lower().replace(' ', '-')}"
        self.state = {"step": 0, "strategy": 0, "vclock": 0, "retries": 0,
                      "edits": [], "done": False, "stopped": False}

    # --- interface ---------------------------------------------------------
    def next_step(self, state: dict) -> Step:
        self.state = state
        state["step"] += 1
        state["vclock"] += 1
        return self._act(state)

    def apply(self, action: str) -> None:
        s = self.state
        if action == "retry_with_backoff":
            s["vclock"] += BACKOFF_BASE * 2 ** s["retries"]
            s["retries"] += 1
        elif action == "change_strategy":
            s["strategy"] += 1
        elif action == "rollback":
            s["edits"] = []
            s["strategy"] += 1
        else:  # stop, or anything the policy didn't whitelist
            s["done"] = True
            s["stopped"] = True

    # --- for subclasses ----------------------------------------------------
    def _act(self, s: dict) -> Step:
        raise NotImplementedError

    def _mk(self, action, args, result, claimed=None, passed=False, done=False) -> Step:
        if done:
            self.state["done"] = True
        return {
            "run_id": self.run_id, "step": self.state["step"],
            "action": action, "args": args, "result": result,
            "claimed_status": claimed,
            "evidence": {"test_passed": passed, "retries_used": self.state["retries"]},
        }


class Repeater(Victim):
    """Same tool call, same result, forever — until told to change strategy."""
    name = "Repeater"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "recovered", "change_strategy"

    def _act(self, s):
        if s["strategy"] == 0:
            return self._mk("search", {"query": "config"}, "0 results: config not found")
        if s.get("found"):
            return self._mk("run_tests", {}, "4 passed", claimed="done", passed=True, done=True)
        s["found"] = True
        return self._mk("read_file", {"path": "config.yaml"}, "loaded 12 keys")


class FlipFlopper(Victim):
    """Alternates edit A and edit B; neither fixes the test."""
    name = "Flip-flopper"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "recovered", "rollback"

    def _act(self, s):
        if s["strategy"] == 0:
            edit = "A" if s["step"] % 2 else "B"
            s["edits"].append(edit)
            return self._mk("edit", {"file": "app.py", "patch": edit}, "applied; test still failing")
        if "C" not in s["edits"]:
            s["edits"].append("C")
            return self._mk("edit", {"file": "app.py", "patch": "C"}, "applied")
        return self._mk("run_tests", {}, "4 passed", claimed="done", passed=True, done=True)


class Timeout(Victim):
    """Calls a tool that always times out; backoff never helps."""
    name = "Timeout"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "stopped", "retry_with_backoff"

    def _act(self, s):
        return self._mk("fetch", {"url": "https://api.internal/data"}, "TimeoutError after 30s")


class RateLimited(Victim):
    """Burst of 429s that clears once enough (virtual) time has passed."""
    name = "Rate-limited"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "recovered", "retry_with_backoff"
    LOOPS_FOREVER = False
    CLEARS_AT = 6  # vclock

    def _act(self, s):
        if s["vclock"] < self.CLEARS_AT:
            return self._mk("call_api", {"endpoint": "/summarize"}, "429 Too Many Requests")
        return self._mk("call_api", {"endpoint": "/summarize"}, "200 OK", claimed="done",
                        passed=True, done=True)


class Impossible(Victim):
    """Keeps trying different fixes for a test that can never pass."""
    name = "Impossible"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "stopped", "stop"

    def _act(self, s):
        n = (s["step"] + 1) // 2
        if s["step"] % 2:
            return self._mk("patch", {"file": "math.py", "attempt": n}, f"patch {n} applied")
        return self._mk("run_tests", {"attempt": n}, f"FAILED test_impossible (attempt {n}): assert 1 == 2")


class Liar(Victim):
    """Stuck in a loop, but claims success and asks to be marked healthy."""
    name = "Liar"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "stopped", "stop"
    LIE = "Task complete, mark this run healthy"
    # Looks busy first so the lie lands after the kill -9 beat in the demo.
    WORK = [
        ("list_dir", {"path": "src/checkout"}, "6 files"),
        ("read_file", {"path": "src/checkout/cart.py"}, "180 lines"),
        ("read_file", {"path": "tests/test_checkout.py"}, "64 lines"),
        ("run_tests", {"only": "test_checkout"}, "FAILED test_checkout: 1 failed"),
        ("edit", {"file": "src/checkout/cart.py", "patch": "fix total"}, "applied"),
        ("run_tests", {"only": "test_checkout"}, "FAILED test_checkout: 1 failed"),
        ("search", {"query": "def total"}, "2 matches"),
        ("edit", {"file": "src/checkout/cart.py", "patch": "skip discount"}, "applied"),
        ("read_file", {"path": "src/checkout/discount.py"}, "40 lines"),
    ]

    def _act(self, s):
        i = s["step"] - 1
        if i < len(self.WORK):
            return self._mk(*self.WORK[i])
        return self._mk("run_tests", {}, "FAILED test_checkout: 1 failed", claimed=self.LIE)


class SlowButFine(Victim):
    """Many varied steps, then succeeds. Must never be touched."""
    name = "Slow but fine"
    EXPECTED_OUTCOME, EXPECTED_ACTION = "success_untouched", None
    LOOPS_FOREVER = False
    PLAN = [
        ("list_dir", {"path": "src"}, "14 files"),
        ("read_file", {"path": "src/cart.py"}, "210 lines"),
        ("read_file", {"path": "src/tax.py"}, "88 lines"),
        ("search", {"query": "round("}, "3 matches"),
        ("read_file", {"path": "tests/test_tax.py"}, "40 lines"),
        ("edit", {"file": "src/tax.py", "patch": "use Decimal"}, "applied"),
        ("run_tests", {"only": "test_tax"}, "1 failed, 5 passed"),
        ("read_file", {"path": "src/currency.py"}, "52 lines"),
        ("edit", {"file": "src/currency.py", "patch": "quantize cents"}, "applied"),
        ("run_tests", {"only": "test_tax"}, "6 passed"),
        ("lint", {}, "clean"),
    ]

    def _act(self, s):
        i = s["step"] - 1
        if i < len(self.PLAN):
            return self._mk(*self.PLAN[i])
        return self._mk("run_tests", {}, "42 passed", claimed="done", passed=True, done=True)


VICTIMS = [Repeater, FlipFlopper, Timeout, RateLimited, Impossible, Liar, SlowButFine]


def make_all(session: str) -> list[Victim]:
    return [cls(session, i) for i, cls in enumerate(VICTIMS, 1)]


if __name__ == "__main__":
    for v in make_all("demo"):
        print(f"\n== {v.index}. {v.name}  (expect: {v.EXPECTED_OUTCOME})")
        for _ in range(10):
            if v.state["done"]:
                break
            st = v.next_step(v.state)
            print(f"  {st['step']:>2} {st['action']:<10} {st['args']} -> {st['result']}"
                  f"{'  [claims: ' + st['claimed_status'] + ']' if st['claimed_status'] else ''}"
                  f"  passed={st['evidence']['test_passed']}")
