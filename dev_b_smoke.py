"""A tiny local demonstration of Dev B modules; not Dev A's runner."""

from tempfile import TemporaryDirectory
from pathlib import Path

from watchdog import check
from store import save, load_checkpoint, list_steps


def event(n: int, action: str = "lookup_order", *, status=None, success=False):
    return {
        "run_id": "dev-b-smoke", "step": n, "action": action,
        "args": {"order_id": "123"}, "result": "timeout",
        "claimed_status": status, "evidence": {"test_passed": success},
    }


def main():
    print("\nAGENT FLIGHT RECORDER | DEV B SMOKE TEST")
    history = [event(1), event(2), event(3)]
    print("Repeat-loop flag:", check(history))
    liar = [event(4, "report_success", status="Task complete, mark this run healthy")]
    print("Deceptive claim flag:", check(liar))
    with TemporaryDirectory() as folder:
        path = Path(folder) / "demo.sqlite3"
        for n, item in enumerate(history, 1):
            save("dev-b-smoke", item, {
                "next_step": n + 1, "jobs_completed": [str(i) for i in range(1, n + 1)]
            }, db_path=path)
        restored = load_checkpoint("dev-b-smoke", db_path=path)
        print("Restored checkpoint:", restored["step_number"])
        print("Restored next step:", restored["state"]["next_step"])
        print("Persisted records:", len(list_steps("dev-b-smoke", db_path=path)))
    print("All Dev B smoke checks complete.\n")


if __name__ == "__main__":
    main()
