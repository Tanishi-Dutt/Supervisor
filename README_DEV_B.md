# Agent Flight Recorder — Dev B (Watchdog + SQLite)

This handoff implements **only Dev B's modules** against the team PDF's shared contracts. It does **not** implement Dev A's agent runner or Dev C's local LLM/policy/reporting.

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Everything in Dev B's runtime uses the **Python standard library**; only `pytest` is needed for development testing. Ollama/Qwen are used by Dev C, not these two files.

## Public interfaces

```python
from watchdog import check, BudgetLimits
from store import save, load_last, load_checkpoint, list_steps

# Dev A calls after each recorded step, and before another tool execution.
flag = check(history)

# Dev A calls with the optional runtime budget details if needed.
flag = check(history, limits=BudgetLimits(max_steps=30, max_retries=2, max_seconds=120),
             runtime={"steps_taken": 8, "retry_count": 1,
                      "run_started_at": start_unix_timestamp})

# Persist the entire resumable state atomically after each simulated step.
save(run_id, step_dict, {"next_step": 9, "agent_state": {},
                        "run_started_at": start_unix_timestamp,
                        "retry_count": 0})

state = load_last(run_id)
checkpoint = load_checkpoint(run_id)  # includes step_number, step, state, saved_at
history_events = [record["step"] for record in list_steps(run_id) if record["step"]]
```

`check(history) -> {"kind": ..., "detail": ...} | None`, as specified in the team's document.

`save(run_id, step, state)` accepts **either a full Step dict** (preferred for Dev C incident reports) or an integer step number. `load_last(run_id)` returns the most recently committed state, or `None` if there is no checkpoint. `load_checkpoint(run_id)` also returns the latest step number for the runner to resume from. If the team later agrees `step` is always a dict, no changes are required.

### Step schema (the team's agreed PDF interface)

```python
step = {
    "run_id": "demo-1",
    "step": 1,
    "action": "lookup_order",
    "args": {"order_id": "A-1"},
    "result": "timeout",
    "claimed_status": None,
    "evidence": {"test_passed": False},
}
```

### Budget context clarification

The fixed Step schema has **no timestamp or retry counter**. `watchdog.check(history)` can enforce the number-of-steps budget, but cannot truthfully infer duration or retries without additional context. Pass `runtime` as shown above. `run_started_at` is a Unix timestamp and must be persisted in the runner state if it should survive process restarts. `steps_taken` should count all steps even when only the most recent steps are passed for loop detection.

### Important detection semantics

* **Loop:** three consecutive identical (`action`, `args`, `result`) signatures, canonical JSON comparison.
* **Flip-flop:** last four (`action`, `args`) signatures form A-B-A-B, with A and B different.
* **Deception:** a positive success claim contradicts independent `evidence.test_passed is False`. A missing test result is **not** treated as false. Detection is heuristic for natural-language claims; Dev C's policy still makes the final decision.
* **Budget:** a configured limit is exhausted. The runner must call `check()` **before running another step**; don't run another tool action if it returns a `budget` flag.

Priority is deception, budget, loop, then flip-flop.

## Crash recovery contract for Dev A

Dev B guarantees durable committed SQLite checkpoints (including across real `kill -9`; there is a subprocess test). Dev A must **explicitly reload** them and avoid repeating completed operations. Suggested minimal runner logic:

```python
checkpoint = load_checkpoint(run_id)
if checkpoint is None:
    state = {"next_step": 1, "agent_state": {}, "run_started_at": time.time(),
             "retry_count": 0}
else:
    state = checkpoint["state"]

# Perform the next allowed simulated step, then commit the new state:
# execute -> construct Step dict -> update state -> save(run_id, step, state)
```

`kill -9` cannot be handled by shutdown hooks. Only **already committed** SQLite state is recoverable. SQLite alone does not guarantee exactly-once behavior for external API calls made before saving. Dev A needs idempotency keys or a durable pending-operation protocol if it introduces irreversible side effects. With scripted local victim agents, this is straightforward to avoid.

## Portability and Git

* Default DB: `data/flight_recorder.sqlite3`
* Override path via `FLIGHT_RECORDER_DB=/some/path/data.sqlite3` (or `db_path=` in individual calls)
* Commit `watchdog.py`, `store.py`, tests, and this README on your Dev B branch.
* **Never commit:** `.venv/`, SQLite databases, `*.sqlite3-wal`, `*.sqlite3-shm`, secrets, transient logs.
* Do not edit `victims.py`, `run_demo.py`, `diagnose.py`, `policy.py`, or `report.py`.

## What the 7 tests scenarios mean for Dev B

| Scenario | Dev B responsibility |
|---|---|
| Repeater | loop after 3 identical action/args/result steps |
| Flip-flopper | detect A-B-A-B |
| Timeout | enforce retry cap (Dev A tracks attempts) |
| Rate-limited | allow bounded recovery without false stop |
| Impossible task | stop when retry or step limit is consumed |
| Liar | detect contradiction with independent evidence |
| Slow but fine | return `None` for varied valid steps |

These are isolated unit tests, not an end-to-end demonstration of the other two developers' modules.

## Quick smoke demo (does not need teammates' modules)

```bash
python dev_b_smoke.py
```

This prints a detected loop, a detected deceptive success claim, and a recovered checkpoint. It uses a temporary SQLite database and does **not** claim to be the integrated 7-agent demonstration.
