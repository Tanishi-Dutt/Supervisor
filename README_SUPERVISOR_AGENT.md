# SupervisorAgent — an autonomous AI rescue agent

**SupervisorAgent** is our agent. It watches other AI agents ("workers"), detects
when they go wrong, decides how to fix it, makes sure each fix is safe, applies it,
checks whether it worked, and stops a worker when continuing isn't justified. It
survives `kill -9`. Its goal:

> Keep supervised workers productive and safe, recover from fixable failures, and stop
> execution when continued operation is unjustified.

The LLM (Qwen 3.5 9B via Ollama today) is only its **pluggable reasoning engine**. The
goal, control loop, tools, memory, safety policy and kill switches are our code.

## Setup (macOS, once)

Requirements: Python 3.12, [Ollama](https://ollama.com) running locally, about 8 GB free
for the model. No API keys, no paid services.

```bash
cd ~/Downloads/Supervisor-
git switch feature/supervisor-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
ollama pull qwen3.5:9b                 # skip if `ollama list` already shows it
curl -s localhost:11434/api/tags       # Ollama must answer; the list must include qwen3.5:9b
python -m pytest -q                    # expect: 112 passed (no LLM needed)
```

To use a different installed tag, set `OLLAMA_MODEL=<tag>` or pass `--model <tag>`. If
the model is missing, the live demos exit with an error instead of faking output.

## Demos (run from the repo root with the venv active)

| Command | Takes | What you should see |
|---|---|---|
| `python agent_demo.py live` | ~45 s | LIVE Qwen worker reads, edits, hits `[INJECTED FAULT]` 429s → DETECT → REASON (LIVE Qwen) → POLICY APPROVED `retry_with_backoff` → worker pauses 12 s → tests pass → **COMPLETED (recovered)**, supervisor's own pytest: 3 passed |
| `python agent_demo.py kill` | ~25 s | step budget (deliberately 2) runs out → POLICY VETO → STOP_AGENT report `verified: True` → STOP_LOCAL_MODEL **refused** (shared with `supervisor:reasoning`) → then **unloaded** |
| `python crash_recovery_check.py` | ~60 s | SIGKILLs the supervisor after 3 worker steps, relaunches with `--resume`, prints 8 PASS lines |
| `python agent_demo.py scripted` | ~40 s | SIMULATED 7 victims, LIVE Qwen reasoning → `7/7 expected outcomes` |
| `python agent_demo.py scripted --offline` | ~1 s | same with the rule-based fallback, no LLM |
| `python run_demo.py --fresh` | ~10 s | the original team demo, unchanged |

Run `kill` last on stage: it unloads the model, so the next live run spends ~2 s reloading it.

Manual crash demo instead of the script: start `python agent_demo.py live`, note the
"supervisor PID" in the banner, run `kill -9 <that PID>` from a second terminal after a
few steps, then run `python agent_demo.py live --resume`. Look for the yellow
**RESTORED FROM SQLITE CHECKPOINT** line. The worker continues from its next step.

Where things go: run directories (sandbox, worker events, logs) are in
`$TMPDIR/flight_recorder_runs/<session>/`; SQLite is `data/flight_recorder.sqlite3`
(override with `FLIGHT_RECORDER_DB`); incident reports are in `reports/`. All are
git-ignored or outside the repo.

Troubleshooting: "Model … is not available" means Ollama isn't running or the tag isn't
pulled. If a demo is interrupted with Ctrl-C, its worker exits by itself within a few
seconds, because workers exit when their supervisor is gone.

## Architecture

```
                 ┌────────────────────────── SupervisorAgent (supervisor_agent.py) ───────────────────────────┐
                 │ OBSERVE → DETECT → REASON → PLAN → VALIDATE → ACT → VERIFY → REMEMBER → REPEAT              │
                 │   │         │         │               │          │       │          │                      │
 worker events ──┼─► adapter  watchdog  ChatModel     SupervisorPolicy tools  pytest   SupervisorMemory         │
 (Step format)   │  workers.py watchdog.py llm_provider  + policy.py  (retry, hint, (own run) store.py SQLite      │
                 │            + own detectors (any LLM)  (Dev C veto) rollback,              + report.py         │
                 │                                                    stop_agent)                              │
                 └──────────────┬──────────────────────────────────────────────┬────────────────────────────────┘
                                │ control.jsonl (hold/backoff/hint/cancel)      │ kill_controls.py
                                ▼                                               ▼
              worker_process.py: real Qwen coding agent             STOP_AGENT (owned PID only)
              tools via sandbox_tools.ToolGateway                   STOP_LOCAL_MODEL (lease-gated unload)
              (read_file / edit_file / run_tests / finish)
```

| File | What it is |
|---|---|
| `supervisor_agent.py` | **SupervisorAgent**: control loop, detectors, reasoning prompt, `SupervisorPolicy`, recovery tools, `SupervisorMemory` |
| `workers.py` | `WorkerAdapter` protocol; `OllamaProcessWorker` (LIVE subprocess); `ScriptedVictimWorker` (SIMULATED) |
| `worker_process.py` | The real LLM worker: Qwen picks each tool call; checkpoints itself; exits if the supervisor dies |
| `sandbox_tools.py` | The coding task, the allowlisted tools, the INJECTED fault, the independent verifier |
| `kill_controls.py` | `stop_agent`, `stop_local_model`, process ownership checks, model leases |
| `llm_provider.py` | `ChatModel` protocol, `OllamaChatModel`, `ScriptedModel` (tests only) |
| `agent_demo.py` | Rich terminal demo |
| `crash_recovery_check.py` | Real kill -9 / resume experiment with pass/fail checks |

Reused unchanged from the team: `watchdog.py` (loop/flip-flop/budget/deception detection),
`store.py` (SQLite checkpoints), `policy.py` (Dev C hard veto), `report.py` (incident
reports), `victims.py` (scripted scenarios), `interfaces.py` (Step format).

## What makes it autonomous

After launch nobody prompts it. Each tick, for each worker, it:

1. **Observes** new worker events (tool calls, results, claims) through the adapter.
2. **Detects** problems: the team watchdog (loops, A-B-A-B flip-flops, step/time/retry
   budgets, false success claims) plus its own detectors (repeated tool failures,
   invalid model output, stalls, token budget).
3. **Pauses** the worker (`hold`) so it can't keep acting while the supervisor thinks.
4. **Reasons**: sends the flag, recent steps, earlier interventions and remaining
   budgets to its model, which returns a diagnosis, one tool, and a hint for the worker.
5. **Validates** with `SupervisorPolicy`, in plain deterministic code: tool whitelist;
   Dev C's hard veto (claimed success + failing tests ⇒ stop); budget and deception
   flags ⇒ stop; at most 4 interventions; retry cap 2; a fix that failed twice is not
   tried again; no rollback without a snapshot.
6. **Acts**: `retry_with_backoff` (pauses the worker), `change_strategy` (sends the hint),
   `rollback` (restores the original files), `stop_agent` (kill switch).
7. **Verifies** the fix using only steps the worker took *after* acknowledging it, and
   checks every completion claim by running the tests itself. The worker's word is
   never enough.
8. **Remembers**: checkpoints worker records, decisions, verdicts and evidence to SQLite.
9. **Decides when to stop**: budget, deception, an exhausted intervention budget, or
   fixes that keep failing all end in `stop_agent`. When a stopped worker's local model
   is no longer needed by anything of ours, it unloads it.

## What we built vs. what the LLM provides

| Ours (deterministic code) | The LLM's part |
|---|---|
| Goal, loop, scheduling, state machine (`running → recovering → completed/stopped`) | Diagnosis label + root cause |
| Detection (watchdog + supervisor detectors) | Choice of **one** of 4 tools (a proposal only) |
| Policy: every action authorised or vetoed | Hint text sent to the worker |
| Tool execution, process control, kill switches | In the worker: which tool to call next |
| Independent verification (supervisor-run pytest) | |
| SQLite memory, crash recovery, reports | |

If the model is down, returns malformed JSON, or proposes something unsafe, the
supervisor falls back to a rule table (shown as **RULE-BASED FALLBACK**, `live=False`)
or the policy vetoes it. A fallback is never presented as model inference.

## Model adapters (model independence)

Supervisor side: anything that implements `ChatModel` (`name`, `provider`, `local`,
`complete_json(system, messages) -> ModelReply`). `OllamaChatModel` is implemented; a
hosted provider is one more class of the same shape.

Worker side: anything that implements `WorkerAdapter` (`start`, `observe`,
`get_history`, `send`, `rollback`, `revoke`, `request_cancel`, `is_running`,
`get_result`, `verify`, `snapshot/restore`) and reports steps in the team's `Step`
format. The supervisor never looks at which model a worker uses. Implemented: a live
Ollama subprocess worker and the scripted victims.

## The two kill controls and their safety limits

**STOP_AGENT** (`kill_controls.stop_agent`), works for any worker:
1. revoke: writes `REVOKED`; the tool gateway refuses every later tool call;
2. cancel: asks the worker loop to stop;
3. waits, then SIGTERM, then SIGKILL, each only if still needed;
4. verifies the process is gone and records method, PID, exit code and reason in SQLite.

Limits: it only signals processes **we spawned**. Each worker is started with a unique
`--owner-token`, and before any signal we check that PID's command line still carries
that token. It refuses our own PID, our parent and PID 1. No `pkill`, no name matching.
In-process or hosted-model workers are stopped by cancelling our loop and requests.

**STOP_LOCAL_MODEL** (`kill_controls.stop_local_model`):
- unloads one named model through Ollama's documented `keep_alive: 0` request, then
  polls `/api/ps` to verify;
- refuses while any other holder leases the model, including the supervisor's own
  reasoning (`supervisor:reasoning`);
- never deletes model files and never stops the Ollama server or other apps;
- for hosted providers it refuses: we can stop our worker and requests, not the
  provider's servers.

Limit: leases only know about *our* components. If another app on the machine uses the
same Ollama model, it is unloaded too (Ollama reloads it on that app's next request).

## LIVE vs SIMULATED

| Thing | Status |
|---|---|
| `agent_demo.py live/kill` worker | **LIVE**: real Qwen calls choose every tool; real files; real pytest |
| Supervisor reasoning in `live`, `kill`, `scripted` | **LIVE** Qwen (each decision shows its source and latency) |
| The 429 rate limiter on `run_tests` | **INJECTED**, labelled `[INJECTED FAULT #n]` in the worker's output |
| The 7 victims (`scripted`, `run_demo.py`) | **SIMULATED** workers |
| `pytest` suite | Uses `ScriptedModel` / fakes; no LLM |

## Crash recovery

- Worker progress (conversation, step, processed commands, fault state) is checkpointed
  by the worker; supervisor state (records, decisions, event offset, command sequence)
  by the supervisor, both through `store.py`.
- If the supervisor dies, its worker notices the parent is gone and exits.
- On `--resume`, the supervisor restores its records and starts a new worker process,
  which loads its own checkpoint and continues from the next step.
- Commands are numbered, so a command re-sent after a restart is applied once. Steps
  are deduplicated by step number. Reports are written once (persisted `reported` flag).
- **Not exactly-once:** the tool call in flight at the moment of a crash can run again.
  The worker's tools are idempotent (read a file, write a whole file, run tests), so
  replay is harmless here; tools with external side effects would need idempotency keys.

## Known limitations

- The sandbox is a guardrail, not OS isolation. Code the model writes must pass an AST
  check (allowlisted imports; no `exec/eval/open/__import__`; no dunder access) and runs
  under pytest with a timeout and a minimal environment, but as the current user. Run
  directories live in `$TMPDIR/flight_recorder_runs/`, outside the repository.
- One live worker at a time in the demo (the loop supports several).
- Qwen's diagnoses are sometimes imprecise. The policy keeps them safe, but not always right.
- The time budget counts wall-clock time since the run first started, including downtime.
- Token budget counts worker `eval_count` tokens when Ollama reports them.
