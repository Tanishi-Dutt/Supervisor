# Agent Flight Recorder

A supervisor that watches AI agents, catches loops, budget overruns and deception,
applies a policy-approved recovery, survives `kill -9`, and writes a black-box report per run.
Fully offline on a local LLM (Qwen 3.5 via Ollama).

## Run

```
pip install -r requirements.txt
python3 run_demo.py --fresh        # new run, live table
python3 run_demo.py                # resume after kill -9
python3 run_demo.py --fake         # use fakes for every Dev B / Dev C module
python3 victims.py                 # print each victim's first 10 steps
```

Any missing module (`watchdog`, `store`, `diagnose`, `policy`, `report`) falls back to
its stand-in in `fakes.py`; the runner prints which ones are faked.

## Modules

| Owner | Files | Function signatures (see `interfaces.py`) |
|---|---|---|
| Dev A | `victims.py`, `run_demo.py` | `agent.next_step(state) -> Step`, `agent.apply(action)` |
| Dev B | `watchdog.py`, `store.py` | `check(history) -> Flag \| None`, `save(run_id, step, state)`, `load_last(run_id) -> state` |
| Dev C | `diagnose.py`, `policy.py`, `report.py` | `diagnose(last_steps, flag) -> Proposal`, `approve(proposal, evidence, step=None) -> action`, `write_report(run_id) -> path` |

Things to know when integrating:
- `check()` receives only the steps **since the last intervention**, so a handled loop isn't re-flagged.
- `evidence` carries `retries_used` alongside `test_passed`; use it for the retry budget.
- `state` passed to `save()` is the whole run record (agent state, history, events);
  `report.py` should rebuild the run from `load_last(run_id)` alone.
- `diagnose()` must return one of `ACTIONS`; `approve()` must turn anything else into `stop`.
- LLM: `OLLAMA_MODEL` (default `qwen3.5`). Use `"format": "json"` and `"think": false`.

Done check: `python3 run_demo.py --fresh --no-live --speed 0` ends with
`7/7 expected outcomes · false stops: 0`.

## Demo (on stage)

Two panes, font zoomed. Warm up Ollama first (one dummy request) so the first diagnosis isn't slow.

1. `python3 run_demo.py --fresh --stage --viewer` (1s per tick, ~25s run; opens the city view at http://localhost:8765/viewer/). Point at loops caught, timeout backing off, agent 7 untouched.
2. Other pane, around tick 6–8: `./kill.sh`
3. `python3 run_demo.py --stage`: title shows **RESUMED from checkpoint**; the Liar starts lying at step 10, gets flagged, and the **POLICY VETO** line stays pinned in red.
4. Results table stays up until Enter. Then `python3 show_report.py 6` for the Liar's black box.

The agents are scripted simulations (no real tools, no sandbox): every run behaves the same, offline.

## City view

`--viewer` starts a small static server (separate process, so it survives `kill -9`) and opens
`viewer/index.html`. The page polls `viewer/live.json` (written every tick) and
`viewer/heartbeat.json` (every 0.5s). It only reads: nothing in the browser can affect a run.

- Worker bots walk to the building for their current tool: Library (search), Archive (read_file,
  list_dir), Workshop (edit, patch), Test Lab (run_tests, lint), Network (fetch, call_api).
- The dark supervisor drone flies to each flagged bot: LLM proposal, then the policy verdict
  (red VETOED stamp on the Liar).
- Heartbeat older than 2s means **SUPERVISOR KILLED**: bots freeze and turn grey. On restart the page shows
  **RESUMED from checkpoint**.
- Results pop up when the run ends; press R to toggle.

## Real agent (`--real`)

`python3 run_demo.py --fresh --real` adds agent #8, **Real coder**: it edits real files in a sandbox
(`$TMPDIR/flight_recorder_sandbox/<run_id>/`) to fix a genuinely failing `pytest`, and its evidence
is a real pytest run recomputed by the harness every step, never taken from the agent's word.

- `--real` (auto): Qwen drives it when a Qwen model is installed in Ollama (any tag, e.g. `qwen3.5:latest`).
  In testing it fixed the bug in 5 steps with no intervention, which shows the supervisor doesn't stop healthy real agents.
- `--real scripted`: same real tools, but a brain that re-runs the tests until the supervisor says
  `change_strategy`, then reads both files, fixes the bug and passes, which reliably shows supervision of a real-tools agent.

Tip: set `OLLAMA_MODEL` to the tag you actually have (`ollama list`) so diagnosis uses the LLM too.

## Use it on your own agent

```python
from flight_recorder import FlightRecorder

with FlightRecorder("checkout-bot-42") as fr:
    while not agent.finished:
        action, args = agent.decide()
        result = agent.run(action, args)
        verdict = fr.step(action, args, result, claimed=agent.status, test_passed=run_my_tests())
        if verdict == "stop":
            break
        if verdict != "continue":
            agent.recover(verdict)  # retry_with_backoff | change_strategy | rollback
```

Same watchdog, LLM diagnosis, policy veto, SQLite checkpoints and incident report as the demo;
reopening the same `run_id` after a crash resumes it. See `tests/test_flight_recorder.py`.
