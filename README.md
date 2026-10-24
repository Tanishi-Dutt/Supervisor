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

1. `python3 run_demo.py --fresh --stage` (1s per tick, ~25s run). Point at loops caught, timeout backing off, agent 7 untouched.
2. Other pane, around tick 6–8: `./kill.sh`
3. `python3 run_demo.py --stage`: title shows **RESUMED from checkpoint**; the Liar starts lying at step 10, gets flagged, and the **POLICY VETO** line stays pinned in red.
4. Results table stays up until Enter. Then `python3 show_report.py 6` for the Liar's black box.

The agents are scripted simulations (no real tools, no sandbox): every run behaves the same, offline.
