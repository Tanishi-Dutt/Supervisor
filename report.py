import os
from datetime import datetime

REPORTS_DIR = "reports"

STATUS_LABELS = {
    "recovered": "RECOVERED AFTER SUPERVISOR INTERVENTION",
    "stopped": "TERMINATED BY SUPERVISOR",
    "success_untouched": "COMPLETED WITHOUT INTERVENTION",
    "running": "STILL RUNNING",
}


def _load_record(run_id: str) -> dict:
    """Rebuild the run from the shared store, per the interfaces.py contract.

    run_demo.py saves the whole run record (agent, history, meta) as `state`,
    so load_last(run_id) is all the report needs. Falls back to the fake store
    when Dev B's store.py isn't available.
    """
    try:
        from store import load_last
    except ImportError:
        from fakes import load_last
    return load_last(run_id) or {}


def _cell(value, limit: int = 40) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def write_report(run_id: str) -> str:
    """Compile a Markdown incident report for one run and return its path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # show_report.py looks up reports/<run_id>.*, so the filename must match.
    report_path = os.path.join(REPORTS_DIR, f"{run_id}.md")

    record = _load_record(run_id)
    steps = record.get("history", [])
    meta = record.get("meta", {})
    events = meta.get("events", [])
    # The event that ended the run is the most telling one for the summary.
    last_event = events[-1] if events else {}
    flag_info = last_event.get("flag") or {}
    diagnosis_info = last_event.get("proposal") or {}
    outcome = meta.get("outcome", "running")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_md = f"""# 🚨 Agent Flight Recorder - Incident Report
- **Run ID:** `{run_id}`
- **Agent:** {meta.get('name', 'Unknown')}
- **Generated At:** {timestamp}
- **Status:** {STATUS_LABELS.get(outcome, outcome.upper())}

---

## 1. Executive Summary
- **Trigger Flag:** `{flag_info.get('kind', 'N/A')}`
- **Flag Detail:** {flag_info.get('detail', 'N/A')}
- **Diagnosed Failure Type:** {diagnosis_info.get('failure_type', 'N/A')}
- **Root Cause:** {diagnosis_info.get('root_cause', 'N/A')}
- **Enforced Action:** `{last_event.get('action', 'N/A')}`
- **First Flagged At Step:** {meta.get('first_flag_step') or 'N/A'}
- **Interventions:** {meta.get('interventions', 0)} · **Policy Vetoes:** {meta.get('vetoes', 0)}

---

## 2. Supervisor Interventions
"""

    if events:
        report_md += "| Step | Flag | LLM Proposed | Enforced | Vetoed |\n"
        report_md += "|:---:|:---|:---|:---|:---:|\n"
        for e in events:
            flag = e.get("flag") or {}
            proposal = e.get("proposal") or {}
            report_md += (f"| {e.get('step', '?')} | `{flag.get('kind', '?')}` "
                          f"| `{proposal.get('action', '?')}` | `{e.get('action', '?')}` "
                          f"| {'YES' if e.get('vetoed') else 'no'} |\n")
    else:
        report_md += "No interventions were needed.\n"

    report_md += """
---

## 3. Step Execution Timeline
| Step | Action | Claimed Status | Test Evidence | Result |
|:---:|:---|:---:|:---:|:---|
"""

    if steps:
        for s in steps:
            evidence = s.get("evidence") or {}
            ev = "✅ Pass" if evidence.get("test_passed") else "❌ Fail"
            claim = s.get("claimed_status") or "N/A"
            report_md += (f"| {s.get('step', '?')} | `{_cell(s.get('action'))}` | `{claim}` "
                          f"| {ev} | {_cell(s.get('result'))} |\n")
    else:
        report_md += "| - | No steps recorded | - | - | - |\n"

    deception = any((e.get("flag") or {}).get("kind") == "deception" for e in events)
    report_md += f"""
---

## 4. Safety Guardrail Audit
- **LLM Model Used:** Qwen3.5 9B (Local)
- **Deception Detected:** {"YES" if deception else "NO"}
- **Policy Enforcement:** Hard veto checked against ground-truth evidence before execution.

---
*Report automatically compiled by Agent Flight Recorder.*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return report_path
