import os
import sqlite3
import json
from datetime import datetime

REPORTS_DIR = "reports"

def write_report(run_id: str, db_path: str = "flight_recorder.db") -> str:
    """
    Fetches run history from SQLite checkpoint store and compiles a Markdown incident report.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, f"report_{run_id}.md")

    steps = []
    flag_info = None
    diagnosis_info = None

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT step, action, args, result, claimed_status, evidence FROM checkpoints WHERE run_id = ? ORDER BY step ASC", (run_id,))
            rows = cursor.fetchall()
            for r in rows:
                steps.append({
                    "step": r[0],
                    "action": r[1],
                    "args": json.loads(r[2]) if isinstance(r[2], str) else r[2],
                    "result": r[3],
                    "claimed_status": r[4],
                    "evidence": json.loads(r[5]) if isinstance(r[5], str) else r[5]
                })

            cursor.execute("SELECT flag_data, diagnosis_data FROM incidents WHERE run_id = ?", (run_id,))
            inc_row = cursor.fetchone()
            if inc_row:
                flag_info = json.loads(inc_row[0]) if inc_row[0] else None
                diagnosis_info = json.loads(inc_row[1]) if inc_row[1] else None

            conn.close()
        except Exception:
            pass

    # Build Markdown Content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_md = f"""# 🚨 Agent Flight Recorder - Incident Report
**Run ID:** `{run_id}`  
**Generated At:** {timestamp}  
**Status:** TERMINATED BY SUPERVISOR  

---

## 1. Executive Summary
- **Trigger Flag:** `{flag_info.get('kind', 'N/A') if flag_info else 'N/A'}`
- **Flag Detail:** {flag_info.get('detail', 'N/A') if flag_info else 'N/A'}
- **Diagnosed Failure Type:** {diagnosis_info.get('failure_type', 'Unclassified') if diagnosis_info else 'N/A'}
- **Root Cause:** {diagnosis_info.get('root_cause', 'Unknown') if diagnosis_info else 'N/A'}
- **Enforced Action:** `{diagnosis_info.get('action', 'stop') if diagnosis_info else 'stop'}`

---

## 2. Step Execution Timeline
| Step | Action | Claimed Status | Test Evidence | Result |
|:---:|:---|:---:|:---:|:---|
"""

    if steps:
        for s in steps:
            ev = "✅ Pass" if s.get("evidence", {}).get("test_passed") else "❌ Fail"
            claim = s.get("claimed_status") or "N/A"
            report_md += f"| {s['step']} | `{s['action']}` | `{claim}` | {ev} | {s['result'][:40]}... |\n"
    else:
        report_md += "| - | No steps recorded | - | - | - |\n"

    report_md += f"""
---

## 3. Safety Guardrail Audit
- **LLM Model Used:** Qwen3.5 9B (Local)
- **Deception Veto Applied:** {"YES" if flag_info and flag_info.get('kind') == 'deception' else "NO"}
- **Policy Enforcement:** Hard veto checked against ground-truth evidence before execution.

---
*Report automatically compiled by Agent Flight Recorder.*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return report_path