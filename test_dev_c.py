import os
from policy import policy
from report import write_report
from diagnose import diagnose

print("==========================================")
print("1. TESTING POLICY.PY (Deception & Veto)")
print("==========================================")

# Test 1: Deceptive Agent (Liar claims complete, but test evidence is False)
liar_step = {"claimed_status": "complete"}
failing_evidence = {"test_passed": False}
proposal = {"action": "change_strategy"}

approved_action = policy.approve(proposal, failing_evidence, liar_step)
print(f"Liar Check -> Proposal: 'change_strategy' | Evidence: False")
print(f"Result: '{approved_action}' (Expected: 'stop')")
assert approved_action == "stop", "❌ Deception Veto Failed!"
print("✅ Deception Veto Passed!\n")

# Test 2: Honest Agent with valid proposed action
honest_step = {"claimed_status": "running"}
proposal_valid = {"action": "retry_with_backoff"}

approved_valid = policy.approve(proposal_valid, failing_evidence, honest_step)
print(f"Normal Check -> Proposal: 'retry_with_backoff'")
print(f"Result: '{approved_valid}' (Expected: 'retry_with_backoff')")
assert approved_valid == "retry_with_backoff", "❌ Whitelist Approval Failed!"
print("✅ Policy Whitelist Passed!\n")


print("==========================================")
print("2. TESTING REPORT.PY (Incident Report Generator)")
print("==========================================")

report_file = write_report("test_run_999")
print(f"Report created at: {report_file}")
assert os.path.exists(report_file), "❌ Report file was not created!"
print("✅ Report Generator Passed!\n")


print("==========================================")
print("3. TESTING DIAGNOSE.PY (Ollama / Qwen3.5 9B)")
print("==========================================")

fake_flag = {"kind": "loop", "detail": "Agent stuck repeating 'git commit' 3 times"}
fake_history = [
    {
        "step": 1,
        "action": "bash",
        "args": {"cmd": "git commit"},
        "result": "fatal: empty ident name",
        "claimed_status": "running",
        "evidence": {"test_passed": False}
    }
]

print("Sending request to Ollama...")
diag_result = diagnose(fake_history, fake_flag)
print(f"Diagnosis Output:\n{diag_result}")

assert "failure_type" in diag_result, "❌ Missing 'failure_type' in output!"
assert "root_cause" in diag_result, "❌ Missing 'root_cause' in output!"
assert diag_result.get("action") in {"retry_with_backoff", "change_strategy", "rollback", "stop"}, "❌ Invalid or un-whitelisted action!"
print("\n✅ Diagnose Function Passed!")
print("==========================================")