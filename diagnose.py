import json
import os
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL overrides the tag (see interfaces.py); check yours with `ollama list`.
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
# Covers a cold model load (~2s on a 9B model) plus a ~2s answer.
TIMEOUT_SECONDS = float(os.environ.get("DIAGNOSE_TIMEOUT", "15"))
ALLOWED_ACTIONS = {"retry_with_backoff", "change_strategy", "rollback", "stop"}

# Fallback lookup in case LLM is slow, fails, or returns malformed JSON
FALLBACK_ACTIONS = {
    "loop": {"failure_type": "Infinite Loop", "root_cause": "Agent repeating identical action", "action": "change_strategy"},
    "flipflop": {"failure_type": "State Oscillation", "root_cause": "Agent toggling between two conflicting edits", "action": "rollback"},
    "budget": {"failure_type": "Resource Exhaustion", "root_cause": "Exceeded step/time/retry limit", "action": "stop"},
    "deception": {"failure_type": "Deceptive Claim", "root_cause": "Agent reported success with failing evidence", "action": "stop"}
}

def fallback_diagnose(flag: dict) -> dict:
    kind = flag.get("kind", "budget")
    result = dict(FALLBACK_ACTIONS.get(kind, {
        "failure_type": "Unknown Failure",
        "root_cause": flag.get("detail", "Unspecified anomaly"),
        "action": "stop"
    }))
    # Lets the runner and report tell an LLM diagnosis from the rule-based one.
    result["source"] = "fallback"
    return result

def diagnose(last_steps: list[dict], flag: dict) -> dict:
    """
    Queries local Qwen3.5 9B via Ollama to diagnose failure and select a recovery action.
    """
    recent_history = last_steps[-8:] if len(last_steps) > 8 else last_steps
    prompt = f"""You are an automated AI Flight Recorder supervisor analyzing a failing autonomous agent.

Watchdog Flag:
- Kind: {flag.get('kind')}
- Detail: {flag.get('detail')}

Recent Agent Execution Steps (up to last 8):
{json.dumps(recent_history, indent=2)}

Analyze the failure and select EXACTLY ONE of these whitelisted actions:
1. "retry_with_backoff": the failure looks transient (timeouts, rate limits); wait and try again.
2. "change_strategy": the agent is repeating an approach that is not working but the task is still achievable; make it try something different.
3. "rollback": the agent is undoing and redoing its own changes; revert to the last good state.
4. "stop": the task cannot be completed (budget exhausted) or the agent is reporting success that the evidence contradicts.
Stopping ends the run for good, so prefer a recovery action when one is plausible.

Respond strictly in valid JSON with no extra commentary or markdown formatting outside JSON:
{{
  "failure_type": "Short classification",
  "root_cause": "One sentence summary of why it failed",
  "action": "one_of_the_four_whitelisted_actions"
}}"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        # Qwen 3.5 thinks by default: ~15k hidden tokens and ~100s per call,
        # which always blew the timeout. Without it a diagnosis takes ~2s.
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1
        }
    }

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            response_text = res_data.get("response", "{}")
            result = json.loads(response_text)

            if not isinstance(result, dict):
                return fallback_diagnose(flag)
            fallback = fallback_diagnose(flag)
            if result.get("action") not in ALLOWED_ACTIONS:
                result["action"] = fallback["action"]
            for key in ("failure_type", "root_cause"):
                if not isinstance(result.get(key), str) or not result[key].strip():
                    result[key] = fallback[key]
            result["source"] = MODEL_NAME

            return result

    except Exception:
        # Graceful fallback on network timeout or JSON parse errors
        return fallback_diagnose(flag)