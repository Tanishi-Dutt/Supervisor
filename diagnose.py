import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:9b"
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
    return FALLBACK_ACTIONS.get(kind, {
        "failure_type": "Unknown Failure",
        "root_cause": flag.get("detail", "Unspecified anomaly"),
        "action": "stop"
    })

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
1. "retry_with_backoff"
2. "change_strategy"
3. "rollback"
4. "stop"

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
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            response_text = res_data.get("response", "{}")
            result = json.loads(response_text)

            action = result.get("action")
            if action not in ALLOWED_ACTIONS:
                result["action"] = fallback_diagnose(flag)["action"]

            return result

    except Exception:
        # Graceful fallback on network timeout or JSON parse errors
        return fallback_diagnose(flag)