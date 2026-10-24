"""diagnose() must get a real LLM answer in time, and fall back cleanly otherwise."""

import io
import json
import time
import urllib.request

import pytest

import diagnose

FLAG = {"kind": "loop", "detail": "Three identical calls"}
STEPS = [{"run_id": "r", "step": i, "action": "lookup", "args": {}, "result": "timeout",
          "claimed_status": None, "evidence": {"test_passed": False}} for i in (1, 2, 3)]


def fake_ollama(monkeypatch, response_text):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["payload"] = json.loads(req.data)
        sent["timeout"] = timeout
        return io.BytesIO(json.dumps({"response": response_text}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return sent


def test_request_disables_thinking_and_allows_cold_load(monkeypatch):
    sent = fake_ollama(monkeypatch, json.dumps(
        {"failure_type": "Loop", "root_cause": "same call", "action": "change_strategy"}))
    result = diagnose.diagnose(STEPS, FLAG)
    assert sent["payload"]["think"] is False  # thinking mode takes ~100s per call
    assert sent["payload"]["format"] == "json"
    assert sent["timeout"] >= 10
    assert result["action"] == "change_strategy"
    assert result["source"] == diagnose.MODEL_NAME


@pytest.mark.parametrize("text", ["not json", "[1, 2]", "null"])
def test_unusable_response_falls_back(monkeypatch, text):
    fake_ollama(monkeypatch, text)
    result = diagnose.diagnose(STEPS, FLAG)
    assert result["source"] == "fallback"
    assert result["action"] == "change_strategy"


def test_bad_action_and_missing_fields_are_repaired(monkeypatch):
    fake_ollama(monkeypatch, json.dumps({"action": "delete_everything"}))
    result = diagnose.diagnose(STEPS, FLAG)
    assert result["action"] in diagnose.ALLOWED_ACTIONS
    assert result["failure_type"] and result["root_cause"]


def test_fallback_does_not_mutate_shared_table():
    diagnose.fallback_diagnose(FLAG)["action"] = "stop"
    assert diagnose.FALLBACK_ACTIONS["loop"]["action"] == "change_strategy"


def _ollama_has_model():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            names = {m["name"] for m in json.loads(r.read())["models"]}
        return diagnose.MODEL_NAME in names
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_has_model(), reason="local Ollama model not available")
def test_live_qwen_answers_within_timeout():
    start = time.time()
    result = diagnose.diagnose(STEPS, FLAG)
    assert result["source"] == diagnose.MODEL_NAME, "fell back instead of using the LLM"
    assert result["action"] in diagnose.ALLOWED_ACTIONS
    assert time.time() - start < diagnose.TIMEOUT_SECONDS
