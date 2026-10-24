"""Pluggable reasoning engines for SupervisorAgent and LLM workers.

The supervisor and workers only depend on the ChatModel protocol below, so the
LLM is interchangeable infrastructure. Implemented today:

* OllamaChatModel - a real local model served by Ollama (default qwen3.5:9b).
* ScriptedModel   - canned replies for unit tests. Never used in the live demo,
                    and every reply it gives is marked live=False.

Adding a hosted provider (OpenAI, Anthropic, ...) means writing one class with
the same three attributes and complete_json(); nothing else changes.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = "qwen3.5:9b"


@dataclass
class ModelReply:
    ok: bool                      # True only when `data` is a usable JSON object
    data: dict[str, Any] | None
    text: str                     # raw model output, kept for audit
    model: str
    provider: str
    live: bool                    # True only if a real model produced `text`
    latency_s: float
    tokens: int | None = None
    error: str | None = None


class ChatModel(Protocol):
    name: str
    provider: str
    local: bool  # True when we run the model ourselves and may unload it

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> ModelReply: ...


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


class OllamaChatModel:
    provider = "ollama"
    local = True

    def __init__(self, name: str | None = None, url: str = DEFAULT_OLLAMA_URL,
                 timeout: float = 90.0, temperature: float = 0.2):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.name = name or resolve_installed_model(
            os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL), self.url) or DEFAULT_MODEL

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> ModelReply:
        payload = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "format": "json",
            "stream": False,
            "think": False,  # Qwen 3.5 thinking mode costs ~100s per call
            "keep_alive": "30m",
            "options": {"temperature": self.temperature},
        }
        start = time.time()
        try:
            res = _post(f"{self.url}/api/chat", payload, self.timeout)
        except Exception as exc:  # unreachable, timeout, HTTP error
            return ModelReply(False, None, "", self.name, self.provider, False,
                              time.time() - start, error=f"{type(exc).__name__}: {exc}")
        text = (res.get("message") or {}).get("content", "")
        latency = time.time() - start
        tokens = res.get("eval_count")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return ModelReply(False, None, text, self.name, self.provider, True, latency,
                              tokens, error="malformed JSON")
        if not isinstance(data, dict):
            return ModelReply(False, None, text, self.name, self.provider, True, latency,
                              tokens, error="JSON is not an object")
        return ModelReply(True, data, text, self.name, self.provider, True, latency, tokens)

    # --- local model lifecycle (used by kill_controls.stop_local_model) ---------------
    def available(self) -> bool:
        try:
            names = {m["name"] for m in _get(f"{self.url}/api/tags", 3).get("models", [])}
        except Exception:
            return False
        return self.name in names

    def loaded(self) -> bool:
        try:
            names = {m["name"] for m in _get(f"{self.url}/api/ps", 3).get("models", [])}
        except Exception:
            return False
        return self.name in names

    def unload(self) -> dict:
        """Ollama's documented unload: keep_alive=0. Model files are never touched."""
        return _post(f"{self.url}/api/generate", {"model": self.name, "keep_alive": 0}, 30)


def resolve_installed_model(wanted: str, url: str = DEFAULT_OLLAMA_URL) -> str | None:
    """Exact tag if installed, else another tag of the same model (qwen3.5 -> qwen3.5:9b)."""
    try:
        names = [m["name"] for m in _get(f"{url}/api/tags", 3).get("models", [])]
    except Exception:
        return None
    if wanted in names:
        return wanted
    base = wanted.split(":")[0]
    return next((n for n in names if n.split(":")[0] == base), None)


@dataclass
class ScriptedModel:
    """Test double. Replies are canned and always labelled live=False."""
    replies: list[Any] = field(default_factory=list)
    name: str = "scripted-test-model"
    provider: str = "scripted"
    local: bool = False
    calls: list[dict] = field(default_factory=list)

    def complete_json(self, system: str, messages: list[dict[str, str]]) -> ModelReply:
        self.calls.append({"system": system, "messages": messages})
        reply = self.replies.pop(0) if self.replies else {"error": "no scripted reply"}
        if isinstance(reply, Exception):
            return ModelReply(False, None, "", self.name, self.provider, False, 0.0,
                              error=f"{type(reply).__name__}: {reply}")
        if isinstance(reply, str):
            try:
                data = json.loads(reply)
            except json.JSONDecodeError:
                return ModelReply(False, None, reply, self.name, self.provider, False, 0.0,
                                  error="malformed JSON")
            reply = data
        if not isinstance(reply, dict) or "error" in reply and len(reply) == 1:
            return ModelReply(False, None, json.dumps(reply), self.name, self.provider, False,
                              0.0, error=str(reply))
        return ModelReply(True, reply, json.dumps(reply), self.name, self.provider, False, 0.0)
