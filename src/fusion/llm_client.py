"""
Unified LLM client — supports Ollama (local, open-source) and Claude (cloud).

Usage:
    # Auto-detect: tries Ollama first, falls back to Claude
    client = LLMClient.auto()

    # Force Ollama (llama3.2-vision, mistral, etc.)
    client = LLMClient.ollama(model="llama3.2-vision")

    # Force Claude
    client = LLMClient.claude(model="claude-sonnet-4-20250514")

    response = client.chat("Describe this video frame.", image_b64=frame_b64)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from dataclasses import dataclass

from src.utils.common import get_logger

log = get_logger(__name__)


# ─── Response wrapper ─────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    model: str
    backend: str          # "ollama" | "claude" | "stub"
    input_tokens: int = 0
    output_tokens: int = 0


# ─── Ollama backend ───────────────────────────────────────────────────────────

class OllamaClient:
    """
    Calls a locally running Ollama server.
    Supports vision models: llama3.2-vision, llava, bakllava, moondream.
    Text-only models: mistral, llama3, phi3, gemma2, qwen2.5.

    Install Ollama: https://ollama.com
    Pull a vision model: ollama pull llama3.2-vision
    """

    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: str = "llama3.2-vision",
        host: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST", self.DEFAULT_HOST)).rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        image_b64: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        user_msg: dict = {"role": "user", "content": prompt}
        if image_b64:
            user_msg["images"] = [image_b64]
        messages.append(user_msg)

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="ignore")[-400:]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama connection error: {exc.reason}") from exc

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Ollama response type: {type(data)}")
        text = data.get("message", {}).get("content", "") or ""
        return LLMResponse(
            text=text,
            model=self.model,
            backend="ollama",
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    def list_models(self) -> list[str]:
        req = urllib.request.Request(f"{self.host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]


# ─── Claude backend ───────────────────────────────────────────────────────────

class ClaudeClient:
    """
    Calls the Anthropic Claude API.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: int = 60,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        image_b64: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or switch to Ollama: LLMClient.ollama()"
            )

        content: list[dict] = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        body: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            self.API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="ignore")[-400:]
            raise RuntimeError(f"Claude HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Claude connection error: {exc.reason}") from exc

        if not isinstance(data, dict) or "content" not in data:
            raise RuntimeError(f"Unexpected Claude response structure: {str(data)[:200]}")
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=self.model,
            backend="claude",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


# ─── Stub backend (no API key, no Ollama) ────────────────────────────────────

class StubClient:
    """Returns deterministic placeholder responses for testing/CI."""

    def is_available(self) -> bool:
        return True

    def chat(self, prompt: str, system: str | None = None,
             image_b64: str | None = None, temperature: float = 0.2) -> LLMResponse:
        log.debug("StubClient: returning placeholder response")
        return LLMResponse(
            text="[stub] Objects: vehicle (0.85), person (0.72) | Scene: outdoor street | Activity: traffic",
            model="stub",
            backend="stub",
        )


# ─── Unified client facade ────────────────────────────────────────────────────

class LLMClient:
    """
    Unified facade. Delegates to OllamaClient, ClaudeClient, or StubClient.

    Priority (auto mode):
      1. Ollama (if server is running locally)
      2. Claude (if ANTHROPIC_API_KEY is set)
      3. Stub  (for testing / CI)
    """

    def __init__(self, backend: OllamaClient | ClaudeClient | StubClient) -> None:
        self._backend = backend
        log.info(
            "LLM backend: %s | model: %s",
            type(self._backend).__name__,
            getattr(self._backend, "model", "stub"),
        )

    def is_available(self) -> bool:
        """Check if the underlying backend is available."""
        if hasattr(self._backend, "is_available"):
            return self._backend.is_available()
        return True

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def auto(
        cls,
        ollama_model: str = "llama3.2-vision",
        claude_model: str = ClaudeClient.DEFAULT_MODEL,
    ) -> "LLMClient":
        """Try Ollama → Claude → Stub."""
        ollama = OllamaClient(model=ollama_model)
        if ollama.is_available():
            log.info("Auto-selected: Ollama (%s)", ollama_model)
            return cls(ollama)

        claude = ClaudeClient(model=claude_model)
        if claude.is_available():
            log.info("Auto-selected: Claude (%s)", claude_model)
            return cls(claude)

        log.warning("No LLM available — using stub backend")
        return cls(StubClient())

    @classmethod
    def ollama(cls, model: str = "llama3.2-vision", host: str | None = None) -> "LLMClient":
        return cls(OllamaClient(model=model, host=host))

    @classmethod
    def claude(cls, model: str = ClaudeClient.DEFAULT_MODEL) -> "LLMClient":
        return cls(ClaudeClient(model=model))

    @classmethod
    def stub(cls) -> "LLMClient":
        return cls(StubClient())

    # ── Delegate ──────────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        image_b64: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        return self._backend.chat(
            prompt=prompt, system=system,
            image_b64=image_b64, temperature=temperature,
        )

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__

    @property
    def model_name(self) -> str:
        return getattr(self._backend, "model", "stub")
