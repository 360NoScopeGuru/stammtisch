"""Streaming client for any OpenAI-compatible local server.

Works unchanged against Ollama (:11434/v1), LM Studio (:1234/v1) and
llama.cpp's server (:8080/v1).
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import httpx

from .config import Config

log = logging.getLogger(__name__)

Message = dict[str, str]


class LlmClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.llm.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.llm.api_key}"},
            timeout=httpx.Timeout(120.0, connect=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[bool, str]:
        """Check the server is up and the configured model is present."""
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
        except Exception as e:
            return False, f"cannot reach {self.cfg.llm.base_url}: {e}"

        names = {m.get("id", "") for m in r.json().get("data", [])}
        missing = [
            m for m in {self.cfg.llm.model, self.cfg.llm.corrector_model}
            if m not in names
        ]
        if missing:
            return False, (
                f"model(s) not served: {missing}\n"
                f"available: {sorted(names) or '(none)'}"
            )
        return True, "ok"

    async def pin_model(self) -> str | None:
        """Ask Ollama to keep the model resident and size its context.

        Ollama unloads after 5 minutes idle by default, so the first reply
        after the learner pauses to read costs a full model load — measured at
        16 s on this machine, which reads as the app being broken.

        This uses Ollama's native endpoint, not the OpenAI-compatible one,
        which has no way to express keep_alive. Any other server just 404s and
        we carry on.
        """
        base = self.cfg.llm.base_url.rstrip("/")
        native = base[:-3] if base.endswith("/v1") else base

        try:
            r = await self._client.post(
                f"{native}/api/generate",
                json={
                    "model": self.cfg.llm.model,
                    "prompt": "",          # load only, generate nothing
                    "keep_alive": self.cfg.llm.keep_alive,
                    "options": {"num_ctx": self.cfg.llm.num_ctx},
                },
                timeout=httpx.Timeout(300.0, connect=5.0),
            )
            if r.status_code == 200:
                log.info(
                    "pinned %s (keep_alive=%s, num_ctx=%d)",
                    self.cfg.llm.model, self.cfg.llm.keep_alive,
                    self.cfg.llm.num_ctx,
                )
                return None
            return f"HTTP {r.status_code}"
        except Exception as e:
            return str(e)[:120]

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield content deltas as they arrive."""
        payload = {
            "model": self.cfg.llm.model,
            "messages": messages,
            "temperature": self.cfg.llm.temperature,
            "max_tokens": self.cfg.llm.max_tokens,
            "stream": True,
        }

        async with self._client.stream("POST", "/chat/completions", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"LLM {r.status_code}: {body[:400]}")

            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if content := delta.get("content"):
                    yield content

    async def complete_json(self, messages: list[Message], *, model: str | None = None,
                            temperature: float | None = None) -> dict | None:
        """Non-streaming call that expects a JSON object back. Returns None on
        any failure — the correction path must never break the conversation."""
        payload = {
            "model": model or self.cfg.llm.corrector_model,
            "messages": messages,
            "temperature": (
                self.cfg.llm.corrector_temperature if temperature is None else temperature
            ),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return _loads_lenient(text)
        except Exception:
            log.exception("correction pass failed (non-fatal)")
            return None


def _loads_lenient(text: str) -> dict | None:
    """Parse JSON that may be wrapped in prose or a ```json fence."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
