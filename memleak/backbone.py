"""Recorded LLM backbone for the text-comparison system runs.

Every system that calls the backbone directly (Oracle, Naive, LongContext and
Mem0's answer generation) goes through ``Backbone`` so that one model, one set
of sampling parameters and one call log apply to all of them. Each call is
appended to a JSONL log with the request, the complete response text, the
provider-resolved model ID, token usage, stop reason and latency.

The log never contains credentials; the client reads keys from the process
environment (``memleak.llm`` loads the repo .env).
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

import memleak.llm  # noqa: F401  (loads .env into the environment)

logger = logging.getLogger(__name__)

# Set by the experiment driver so each logged call can be attributed.
call_context: contextvars.ContextVar[dict] = contextvars.ContextVar("call_context", default={})

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class JsonlLog:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if not self.path:
            return
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")


def _image_block(path: str) -> dict:
    p = Path(path)
    suffix = p.suffix.lstrip(".").lower()
    media = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return {"type": "image", "source": {
        "type": "base64", "media_type": media,
        "data": base64.b64encode(p.read_bytes()).decode()}}


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI-style messages -> (system text, Anthropic messages).

    System messages anywhere in the list are concatenated into the system
    parameter, matching ``memleak.llm.call_claude``.
    """
    system_parts: list[str] = []
    converted: list[dict] = []
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    header, data = url.split(",", 1)
                    parts.append({"type": "image", "source": {
                        "type": "base64", "media_type": header.split(":")[1].split(";")[0],
                        "data": data}})
            content = parts
        converted.append({"role": msg["role"], "content": content})
    return "\n".join(system_parts).strip(), converted


def _redact_images(obj: Any) -> Any:
    """Replace base64 payloads in a logged request with their length."""
    if isinstance(obj, dict):
        if obj.get("type") == "base64" and "data" in obj:
            return {**obj, "data": f"<{len(obj['data'])} base64 chars>"}
        if obj.get("type") == "image_url":
            return {"type": "image_url", "image_url": "<data uri omitted>"}
        return {k: _redact_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_images(v) for v in obj]
    return obj


class Backbone:
    """Anthropic Messages API backbone with retries and a complete call log."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        log_path: Path | None = None,
        max_retries: int = 8,
        concurrency: int = 8,
    ):
        import anthropic

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.log = JsonlLog(log_path)
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com",
            max_retries=0,  # retries are handled (and logged) here
        )
        self._sem = asyncio.Semaphore(concurrency)

    def settings(self) -> dict:
        return {"provider": "anthropic", "transport": "anthropic_messages_api",
                "model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "extended_thinking": False}

    async def complete(self, messages: list[dict]) -> str:
        """OpenAI-style message list in, response text out."""
        system, converted = to_anthropic(messages)
        return await self._create(system, converted)

    async def system_user(self, system_prompt: str, user_message: str,
                          images: list[str] | None = None) -> str:
        """Signature used by OracleExplicit/NaiveBaseline/Mem0System."""
        real = [p for p in (images or []) if not p.startswith("PLACEHOLDER:") and Path(p).exists()]
        if real:
            content: Any = [{"type": "text", "text": user_message}] + [_image_block(p) for p in real]
        else:
            content = user_message
        return await self._create(system_prompt, [{"role": "user", "content": content}])

    async def _create(self, system: str, messages: list[dict]) -> str:
        import anthropic

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages,
                                  "max_tokens": self.max_tokens, "temperature": self.temperature}
        if system:
            kwargs["system"] = system
        ctx = dict(call_context.get())
        errors = []
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                async with self._sem:
                    resp = await self._client.messages.create(**kwargs)
            except anthropic.APIStatusError as e:
                errors.append({"status": e.status_code, "type": type(e).__name__,
                               "message": str(e.message)[:300]})
                if e.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    self._log(ctx, kwargs, None, started, errors)
                    raise
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                errors.append({"type": type(e).__name__, "message": str(e)[:300]})
                if attempt == self.max_retries:
                    self._log(ctx, kwargs, None, started, errors)
                    raise
            else:
                text = "".join(b.text for b in resp.content if b.type == "text")
                self._log(ctx, kwargs, resp, started, errors, text)
                return text
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
        raise RuntimeError("unreachable")

    def _log(self, ctx, kwargs, resp, started, errors, text=None) -> None:
        record = {
            **ctx,
            "kind": "backbone",
            "ts": started,
            "latency_s": round(time.time() - started, 3),
            "requested_model": kwargs["model"],
            "temperature": kwargs["temperature"],
            "max_tokens": kwargs["max_tokens"],
            "system": kwargs.get("system", ""),
            "messages": _redact_images(kwargs["messages"]),
            "retries": errors,
        }
        if resp is not None:
            usage = resp.usage
            record.update({
                "resolved_model": resp.model,
                "response_id": resp.id,
                "stop_reason": resp.stop_reason,
                "response_text": text,
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
                    "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                },
            })
        else:
            record["failed"] = True
        self.log.write(record)
