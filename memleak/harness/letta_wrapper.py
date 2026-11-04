"""Letta server integration: a real, version-pinned Letta agent per profile.

This replaces the simulated wrapper used for the published "Letta (sim.)" row
(now ``memleak.harness.letta_simulated``). Every turn, the forget instruction
and every probe are sent as ordinary user messages to a stateful agent on a
running Letta server (pinned: letta==0.16.8, REST API via letta-client).

Deletion is Letta's own: the forget instruction is delivered to the agent,
which decides what to remove and applies its built-in memory-editing tools
(for the default ``letta_v1_agent``: ``memory_replace``/``memory_insert`` on
core-memory blocks; deleting text is ``memory_replace(old_string, "")``).
The harness performs no deletion of its own. Letta's recall memory (the
persisted conversation, of which the recent part stays in the context window)
is not agent-deletable in this configuration and is therefore retained; the
wrapper records it, together with block contents before and after the forget,
every tool call, and per-request token usage.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from memleak.harness.base import AgentMemorySystem, ForgetResult
from memleak.harness.letta_simulated import LettaSimulated  # noqa: F401  (legacy re-export)
from memleak.schema.types import Profile, Turn

logger = logging.getLogger(__name__)

DEFAULT_PERSONA = "I am a helpful personal assistant with long-term memory."
LETTA_SERVER_VERSION = "0.16.8"


def _assistant_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m.get("message_type") != "assistant_message":
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(c.get("text", "") for c in content if isinstance(c, dict))
    return "\n".join(p for p in parts if p)


def _image_part(path: str) -> dict:
    p = Path(path)
    suffix = p.suffix.lstrip(".").lower()
    media = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return {"type": "image", "source": {
        "type": "base64", "media_type": media,
        "data": base64.b64encode(p.read_bytes()).decode()}}


class LettaServerSystem(AgentMemorySystem):
    """One fresh Letta agent per profile on a running Letta server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8283",
        model: str = "anthropic/claude-sonnet-4-5-20250929",
        embedding: str = "openai/text-embedding-3-small",
        agent_type: str = "letta_v1_agent",
        model_settings: dict | None = None,
        reasoning: bool = False,
        persona: str = DEFAULT_PERSONA,
        name_prefix: str = "memleak",
        timeout_s: float = 600.0,
        client: Any = None,
    ):
        if client is None:
            from letta_client import Letta
            client = Letta(base_url=base_url, timeout=timeout_s, max_retries=0)
        self._client = client
        self.base_url = base_url
        self.model = model
        self.embedding = embedding
        self.agent_type = agent_type
        self.model_settings = model_settings
        self.reasoning = reasoning
        self.persona = persona
        self.name_prefix = name_prefix
        self.agent_id: str | None = None
        self.agent_info: dict = {}
        self.exchanges: list[dict] = []   # every request/response, in order
        self._profile_id = "unbound"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, profile: Profile) -> None:
        self._profile_id = profile.profile_id
        self.reset()

    def reset(self) -> None:
        """Start a brand-new agent; earlier agents are kept on the server for audit."""
        kwargs: dict[str, Any] = dict(
            name=f"{self.name_prefix}-{self._profile_id}-{int(time.time() * 1000)}",
            agent_type=self.agent_type,
            model=self.model,
            embedding=self.embedding,
            memory_blocks=[{"label": "human", "value": ""},
                           {"label": "persona", "value": self.persona}],
            reasoning=self.reasoning,
            tags=["memleak", self._profile_id],
        )
        if self.model_settings:
            kwargs["model_settings"] = self.model_settings
        agent = self._client.agents.create(**kwargs)
        self.agent_id = agent.id
        self.exchanges = []
        self.agent_info = {
            "agent_id": agent.id,
            "name": agent.name,
            "agent_type": str(agent.agent_type),
            "tools": sorted(t.name for t in agent.tools),
            "llm_config": agent.llm_config.model_dump(mode="json"),
            "embedding_config": agent.embedding_config.model_dump(mode="json")
            if agent.embedding_config else None,
            "system_prompt_chars": len(agent.system or ""),
            "persona": self.persona,
            "requested_model_settings": self.model_settings,
            "reasoning": self.reasoning,
        }

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    def _send(self, content: Any, kind: str) -> str:
        started = time.time()
        resp = self._client.agents.messages.create(
            agent_id=self.agent_id, messages=[{"role": "user", "content": content}])
        messages = [m.model_dump(mode="json") for m in resp.messages]
        usage = resp.usage.model_dump(mode="json") if resp.usage else None
        text = _assistant_text(messages)
        self.exchanges.append({
            "kind": kind,
            "request": content if isinstance(content, str) else [
                {**c, "source": {**c["source"], "data": f"<{len(c['source']['data'])} base64 chars>"}}
                if c.get("type") == "image" else c for c in content],
            "messages": messages,
            "stop_reason": resp.stop_reason.model_dump(mode="json") if getattr(resp, "stop_reason", None) else None,
            "usage": usage,
            "assistant_text": text,
            "latency_s": round(time.time() - started, 3),
        })
        return text

    async def inject_turn(self, turn: Turn) -> str:
        if turn.role != "user":
            # Letta writes its own replies; scripted assistant turns are not injectable.
            return turn.text
        content: Any = turn.text
        if turn.images:
            content = [{"type": "text", "text": turn.text}] + [_image_part(p) for p in turn.images]
        return self._send(content, "inject")

    async def forget(self, instruction: str) -> ForgetResult:
        before = self._blocks()
        archival_before = self._passage_count()
        reply = self._send(instruction, "forget")
        after = self._blocks()
        archival_after = self._passage_count()
        tool_calls = [
            {"name": m["tool_call"]["name"], "arguments": m["tool_call"].get("arguments")}
            for m in self.exchanges[-1]["messages"]
            if m.get("message_type") == "tool_call_message" and m.get("tool_call")
        ]
        removed = []
        for label, old in before.items():
            new_lines = set(after.get(label, "").splitlines())
            removed += [f"block:{label}:{line}" for line in old.splitlines()
                        if line.strip() and line not in new_lines]
        return ForgetResult(
            success=True,
            deleted_items=removed,
            method="letta_agent_native_memory_tools",
            metadata={
                "tool_calls": tool_calls,
                "blocks_before": before,
                "blocks_after": after,
                "archival_passages_before": archival_before,
                "archival_passages_after": archival_after,
                "assistant_reply": reply,
                "recall_messages_retained": self._message_count(),
            },
        )

    async def probe(self, query: str, images: list[str] | None = None) -> str:
        content: Any = query
        if images:
            content = [{"type": "text", "text": query}] + [_image_part(p) for p in images]
        return self._send(content, "probe")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def _blocks(self) -> dict[str, str]:
        return {b.label: b.value for b in self._client.agents.blocks.list(agent_id=self.agent_id)}

    def _passage_count(self) -> int:
        return len(list(self._client.agents.passages.list(agent_id=self.agent_id, limit=1000)))

    def _message_count(self) -> int:
        return len(list(self._client.agents.messages.list(agent_id=self.agent_id, limit=1000)))

    def get_memory_state(self) -> dict | None:
        if not self.agent_id:
            return None
        agent = self._client.agents.retrieve(agent_id=self.agent_id)
        return {
            "agent_id": self.agent_id,
            "blocks": self._blocks(),
            "archival_passages": self._passage_count(),
            "recall_messages": self._message_count(),
            "in_context_messages": len(agent.message_ids or []),
        }

    def export_recall(self) -> list[dict]:
        """Complete persisted conversation (recall memory), oldest first."""
        msgs = self._client.agents.messages.list(agent_id=self.agent_id, limit=1000, order="asc")
        return [m.model_dump(mode="json") for m in msgs]
