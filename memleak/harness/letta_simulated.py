"""LEGACY simulated Letta/MemGPT wrapper (not a Letta server).

Kept only to reproduce the published text-only "Letta (sim.)" row. It never
runs Letta: an LLM extracts facts into a Python list and, on forget, returns
list indices to pop. The text-comparison protocol uses the real, version-pinned
Letta server integration in ``memleak.harness.letta_wrapper``.

Original description: Letta/MemGPT wrapper: agent self-managed memory.

Letta agents decide what to remember and forget via tool calls.
When asked to forget, the agent must figure out what to delete
from its own memory using its available tools.

This tests whether an LLM agent is "smart enough" to perform
cross-modal deletion when given the tools to do so.

Expected: between Mem0 and Oracle — the agent may proactively
delete related facts, but is limited by its available tools
(text-only, no image deletion capability).
"""

from __future__ import annotations

import logging

from memleak.harness.base import AgentMemorySystem, ForgetResult
from memleak.schema.types import Turn

logger = logging.getLogger(__name__)


class LettaSimulated(AgentMemorySystem):
    """Simulates Letta/MemGPT-style agent-managed memory.

    Instead of requiring the full Letta server, this simulates the
    key behavior: the agent sees a conversation history + a memory
    store, and decides what to delete when given a forget instruction.

    The LLM is asked to: (1) identify which memories relate to the
    forget request, (2) decide what to delete, (3) generate a response
    using only non-deleted memories.

    This captures the essential IPG behavior: the agent can only
    delete what it can articulate in text, and is limited by its
    own reasoning about what to forget.
    """

    def __init__(self, llm_fn=None):
        """
        Args:
            llm_fn: Async function(messages) -> str. Full message-based LLM.
        """
        self._llm_fn = llm_fn
        self._memories: list[str] = []
        self._conversation: list[dict] = []

    def reset(self) -> None:
        self._memories = []
        self._conversation = []

    async def inject_turn(self, turn: Turn) -> str:
        self._conversation.append({"role": turn.role, "content": turn.text})

        # Extract memories from user turns
        if turn.role == "user" and self._llm_fn:
            extract_prompt = [
                {"role": "system", "content": (
                    "You are a memory management agent. Extract key facts from "
                    "the user's message as a JSON list of strings. Only include "
                    "concrete, specific facts. Output ONLY a JSON array."
                )},
                {"role": "user", "content": turn.text},
            ]
            try:
                import json
                raw = await self._llm_fn(messages=extract_prompt)
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                    if clean.endswith("```"):
                        clean = clean[:-3]
                    clean = clean.strip()
                facts = json.loads(clean)
                if isinstance(facts, list):
                    self._memories.extend(str(f) for f in facts)
            except Exception as e:
                logger.debug(f"Memory extraction failed: {e}")
                # Fall back: store the whole message as a memory
                self._memories.append(turn.text)

        if turn.role == "user" and self._llm_fn:
            response = await self._respond(turn.text)
            self._conversation.append({"role": "assistant", "content": response})
            return response

        return "Noted."

    async def forget(self, instruction: str) -> ForgetResult:
        if not self._llm_fn:
            return ForgetResult(success=False, method="letta_simulated")

        # Ask the agent to identify which memories to delete
        memory_list = "\n".join(f"[{i}] {m}" for i, m in enumerate(self._memories))

        forget_prompt = [
            {"role": "system", "content": (
                "You are a memory management agent. The user wants to forget "
                "certain information. Review the memory store and identify which "
                "memories should be deleted.\n\n"
                f"Current memories:\n{memory_list}\n\n"
                "Output ONLY a JSON array of indices to delete. Example: [0, 3, 5]"
            )},
            {"role": "user", "content": instruction},
        ]

        deleted = []
        try:
            import json
            raw = await self._llm_fn(messages=forget_prompt)
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            indices = json.loads(clean)
            if isinstance(indices, list):
                # Delete in reverse order to preserve indices
                for idx in sorted(indices, reverse=True):
                    if 0 <= idx < len(self._memories):
                        deleted.append(self._memories.pop(idx))
        except Exception as e:
            logger.warning(f"Letta forget failed: {e}")

        return ForgetResult(
            success=True,
            deleted_items=[f"memory:{d[:50]}" for d in deleted],
            method="letta_simulated",
            metadata={
                "deleted_count": len(deleted),
                "remaining_count": len(self._memories),
            },
        )

    async def probe(self, query: str, images: list[str] | None = None) -> str:
        if not self._llm_fn:
            return f"[Letta: {len(self._memories)} memories] {query}"
        return await self._respond(query, images)

    def get_memory_state(self) -> dict:
        return {
            "total_memories": len(self._memories),
            "memories": self._memories[:20],  # cap for readability
        }

    async def _respond(self, query: str, images: list[str] | None = None) -> str:
        memory_context = "\n".join(f"- {m}" for m in self._memories) if self._memories else "(empty)"

        messages = [
            {"role": "system", "content": (
                f"You are a helpful assistant. Your memory store contains:\n"
                f"{memory_context}\n\n"
                f"Use these memories to answer. If you don't have relevant info, say so."
            )},
            {"role": "user", "content": query},
        ]

        return await self._llm_fn(messages=messages)
