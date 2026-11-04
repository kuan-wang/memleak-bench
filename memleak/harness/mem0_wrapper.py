"""Mem0 wrapper: industry-standard RAG + delete agent memory.

Mem0 stores facts as text entries with vector embeddings. On forget,
it searches for matching memories and deletes them by ID. But it does
NOT delete image embeddings, derived facts, or implicit visual features
— exactly the IPG's "linked but untracked" failure mode.

Expected behavior:
- High same-modal forgetting success (text entry deleted)
- High cross-modal leakage (image-related memories persist)
- FC between oracle (0.6%) and long-context (23.6%)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from memleak.harness.base import AgentMemorySystem, ForgetResult
from memleak.schema.types import Turn

logger = logging.getLogger(__name__)

# Load .env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


class Mem0System(AgentMemorySystem):
    """Wraps Mem0 (open-source local mode) as an agent memory system.

    Uses Mem0's in-process memory with local embeddings. No cloud API needed.
    The LLM for extraction and probing is pluggable.
    """

    def __init__(
        self,
        llm_fn=None,
        user_id: str = "default",
        config: dict | None = None,
        forget_top_k: int | None = None,
        forget_threshold: float | None = None,
        state_top_k: int = 1000,
        on_memory_created=None,
    ):
        """
        Args:
            llm_fn: Async function(messages) -> str for probe responses.
            user_id: Mem0 user identifier for memory scoping.
            config: Optional ``MemoryConfig`` dict (LLM, embedder, vector store,
                history DB). None keeps the legacy ``Memory()`` defaults.
            forget_top_k / forget_threshold: Optional explicit search
                parameters for forget(); None uses Mem0's own defaults.
            state_top_k: Listing limit for get_memory_state().
            on_memory_created: Optional callback(Memory) after construction
                (e.g. to attach call logging to Mem0's internal LLM client).
        """
        self._llm_fn = llm_fn
        self._user_id = user_id
        self._config = config
        self._forget_top_k = forget_top_k
        self._forget_threshold = forget_threshold
        self._state_top_k = state_top_k
        self._on_memory_created = on_memory_created
        self._mem0 = None
        self._conversation: list[dict] = []

    def _ensure_mem0(self):
        """Lazy init Mem0 to avoid import cost at module level."""
        if self._mem0 is None:
            from mem0 import Memory
            self._mem0 = Memory.from_config(self._config) if self._config else Memory()
            if self._on_memory_created:
                self._on_memory_created(self._mem0)

    def _search_kwargs(self) -> dict:
        kwargs = {}
        if self._forget_top_k is not None:
            kwargs["top_k"] = self._forget_top_k
        if self._forget_threshold is not None:
            kwargs["threshold"] = self._forget_threshold
        return kwargs

    def reset(self) -> None:
        self._ensure_mem0()
        # Delete all memories for this user
        try:
            all_mems = self._mem0.get_all(filters={"user_id": self._user_id})
            if hasattr(all_mems, 'results'):
                for mem in all_mems.results:
                    self._mem0.delete(mem['id'])
            elif isinstance(all_mems, list):
                for mem in all_mems:
                    mid = mem.get('id', mem.get('memory_id'))
                    if mid:
                        self._mem0.delete(mid)
        except Exception as e:
            logger.warning(f"Reset error: {e}")
        self._conversation = []

    async def inject_turn(self, turn: Turn) -> str:
        self._ensure_mem0()

        # Add to Mem0's memory
        messages = [{"role": turn.role, "content": turn.text}]
        try:
            self._mem0.add(messages, user_id=self._user_id)
        except Exception as e:
            logger.warning(f"Mem0 add error: {e}")

        # Track conversation
        self._conversation.append({
            "role": turn.role,
            "content": turn.text,
        })

        if turn.role == "user" and self._llm_fn:
            # Get relevant memories for context
            memories = self._get_relevant_memories(turn.text)
            response = await self._generate_response(turn.text, memories)
            self._conversation.append({"role": "assistant", "content": response})
            return response

        return "Noted, thank you."

    async def forget(self, instruction: str) -> ForgetResult:
        self._ensure_mem0()

        # Search for memories matching the forget instruction
        deleted_items = []
        deleted_memories = []
        errors = []
        try:
            search_results = self._mem0.search(
                instruction, filters={"user_id": self._user_id}, **self._search_kwargs())
            results = search_results.get('results', []) if isinstance(search_results, dict) else search_results

            for mem in results:
                mid = mem.get('id', mem.get('memory_id'))
                if mid:
                    try:
                        self._mem0.delete(mid)
                        deleted_items.append(f"memory:{mid}")
                        deleted_memories.append({"id": mid, "memory": mem.get("memory"),
                                                 "score": mem.get("score")})
                    except Exception as e:
                        logger.warning(f"Delete error for {mid}: {e}")
                        errors.append(f"delete {mid}: {e}")
        except Exception as e:
            logger.warning(f"Mem0 search/delete error: {e}")
            errors.append(f"search: {e}")

        return ForgetResult(
            success=not errors,
            deleted_items=deleted_items,
            method="mem0_search_delete",
            metadata={
                "deleted_count": len(deleted_items),
                "deleted_memories": deleted_memories,
                "search_query": instruction,
                "search_kwargs": self._search_kwargs() or "mem0_defaults",
                "errors": errors,
            },
        )

    async def probe(self, query: str, images: list[str] | None = None) -> str:
        self._ensure_mem0()

        memories = self._get_relevant_memories(query)

        if self._llm_fn:
            return await self._generate_response(query, memories, images)

        # Placeholder
        return f"[Mem0 response: {len(memories)} memories retrieved] {query}"

    def get_memory_state(self) -> dict | None:
        self._ensure_mem0()
        try:
            all_mems = self._mem0.get_all(filters={"user_id": self._user_id},
                                          top_k=self._state_top_k)
            # Mem0 returns various formats — normalize
            if hasattr(all_mems, 'results'):
                results = all_mems.results
            elif isinstance(all_mems, dict):
                results = all_mems.get('results', all_mems.get('memories', []))
            elif isinstance(all_mems, list):
                results = all_mems
            else:
                results = []

            memories = []
            for m in (results or []):
                if isinstance(m, dict):
                    memories.append({"id": m.get("id"), "memory": m.get("memory", m.get("text", ""))})
                elif isinstance(m, str):
                    memories.append({"id": None, "memory": m})

            return {
                "total_memories": len(memories),
                "memories": memories,
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_relevant_memories(self, query: str) -> list[str]:
        """Search Mem0 for memories relevant to the query."""
        try:
            results = self._mem0.search(query, filters={"user_id": self._user_id})
            memories = results.get('results', []) if isinstance(results, dict) else results
            return [
                m.get('memory', m.get('text', ''))
                for m in (memories or [])
                if m.get('memory') or m.get('text')
            ]
        except Exception as e:
            logger.warning(f"Mem0 search error: {e}")
            return []

    async def _generate_response(
        self,
        query: str,
        memories: list[str],
        images: list[str] | None = None,
    ) -> str:
        """Generate a response using memories as context."""
        memory_context = "\n".join(f"- {m}" for m in memories) if memories else "(no stored memories)"

        system_prompt = (
            f"You are a helpful assistant. Here is what you remember about the user:\n"
            f"{memory_context}\n\n"
            f"Use these memories to answer the user's question. "
            f"If you don't have relevant information, say so."
        )

        if self._llm_fn:
            # Check if the llm_fn accepts system prompt style or messages style
            import inspect
            sig = inspect.signature(self._llm_fn)
            params = list(sig.parameters.keys())

            if 'system_prompt' in params:
                # call_*_with_system style
                return await self._llm_fn(system_prompt, query, images or [])
            else:
                # call_* messages style (call_claude, call_openai)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ]
                return await self._llm_fn(messages=messages)

        return f"[Mem0: {len(memories)} memories] {query}"
