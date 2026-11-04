"""Abstract base class for agent memory systems.

Every system under evaluation — long-context models, RAG+delete, agent-managed,
oracle — implements this interface. The runner orchestrates evaluation through
these methods without knowing system internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from memleak.schema.types import Profile, Turn


@dataclass
class ForgetResult:
    """What the system reports after processing a forget instruction."""
    success: bool
    deleted_items: list[str] = field(default_factory=list)
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMemorySystem(ABC):
    """Uniform interface wrapping any agent memory system.

    Implementations must handle:
    - Multi-turn injection (text + images)
    - Forget instructions via the system's native mechanism
    - Probing with optional image queries
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all state for a fresh evaluation run."""

    def initialize(self, profile: Profile) -> None:
        """Start a run before injection; ordinary systems only reset here.

        The runner owns initialization. Only an explicit reference system
        should override this to preload benchmark annotations.
        """
        self.reset()

    @abstractmethod
    async def inject_turn(self, turn: Turn) -> str:
        """Process one conversation turn (text + optional images).

        Args:
            turn: A Turn object with role, text, and optional image paths.

        Returns:
            The system's response text.
        """

    @abstractmethod
    async def forget(self, instruction: str) -> ForgetResult:
        """Issue a forget instruction via the system's native mechanism.

        Args:
            instruction: Natural language forget request
                (e.g., "Please forget where I live").

        Returns:
            ForgetResult with deletion metadata.
        """

    @abstractmethod
    async def probe(self, query: str, images: list[str] | None = None) -> str:
        """Ask a question, optionally with images.

        Args:
            query: The probe question.
            images: Optional list of image paths to include with the query.

        Returns:
            The system's response text.
        """

    def get_memory_state(self) -> dict | None:
        """Optional: return inspectable memory state snapshot.

        Not all systems support this (e.g., long-context models
        don't have separately inspectable memory). Returns None
        if not supported.
        """
        return None
