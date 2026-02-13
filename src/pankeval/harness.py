"""Agent harness protocol for wrapping agents without code changes."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pankeval.models import AgentResponse


@runtime_checkable
class AgentHarness(Protocol):
    """Protocol that any agent must satisfy for evaluation.

    Implement ``run`` and ``reset`` on your agent class. No inheritance
    required — structural subtyping (duck typing) is used.
    """

    def run(self, question: str) -> AgentResponse:
        """Run the agent on a question, return structured response."""
        ...

    def reset(self) -> None:
        """Reset agent state between trials."""
        ...
