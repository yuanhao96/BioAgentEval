"""Agent harness protocol for wrapping agents without code changes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bioagenteval.models import AgentResponse

if TYPE_CHECKING:
    from bioagenteval.models import Task, TrialResult


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


@runtime_checkable
class TrialHook(Protocol):
    """Protocol for trial lifecycle hooks.

    Hooks run before and after each trial for environment setup/teardown
    (e.g., database reset, file cleanup, state snapshot).

    Implementations must be thread-safe if used with max_concurrency > 1.
    """

    def setup(self, task: Task) -> None:
        """Called before each trial's agent.run()."""
        ...

    def teardown(self, task: Task, trial: TrialResult) -> None:
        """Called after grading completes for a trial."""
        ...
