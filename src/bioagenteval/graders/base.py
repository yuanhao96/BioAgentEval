"""Base grader abstract class and shared utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript


def format_expected_output(task: Task) -> str:
    """Format expected_output items for grading prompts."""
    if not task.expected_output:
        return "N/A"
    parts = []
    for eo in task.expected_output:
        parts.append(f"- Type: {eo.type}, Value: {eo.value}")
    return "\n".join(parts)


class BaseGrader(ABC):
    """Abstract base for all grader implementations."""

    @abstractmethod
    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        """Grade an agent's output for one trial."""
        ...
