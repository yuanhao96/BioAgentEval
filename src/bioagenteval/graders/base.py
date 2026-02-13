"""Base grader abstract class."""
from __future__ import annotations

from abc import ABC, abstractmethod

from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript


class BaseGrader(ABC):
    """Abstract base for all grader implementations."""

    @abstractmethod
    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        """Grade an agent's output for one trial."""
        ...
