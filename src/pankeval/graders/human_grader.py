"""Human grader stub -- returns pending results for manual review."""
from __future__ import annotations

import logging

from pankeval.graders.base import BaseGrader
from pankeval.models import GradeResult, GraderConfig, Task, Transcript

logger = logging.getLogger(__name__)


class HumanGrader(BaseGrader):
    """Stub grader that flags results for human review."""

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        logger.info("Task %s flagged for human review", task.id)
        return GradeResult(
            grader_type="human",
            score=0.0,
            passed=False,
            details={
                "status": "pending_human_review",
                "task_id": task.id,
                "outcome_preview": outcome[:200],
            },
        )
