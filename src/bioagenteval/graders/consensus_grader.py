"""Multi-judge consensus grader: runs an inner grader multiple times and aggregates."""
from __future__ import annotations

import logging
from typing import Any

from bioagenteval.graders.base import BaseGrader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

logger = logging.getLogger(__name__)


class ConsensusGrader(BaseGrader):
    """Wraps an inner grader and runs it N times to reach consensus.

    Aggregation:
    - ``passed``: strict majority vote (pass_count > fail_count).
      On an even split (e.g., 2 pass / 2 fail), the result is ``False``.
    - ``score``: average of all successful judge scores
    - ``details``: includes individual judge results and vote counts
    """

    def __init__(self, inner_grader: BaseGrader, num_judges: int = 3):
        self.inner_grader = inner_grader
        self.num_judges = num_judges

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        judge_results: list[GradeResult] = []
        errors: list[str] = []

        for i in range(self.num_judges):
            try:
                result = self.inner_grader.grade(
                    task, outcome, transcript, config, metrics=metrics,
                )
                judge_results.append(result)
            except Exception as e:
                logger.warning("ConsensusGrader judge %d failed: %s", i, e)
                errors.append(str(e))

        if not judge_results:
            return GradeResult(
                grader_type="consensus",
                score=0.0,
                passed=False,
                details={
                    "errors": errors,
                    "num_judges": self.num_judges,
                    "num_successful": 0,
                },
            )

        pass_count = sum(1 for r in judge_results if r.passed)
        fail_count = len(judge_results) - pass_count
        avg_score = sum(r.score for r in judge_results) / len(judge_results)
        passed = pass_count > fail_count

        return GradeResult(
            grader_type="consensus",
            score=avg_score,
            passed=passed,
            details={
                "num_judges": self.num_judges,
                "num_successful": len(judge_results),
                "pass_votes": pass_count,
                "fail_votes": fail_count,
                "individual_scores": [r.score for r in judge_results],
                "individual_passed": [r.passed for r in judge_results],
                "errors": errors,
            },
        )
