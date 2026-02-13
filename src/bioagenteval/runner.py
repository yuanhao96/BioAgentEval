"""Evaluation runner: orchestrates running tasks against an agent."""
from __future__ import annotations

import logging
import time
from typing import Any

from bioagenteval.harness import AgentHarness
from bioagenteval.models import EvalResult, Task, Transcript, TrialResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates running an evaluation suite against an agent."""

    def __init__(self, agent: AgentHarness, graders: dict[str, Any]):
        self.agent = agent
        self.graders = graders

    def run_suite(self, tasks: list[Task]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for task in tasks:
            logger.info("Running task %s (%d trials)", task.id, task.num_trials)
            result = self.run_task(task)
            results.append(result)
        return results

    def run_task(self, task: Task) -> EvalResult:
        trials: list[TrialResult] = []
        for trial_num in range(task.num_trials):
            trial = self._run_trial(task, trial_num)
            trials.append(trial)
        return EvalResult(task_id=task.id, trials=trials)

    def _run_trial(self, task: Task, trial_num: int) -> TrialResult:
        self.agent.reset()

        start = time.perf_counter()
        try:
            response = self.agent.run(task.question)
            duration_ms = (time.perf_counter() - start) * 1000
            outcome = response.outcome
            transcript = response.transcript
            error = None
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning("Trial %d of task %s failed: %s", trial_num, task.id, e)
            outcome = ""
            transcript = Transcript(task_id=task.id)
            error = str(e)

        grades = []
        for grader_config in task.graders:
            grader = self.graders.get(grader_config.type)
            if grader is None:
                logger.warning("No grader registered for type: %s", grader_config.type)
                continue
            grade = grader.grade(task, outcome, transcript, grader_config)
            grades.append(grade)

        return TrialResult(
            task_id=task.id,
            trial_num=trial_num,
            outcome=outcome,
            transcript=transcript,
            grades=grades,
            duration_ms=duration_ms,
            error=error,
        )
