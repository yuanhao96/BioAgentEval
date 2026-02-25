"""Evaluation runner: orchestrates running tasks against an agent."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bioagenteval.harness import AgentHarness
from bioagenteval.metrics import compute_metrics
from bioagenteval.models import EvalResult, GradeResult, Task, Transcript, TrialResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates running an evaluation suite against an agent."""

    def __init__(
        self,
        agent: AgentHarness,
        graders: dict[str, Any],
        max_concurrency: int = 1,
        max_retries: int = 0,
    ):
        self.agent = agent
        self.graders = graders
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries

    def run_suite(self, tasks: list[Task]) -> list[EvalResult]:
        if self.max_concurrency <= 1:
            return self._run_suite_sequential(tasks)
        return self._run_suite_parallel(tasks)

    def _run_suite_sequential(self, tasks: list[Task]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for task in tasks:
            logger.info("Running task %s (%d trials)", task.id, task.num_trials)
            result = self.run_task(task)
            results.append(result)
        return results

    def _run_suite_parallel(self, tasks: list[Task]) -> list[EvalResult]:
        results: dict[str, EvalResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            future_to_task = {
                pool.submit(self.run_task, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results[task.id] = future.result()
                except Exception as e:
                    logger.error("Task %s failed: %s", task.id, e)
                    results[task.id] = EvalResult(task_id=task.id, trials=[])
        # Preserve original task order
        return [results[task.id] for task in tasks]

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

        # Compute metrics from tracked_metrics groups
        trial_metrics = compute_metrics(
            task.tracked_metrics, transcript, duration_ms,
        )

        grades = []
        for grader_config in task.graders:
            grader = self.graders.get(grader_config.type)
            if grader is None:
                logger.warning("No grader registered for type: %s", grader_config.type)
                continue
            grade = self._grade_with_retry(
                grader, task, outcome, transcript, grader_config, trial_metrics,
            )
            grade.weight = grader_config.weight
            grades.append(grade)

        # Invert pass/fail for negative test cases
        if task.should_fail:
            for grade in grades:
                grade.passed = not grade.passed
                grade.score = 1.0 - grade.score

        return TrialResult(
            task_id=task.id,
            trial_num=trial_num,
            outcome=outcome,
            transcript=transcript,
            grades=grades,
            duration_ms=duration_ms,
            error=error,
            metrics=trial_metrics,
        )

    def _grade_with_retry(
        self, grader, task, outcome, transcript, config, metrics,
    ) -> GradeResult:
        """Grade with exponential backoff retry on failure."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return grader.grade(
                    task, outcome, transcript, config, metrics=metrics,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = 2 ** attempt  # 1s, 2s, 4s, ...
                    logger.warning(
                        "Grader %s failed (attempt %d/%d): %s. Retrying in %ds.",
                        config.type, attempt + 1, self.max_retries + 1, e, delay,
                    )
                    time.sleep(delay)

        logger.error("Grader %s failed after %d attempts: %s", config.type, self.max_retries + 1, last_error)
        return GradeResult(
            grader_type=config.type,
            score=0.0,
            passed=False,
            details={"error": str(last_error)},
        )
