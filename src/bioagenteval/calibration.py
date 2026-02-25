"""Grader calibration: validate grader accuracy against labeled examples."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bioagenteval.models import GraderConfig, Task, Transcript


@dataclass
class CalibrationResult:
    """Result of calibrating a grader against labeled examples."""
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: dict[str, int]  # tp, fp, tn, fn
    total: int
    details: list[dict[str, Any]] = field(default_factory=list)


def calibrate_grader(
    grader: Any,
    examples: list[dict[str, Any]],
) -> CalibrationResult:
    """Run a grader on labeled examples and compute accuracy metrics.

    Each example dict must contain:
    - task: Task instance
    - outcome: str (agent output)
    - transcript: Transcript instance
    - config: GraderConfig instance
    - expected_passed: bool (ground truth)

    Optionally:
    - metrics: dict (passed to grader.grade())

    Returns a CalibrationResult with accuracy, precision, recall,
    and a confusion matrix.
    """
    if not examples:
        return CalibrationResult(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            confusion_matrix={"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            total=0,
        )

    tp = fp = tn = fn = 0
    details: list[dict[str, Any]] = []

    for ex in examples:
        task = ex["task"]
        outcome = ex["outcome"]
        transcript = ex["transcript"]
        config = ex["config"]
        expected = ex["expected_passed"]
        metrics = ex.get("metrics")

        try:
            result = grader.grade(task, outcome, transcript, config, metrics=metrics)
            actual = result.passed
            error = None
        except Exception as e:
            actual = False
            error = str(e)

        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif not expected and not actual:
            tn += 1
        else:
            fn += 1

        details.append({
            "task_id": task.id,
            "expected": expected,
            "actual": actual,
            "correct": expected == actual,
            "error": error,
        })

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return CalibrationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        total=total,
        details=details,
    )
