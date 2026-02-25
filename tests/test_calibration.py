"""Tests for grader calibration module."""
from __future__ import annotations

import pytest

from bioagenteval.calibration import CalibrationResult, calibrate_grader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript


class _PerfectGrader:
    """Always returns the expected result (passes when outcome is 'good')."""
    def grade(self, task, outcome, transcript, config, metrics=None):
        passed = outcome == "good"
        return GradeResult(grader_type="test", score=1.0 if passed else 0.0, passed=passed)


class _AlwaysPassGrader:
    """Always returns passed=True."""
    def grade(self, task, outcome, transcript, config, metrics=None):
        return GradeResult(grader_type="test", score=1.0, passed=True)


class _AlwaysFailGrader:
    """Always returns passed=False."""
    def grade(self, task, outcome, transcript, config, metrics=None):
        return GradeResult(grader_type="test", score=0.0, passed=False)


class _ErrorGrader:
    """Always raises an exception."""
    def grade(self, task, outcome, transcript, config, metrics=None):
        raise RuntimeError("Grader crashed")


def _make_example(task_id: str, outcome: str, expected_passed: bool):
    return {
        "task": Task(id=task_id, question="Q?"),
        "outcome": outcome,
        "transcript": Transcript(task_id=task_id),
        "config": GraderConfig(type="test"),
        "expected_passed": expected_passed,
    }


class TestCalibrateGrader:
    def test_perfect_grader(self):
        examples = [
            _make_example("t1", "good", True),
            _make_example("t2", "bad", False),
            _make_example("t3", "good", True),
            _make_example("t4", "bad", False),
        ]

        result = calibrate_grader(_PerfectGrader(), examples)

        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.confusion_matrix == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
        assert result.total == 4

    def test_always_pass_grader(self):
        examples = [
            _make_example("t1", "good", True),
            _make_example("t2", "bad", False),
        ]

        result = calibrate_grader(_AlwaysPassGrader(), examples)

        assert result.accuracy == 0.5
        assert result.precision == 0.5  # 1 TP / (1 TP + 1 FP)
        assert result.recall == 1.0  # 1 TP / (1 TP + 0 FN)
        assert result.confusion_matrix == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}

    def test_always_fail_grader(self):
        examples = [
            _make_example("t1", "good", True),
            _make_example("t2", "bad", False),
        ]

        result = calibrate_grader(_AlwaysFailGrader(), examples)

        assert result.accuracy == 0.5
        assert result.precision == 0.0  # 0 TP / (0 TP + 0 FP) → 0.0
        assert result.recall == 0.0  # 0 TP / (0 TP + 1 FN) → 0.0
        assert result.confusion_matrix == {"tp": 0, "fp": 0, "tn": 1, "fn": 1}

    def test_empty_examples(self):
        result = calibrate_grader(_PerfectGrader(), [])

        assert result.accuracy == 0.0
        assert result.total == 0

    def test_grader_error_treated_as_fail(self):
        examples = [
            _make_example("t1", "good", True),
        ]

        result = calibrate_grader(_ErrorGrader(), examples)

        assert result.accuracy == 0.0
        assert result.confusion_matrix["fn"] == 1
        assert result.details[0]["error"] is not None

    def test_details_populated(self):
        examples = [
            _make_example("t1", "good", True),
            _make_example("t2", "bad", False),
        ]

        result = calibrate_grader(_PerfectGrader(), examples)

        assert len(result.details) == 2
        assert result.details[0]["task_id"] == "t1"
        assert result.details[0]["correct"] is True
        assert result.details[1]["task_id"] == "t2"
        assert result.details[1]["correct"] is True
