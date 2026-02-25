import pytest
from unittest.mock import MagicMock

from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.consensus_grader import ConsensusGrader
from bioagenteval.models import (
    ExpectedOutput, GradeResult, GraderConfig, Task, Transcript,
)


class _FixedGrader(BaseGrader):
    """Test grader that returns pre-configured results in sequence."""

    def __init__(self, results: list[GradeResult]):
        self._results = list(results)
        self._call_count = 0

    def grade(self, task, outcome, transcript, config, metrics=None):
        idx = self._call_count % len(self._results)
        self._call_count += 1
        return self._results[idx]


class _FailingGrader(BaseGrader):
    """Test grader that always raises an exception."""

    def grade(self, task, outcome, transcript, config, metrics=None):
        raise RuntimeError("Judge failed")


class TestConsensusGraderIsBaseGrader:
    def test_inherits_base(self):
        inner = _FixedGrader([GradeResult(grader_type="model", score=1.0, passed=True)])
        grader = ConsensusGrader(inner_grader=inner)
        assert isinstance(grader, BaseGrader)


class TestConsensusGraderGrade:
    def setup_method(self):
        self.task = Task(
            id="t1",
            question="What genes are associated with diabetes?",
            expected_output=[
                ExpectedOutput(type="entities", value=["INS"]),
            ],
        )
        self.transcript = Transcript(task_id="t1")
        self.config = GraderConfig(type="consensus", rubric="test")

    def test_unanimous_pass(self):
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.9, passed=True),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.passed is True
        assert result.score == pytest.approx(0.9)
        assert result.grader_type == "consensus"
        assert result.details["pass_votes"] == 3
        assert result.details["fail_votes"] == 0
        assert result.details["num_successful"] == 3

    def test_unanimous_fail(self):
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.1, passed=False),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.passed is False
        assert result.score == pytest.approx(0.1)
        assert result.details["pass_votes"] == 0
        assert result.details["fail_votes"] == 3

    def test_majority_pass(self):
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.8, passed=True),
            GradeResult(grader_type="model", score=0.3, passed=False),
            GradeResult(grader_type="model", score=0.7, passed=True),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.passed is True
        assert result.details["pass_votes"] == 2
        assert result.details["fail_votes"] == 1
        # Average score: (0.8 + 0.3 + 0.7) / 3 = 0.6
        assert result.score == pytest.approx(0.6)

    def test_majority_fail(self):
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.2, passed=False),
            GradeResult(grader_type="model", score=0.8, passed=True),
            GradeResult(grader_type="model", score=0.3, passed=False),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.passed is False
        assert result.details["pass_votes"] == 1
        assert result.details["fail_votes"] == 2

    def test_all_judges_fail_with_exceptions(self):
        inner = _FailingGrader()
        grader = ConsensusGrader(inner_grader=inner, num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.passed is False
        assert result.score == 0.0
        assert result.details["num_successful"] == 0
        assert len(result.details["errors"]) == 3

    def test_partial_judge_failures(self):
        """When some judges fail, consensus is based on successful ones."""
        # Create a grader that alternates: succeed, fail, succeed
        call_count = [0]
        original_results = [
            GradeResult(grader_type="model", score=0.9, passed=True),
            GradeResult(grader_type="model", score=0.7, passed=True),
        ]

        class _AlternatingGrader(BaseGrader):
            def grade(self, task, outcome, transcript, config, metrics=None):
                idx = call_count[0]
                call_count[0] += 1
                if idx == 1:  # Second call fails
                    raise RuntimeError("Judge 2 failed")
                return original_results[0 if idx == 0 else 1]

        grader = ConsensusGrader(inner_grader=_AlternatingGrader(), num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.details["num_successful"] == 2
        assert len(result.details["errors"]) == 1
        assert result.passed is True

    def test_individual_scores_recorded(self):
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.8, passed=True),
            GradeResult(grader_type="model", score=0.6, passed=True),
            GradeResult(grader_type="model", score=0.4, passed=False),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=3)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.details["individual_scores"] == [0.8, 0.6, 0.4]
        assert result.details["individual_passed"] == [True, True, False]

    def test_custom_num_judges(self):
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.5, passed=True),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=5)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.details["num_judges"] == 5
        assert result.details["num_successful"] == 5
        assert len(result.details["individual_scores"]) == 5

    def test_even_split_tie_fails(self):
        """With even judges and equal pass/fail, result is False (strict majority)."""
        inner = _FixedGrader([
            GradeResult(grader_type="model", score=0.8, passed=True),
            GradeResult(grader_type="model", score=0.3, passed=False),
        ])
        grader = ConsensusGrader(inner_grader=inner, num_judges=2)
        result = grader.grade(self.task, "answer", self.transcript, self.config)
        assert result.passed is False
        assert result.details["pass_votes"] == 1
        assert result.details["fail_votes"] == 1
        # Score is still averaged
        assert result.score == pytest.approx(0.55)

    def test_metrics_passed_through(self):
        calls = []

        class _TrackingGrader(BaseGrader):
            def grade(self, task, outcome, transcript, config, metrics=None):
                calls.append(metrics)
                return GradeResult(grader_type="model", score=0.8, passed=True)

        grader = ConsensusGrader(inner_grader=_TrackingGrader(), num_judges=2)
        metrics = {"n_turns": 3}
        grader.grade(self.task, "answer", self.transcript, self.config, metrics=metrics)
        assert all(c == {"n_turns": 3} for c in calls)
