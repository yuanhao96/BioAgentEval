import pytest
from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.human_grader import HumanGrader
from bioagenteval.models import GraderConfig, Task, Transcript


class TestHumanGrader:
    def test_inherits_base(self):
        grader = HumanGrader()
        assert isinstance(grader, BaseGrader)

    def test_returns_pending_result(self):
        task = Task(id="t1", question="Q?")
        config = GraderConfig(type="human")
        result = HumanGrader().grade(
            task, "some answer", Transcript(task_id="t1"), config
        )
        assert result.grader_type == "human"
        assert result.passed is False
        assert result.score == 0.0
        assert "pending" in result.details.get("status", "")
