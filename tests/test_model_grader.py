import json
import pytest
from unittest.mock import MagicMock, patch

from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.model_grader import ModelGrader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript


def _mock_anthropic_response(score: float, passed: bool, reasoning: str):
    content_text = json.dumps({
        "score": score,
        "passed": passed,
        "reasoning": reasoning,
    })
    mock_block = MagicMock()
    mock_block.text = content_text
    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


class TestModelGraderIsBaseGrader:
    def test_inherits_base(self):
        with patch("bioagenteval.graders.model_grader.anthropic"):
            grader = ModelGrader()
        assert isinstance(grader, BaseGrader)


class TestModelGraderGrade:
    def setup_method(self):
        self.task = Task(
            id="t1",
            question="What genes are associated with diabetes?",
            expected_entities=["INS", "HLA-DRB1"],
            graders=[
                GraderConfig(
                    type="model",
                    rubric="Is the answer complete and accurate?",
                )
            ],
        )
        self.transcript = Transcript(task_id="t1")
        self.config = self.task.graders[0]

    @patch("bioagenteval.graders.model_grader.anthropic")
    def test_passing_grade(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response(
            0.9, True, "The answer is comprehensive."
        )
        grader = ModelGrader()
        result = grader.grade(
            self.task, "INS and HLA-DRB1 are key genes.", self.transcript, self.config
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.grader_type == "model"
        assert "reasoning" in result.details

    @patch("bioagenteval.graders.model_grader.anthropic")
    def test_failing_grade(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response(
            0.2, False, "The answer is incomplete."
        )
        grader = ModelGrader()
        result = grader.grade(
            self.task, "I don't know.", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.2

    @patch("bioagenteval.graders.model_grader.anthropic")
    def test_api_error_returns_zero(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")
        grader = ModelGrader()
        result = grader.grade(
            self.task, "answer", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    @patch("bioagenteval.graders.model_grader.anthropic")
    def test_custom_model(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response(
            0.8, True, "Good."
        )
        grader = ModelGrader(model="claude-opus-4-6")
        result = grader.grade(
            self.task, "answer", self.transcript, self.config
        )
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"
