import json
import pytest
from unittest.mock import MagicMock, patch

from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.model_grader import ModelGrader
from bioagenteval.models import (
    ExpectedOutput, GradeResult, GraderConfig, Task, Transcript,
)


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
        with patch("bioagenteval.graders.model_grader.OpenAI"):
            grader = ModelGrader()
        assert isinstance(grader, BaseGrader)


class TestModelGraderGrade:
    def setup_method(self):
        self.task = Task(
            id="t1",
            question="What genes are associated with diabetes?",
            expected_output=[
                ExpectedOutput(type="entities", value=["INS", "HLA-DRB1"]),
            ],
            graders=[
                GraderConfig(
                    type="model",
                    rubric="Is the answer complete and accurate?",
                )
            ],
        )
        self.transcript = Transcript(task_id="t1")
        self.config = self.task.graders[0]

    @patch("bioagenteval.graders.model_grader.OpenAI")
    def test_passing_grade(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({"score": 0.9, "passed": True, "reasoning": "Comprehensive."})
            ))]
        )
        grader = ModelGrader()
        result = grader.grade(
            self.task, "INS and HLA-DRB1 are key genes.", self.transcript, self.config
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.grader_type == "model"
        assert "reasoning" in result.details

    @patch("bioagenteval.graders.model_grader.OpenAI")
    def test_failing_grade(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({"score": 0.2, "passed": False, "reasoning": "Incomplete."})
            ))]
        )
        grader = ModelGrader()
        result = grader.grade(
            self.task, "I don't know.", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.2

    @patch("bioagenteval.graders.model_grader.OpenAI")
    def test_api_error_returns_zero(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        grader = ModelGrader()
        result = grader.grade(
            self.task, "answer", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    @patch("bioagenteval.graders.model_grader.OpenAI")
    def test_metrics_in_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({"score": 0.8, "passed": True, "reasoning": "Good."})
            ))]
        )
        grader = ModelGrader()
        metrics = {"n_turns": 5, "n_tool_calls": 3}
        result = grader.grade(
            self.task, "answer", self.transcript, self.config, metrics=metrics
        )
        # Verify that the prompt sent to the model includes metrics
        call_args = mock_client.chat.completions.create.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "n_turns" in prompt_content
        assert "n_tool_calls" in prompt_content
        assert result.passed is True
