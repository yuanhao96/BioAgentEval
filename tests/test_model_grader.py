import json
import pytest
from unittest.mock import MagicMock, patch

from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.model_grader import ModelGrader
from bioagenteval.graders.llm_client import LLMClient
from bioagenteval.models import (
    ExpectedOutput, GradeResult, GraderConfig, Task, Transcript,
)


class _MockLLMClient:
    """Test helper implementing LLMClient protocol."""

    def __init__(self, response: str):
        self.response = response
        self.last_messages = None
        self.last_model = None

    def complete(self, messages, *, system="", model="", max_tokens=256):
        self.last_messages = messages
        self.last_model = model
        return self.response


class TestModelGraderIsBaseGrader:
    def test_inherits_base(self):
        client = _MockLLMClient('{"score": 0.5, "passed": true, "reasoning": "ok"}')
        grader = ModelGrader(client=client)
        assert isinstance(grader, BaseGrader)


class TestModelGraderWithLLMClient:
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

    def test_passing_grade(self):
        response = json.dumps({"score": 0.9, "passed": True, "verdict": "pass", "reasoning": "Comprehensive."})
        client = _MockLLMClient(response)
        grader = ModelGrader(client=client)
        result = grader.grade(
            self.task, "INS and HLA-DRB1 are key genes.", self.transcript, self.config
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.grader_type == "model"
        assert "reasoning" in result.details

    def test_failing_grade(self):
        response = json.dumps({"score": 0.2, "passed": False, "verdict": "fail", "reasoning": "Incomplete."})
        client = _MockLLMClient(response)
        grader = ModelGrader(client=client)
        result = grader.grade(
            self.task, "I don't know.", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.2

    def test_unknown_verdict(self):
        response = json.dumps({
            "score": 0.0, "passed": False, "verdict": "unknown",
            "reasoning": "Insufficient information to evaluate."
        })
        client = _MockLLMClient(response)
        grader = ModelGrader(client=client)
        result = grader.grade(
            self.task, "ambiguous response", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.0
        assert result.details.get("status") == "unknown"
        assert "reasoning" in result.details

    def test_api_error_returns_zero(self):
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = Exception("API error")
        grader = ModelGrader(client=client)
        result = grader.grade(
            self.task, "answer", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_metrics_in_prompt(self):
        response = json.dumps({"score": 0.8, "passed": True, "reasoning": "Good."})
        client = _MockLLMClient(response)
        grader = ModelGrader(client=client)
        metrics = {"n_turns": 5, "n_tool_calls": 3}
        result = grader.grade(
            self.task, "answer", self.transcript, self.config, metrics=metrics
        )
        prompt_content = client.last_messages[0]["content"]
        assert "n_turns" in prompt_content
        assert "n_tool_calls" in prompt_content
        assert result.passed is True

    def test_model_override_from_config_params(self):
        response = json.dumps({"score": 0.7, "passed": True, "reasoning": "ok"})
        client = _MockLLMClient(response)
        grader = ModelGrader(client=client, model="default-model")
        config = GraderConfig(
            type="model",
            rubric="test",
            params={"model": "custom-model"},
        )
        grader.grade(self.task, "answer", self.transcript, config)
        assert client.last_model == "custom-model"

    def test_backward_compat_no_verdict_field(self):
        """Old-style response without verdict field should still work."""
        response = json.dumps({"score": 0.85, "passed": True, "reasoning": "Good answer."})
        client = _MockLLMClient(response)
        grader = ModelGrader(client=client)
        result = grader.grade(
            self.task, "INS and HLA-DRB1", self.transcript, self.config
        )
        assert result.passed is True
        assert result.score == 0.85
        assert result.details.get("status") is None


class TestModelGraderProviderFactory:
    @patch("openai.OpenAI")
    def test_default_openai_provider(self, mock_openai_cls):
        grader = ModelGrader(provider="openai")
        assert grader.client is not None

    @patch("anthropic.Anthropic")
    def test_anthropic_provider(self, mock_anthropic_cls):
        grader = ModelGrader(provider="anthropic")
        assert grader.client is not None
