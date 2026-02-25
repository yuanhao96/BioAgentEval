import json
import pytest
from unittest.mock import MagicMock

from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.pairwise_grader import PairwiseGrader
from bioagenteval.graders.llm_client import LLMClient
from bioagenteval.models import (
    ExpectedOutput, GraderConfig, Task, Transcript,
)


class _MockLLMClient:
    def __init__(self, response: str):
        self.response = response

    def complete(self, messages, *, system="", model="", max_tokens=256):
        return self.response


class TestPairwiseGraderIsBaseGrader:
    def test_inherits_base(self):
        client = _MockLLMClient('{}')
        grader = PairwiseGrader(client=client)
        assert isinstance(grader, BaseGrader)


class TestPairwiseGraderGrade:
    def setup_method(self):
        self.task = Task(
            id="t1",
            question="What genes are associated with diabetes?",
            expected_output=[
                ExpectedOutput(type="entities", value=["INS", "HLA-DRB1"]),
            ],
        )
        self.transcript = Transcript(task_id="t1")

    def test_prefers_a(self):
        response = json.dumps({
            "preferred": "A", "score": 0.9, "passed": True,
            "reasoning": "Response A is more comprehensive."
        })
        client = _MockLLMClient(response)
        grader = PairwiseGrader(client=client)
        config = GraderConfig(
            type="pairwise",
            rubric="Which is better?",
            params={"reference_outcome": "Some reference answer"},
        )
        result = grader.grade(self.task, "Agent A answer", self.transcript, config)
        assert result.passed is True
        assert result.score == 0.9
        assert result.grader_type == "pairwise"
        assert result.details["preferred"] == "A"

    def test_prefers_b(self):
        response = json.dumps({
            "preferred": "B", "score": 0.1, "passed": False,
            "reasoning": "Response B is superior."
        })
        client = _MockLLMClient(response)
        grader = PairwiseGrader(client=client)
        config = GraderConfig(
            type="pairwise",
            rubric="Which is better?",
            params={"reference_outcome": "Better reference"},
        )
        result = grader.grade(self.task, "Weak answer", self.transcript, config)
        assert result.passed is False
        assert result.score == 0.1
        assert result.details["preferred"] == "B"

    def test_tie(self):
        response = json.dumps({
            "preferred": "tie", "score": 0.5, "passed": True,
            "reasoning": "Both are equally good."
        })
        client = _MockLLMClient(response)
        grader = PairwiseGrader(client=client)
        config = GraderConfig(
            type="pairwise",
            params={"reference_outcome": "Similar answer"},
        )
        result = grader.grade(self.task, "Agent answer", self.transcript, config)
        assert result.score == 0.5
        assert result.details["preferred"] == "tie"

    def test_missing_reference_outcome(self):
        client = _MockLLMClient('{}')
        grader = PairwiseGrader(client=client)
        config = GraderConfig(type="pairwise")
        result = grader.grade(self.task, "Agent answer", self.transcript, config)
        assert result.passed is False
        assert result.score == 0.0
        assert "missing reference_outcome" in result.details.get("error", "")

    def test_api_error(self):
        client = MagicMock(spec=LLMClient)
        client.complete.side_effect = Exception("API timeout")
        grader = PairwiseGrader(client=client)
        config = GraderConfig(
            type="pairwise",
            params={"reference_outcome": "ref"},
        )
        result = grader.grade(self.task, "answer", self.transcript, config)
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_malformed_json_response(self):
        client = _MockLLMClient("This is not valid JSON {broken")
        grader = PairwiseGrader(client=client)
        config = GraderConfig(
            type="pairwise",
            params={"reference_outcome": "ref answer"},
        )
        result = grader.grade(self.task, "answer", self.transcript, config)
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    def test_model_override(self):
        response = json.dumps({
            "preferred": "A", "score": 0.8, "passed": True, "reasoning": "ok"
        })
        client = _MockLLMClient(response)
        # Track what model was used
        calls = []
        original_complete = client.complete
        def tracking_complete(messages, *, system="", model="", max_tokens=256):
            calls.append(model)
            return original_complete(messages, system=system, model=model, max_tokens=max_tokens)
        client.complete = tracking_complete

        grader = PairwiseGrader(client=client, model="default-model")
        config = GraderConfig(
            type="pairwise",
            params={"reference_outcome": "ref", "model": "custom-model"},
        )
        grader.grade(self.task, "answer", self.transcript, config)
        assert calls[0] == "custom-model"
