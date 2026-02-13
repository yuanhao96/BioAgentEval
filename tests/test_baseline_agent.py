import pytest
from unittest.mock import MagicMock, patch

from pankeval.agents.baseline_qa import BaselineQAAgent
from pankeval.harness import AgentHarness
from pankeval.models import AgentResponse, Transcript


def _mock_openai_response(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 50
    return mock_resp


class TestBaselineQAAgent:
    @patch("pankeval.agents.baseline_qa.openai")
    def test_satisfies_harness_protocol(self, mock_openai_mod):
        agent = BaselineQAAgent()
        assert isinstance(agent, AgentHarness)

    @patch("pankeval.agents.baseline_qa.openai")
    def test_run_returns_agent_response(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "INS gene encodes insulin."
        )
        agent = BaselineQAAgent()
        resp = agent.run("What is the INS gene?")
        assert isinstance(resp, AgentResponse)
        assert resp.outcome == "INS gene encodes insulin."
        assert resp.transcript.task_id == "baseline"

    @patch("pankeval.agents.baseline_qa.openai")
    def test_transcript_captures_events(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "Answer here."
        )
        agent = BaselineQAAgent()
        resp = agent.run("Q?")
        events = resp.transcript.events
        assert len(events) >= 1
        assert events[0].event_type == "llm_call"
        assert "question" in events[0].data

    @patch("pankeval.agents.baseline_qa.openai")
    def test_reset_clears_state(self, mock_openai_mod):
        agent = BaselineQAAgent()
        agent.reset()  # Should not raise

    @patch("pankeval.agents.baseline_qa.openai")
    def test_custom_model(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response("A")
        agent = BaselineQAAgent(model="gpt-4o")
        agent.run("Q?")
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
