import pytest
from pankeval.harness import AgentHarness
from pankeval.models import AgentResponse, Task, Transcript


class FakeAgent:
    """A trivial agent that satisfies the AgentHarness protocol."""
    def __init__(self):
        self.call_count = 0

    def run(self, question: str) -> AgentResponse:
        self.call_count += 1
        return AgentResponse(
            outcome=f"Answer to: {question}",
            transcript=Transcript(task_id="fake"),
        )

    def reset(self) -> None:
        self.call_count = 0


class BadAgent:
    """Missing the run method."""
    def reset(self) -> None:
        pass


class TestAgentHarness:
    def test_fake_agent_satisfies_protocol(self):
        agent = FakeAgent()
        assert isinstance(agent, AgentHarness)

    def test_bad_agent_does_not_satisfy(self):
        agent = BadAgent()
        assert not isinstance(agent, AgentHarness)

    def test_fake_agent_run(self):
        agent = FakeAgent()
        resp = agent.run("What is INS?")
        assert resp.outcome == "Answer to: What is INS?"
        assert resp.transcript.task_id == "fake"
        assert agent.call_count == 1

    def test_fake_agent_reset(self):
        agent = FakeAgent()
        agent.run("Q1")
        agent.run("Q2")
        assert agent.call_count == 2
        agent.reset()
        assert agent.call_count == 0
