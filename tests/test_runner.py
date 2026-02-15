import pytest
from unittest.mock import MagicMock

from bioagenteval.models import (
    AgentResponse, EvalResult, ExpectedOutput, GradeResult, GraderConfig,
    MetricGroup, Task, Transcript, TranscriptEvent, TrialResult,
)
from bioagenteval.runner import EvalRunner


class FakeAgent:
    def __init__(self, answer: str = "Test answer"):
        self.answer = answer
        self.run_count = 0

    def run(self, question: str) -> AgentResponse:
        self.run_count += 1
        return AgentResponse(
            outcome=self.answer,
            transcript=Transcript(
                task_id="fake",
                events=[
                    TranscriptEvent(
                        event_type="llm_call",
                        data={"question": question},
                    )
                ],
            ),
        )

    def reset(self) -> None:
        pass


class FakeGrader:
    def grade(self, task, outcome, transcript, config, metrics=None):
        return GradeResult(grader_type="code", score=1.0, passed=True)


class FailingGrader:
    def grade(self, task, outcome, transcript, config, metrics=None):
        return GradeResult(grader_type="code", score=0.0, passed=False)


class TestEvalRunner:
    def test_run_single_task_single_trial(self):
        task = Task(
            id="t1",
            question="What is INS?",
            expected_output=[ExpectedOutput(type="entities", value=["INS"])],
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(),
            graders={"code": FakeGrader()},
        )
        result = runner.run_task(task)
        assert isinstance(result, EvalResult)
        assert result.task_id == "t1"
        assert len(result.trials) == 1
        assert result.trials[0].outcome == "Test answer"
        assert result.trials[0].grades[0].passed is True

    def test_run_multiple_trials(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=3,
        )
        agent = FakeAgent()
        runner = EvalRunner(agent=agent, graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert len(result.trials) == 3
        for i, trial in enumerate(result.trials):
            assert trial.trial_num == i

    def test_run_task_calls_reset_between_trials(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=3,
        )
        agent = MagicMock()
        agent.run.return_value = AgentResponse(
            outcome="answer",
            transcript=Transcript(task_id="t1"),
        )
        runner = EvalRunner(agent=agent, graders={"code": FakeGrader()})
        runner.run_task(task)
        assert agent.reset.call_count == 3

    def test_run_suite(self):
        tasks = [
            Task(id="t1", question="Q1?", graders=[GraderConfig(type="code")], num_trials=2),
            Task(id="t2", question="Q2?", graders=[GraderConfig(type="code")], num_trials=1),
        ]
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        results = runner.run_suite(tasks)
        assert len(results) == 2
        assert results[0].task_id == "t1"
        assert len(results[0].trials) == 2
        assert results[1].task_id == "t2"
        assert len(results[1].trials) == 1

    def test_duration_is_recorded(self):
        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert result.trials[0].duration_ms >= 0.0

    def test_agent_error_captured(self):
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("Agent crashed")
        agent.reset.return_value = None
        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(agent=agent, graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert len(result.trials) == 1
        assert result.trials[0].error == "Agent crashed"
        assert result.trials[0].outcome == ""

    def test_multiple_graders_per_task(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[
                GraderConfig(type="code"),
                GraderConfig(type="model", rubric="Is it good?"),
            ],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(),
            graders={
                "code": FakeGrader(),
                "model": FailingGrader(),
            },
        )
        result = runner.run_task(task)
        assert len(result.trials[0].grades) == 2

    def test_trial_metrics_populated(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            tracked_metrics=[
                MetricGroup(type="transcript", metrics=["n_turns", "n_tool_calls"]),
            ],
            num_trials=1,
        )
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        trial = result.trials[0]
        assert "n_turns" in trial.metrics
        assert "n_tool_calls" in trial.metrics
        # FakeAgent produces 1 llm_call event
        assert trial.metrics["n_turns"] == 1
        assert trial.metrics["n_tool_calls"] == 0

    def test_empty_tracked_metrics(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert result.trials[0].metrics == {}
