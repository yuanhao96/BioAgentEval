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

    def test_weight_propagated_from_config(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[
                GraderConfig(type="code", weight=2.5),
            ],
            num_trials=1,
        )
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert result.trials[0].grades[0].weight == 2.5

    def test_default_weight_propagated(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert result.trials[0].grades[0].weight == 1.0

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

    def test_should_fail_inverts_grades(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
            should_fail=True,
        )
        # FakeGrader returns score=1.0, passed=True
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        # Inverted: score=0.0, passed=False
        assert result.trials[0].grades[0].score == 0.0
        assert result.trials[0].grades[0].passed is False

    def test_should_fail_inverts_failing_to_passing(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
            should_fail=True,
        )
        # FailingGrader returns score=0.0, passed=False
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FailingGrader()})
        result = runner.run_task(task)
        # Inverted: score=1.0, passed=True
        assert result.trials[0].grades[0].score == 1.0
        assert result.trials[0].grades[0].passed is True

    def test_max_concurrency_parallel(self):
        tasks = [
            Task(id=f"t{i}", question="Q?", graders=[GraderConfig(type="code")], num_trials=1)
            for i in range(5)
        ]
        runner = EvalRunner(
            agent=FakeAgent(), graders={"code": FakeGrader()}, max_concurrency=3,
        )
        results = runner.run_suite(tasks)
        assert len(results) == 5
        # Order preserved
        for i, r in enumerate(results):
            assert r.task_id == f"t{i}"

    def test_retry_on_grader_failure(self):
        call_count = 0

        class FailOnceGrader:
            def grade(self, task, outcome, transcript, config, metrics=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("API timeout")
                return GradeResult(grader_type="code", score=1.0, passed=True)

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(), graders={"code": FailOnceGrader()}, max_retries=1,
        )
        result = runner.run_task(task)
        assert result.trials[0].grades[0].passed is True
        assert call_count == 2

    def test_retry_exhausted(self):
        class AlwaysFailGrader:
            def grade(self, task, outcome, transcript, config, metrics=None):
                raise RuntimeError("Permanent failure")

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(), graders={"code": AlwaysFailGrader()}, max_retries=1,
        )
        result = runner.run_task(task)
        assert result.trials[0].grades[0].passed is False
        assert "error" in result.trials[0].grades[0].details

    def test_hooks_setup_and_teardown_called(self):
        calls = []

        class TrackingHook:
            def setup(self, task):
                calls.append(("setup", task.id))

            def teardown(self, task, trial):
                calls.append(("teardown", task.id))

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=2,
        )
        runner = EvalRunner(
            agent=FakeAgent(), graders={"code": FakeGrader()},
            hooks=[TrackingHook()],
        )
        runner.run_task(task)

        assert calls == [
            ("setup", "t1"), ("teardown", "t1"),
            ("setup", "t1"), ("teardown", "t1"),
        ]

    def test_hook_error_does_not_break_trial(self):
        class FailingHook:
            def setup(self, task):
                raise RuntimeError("Setup boom")

            def teardown(self, task, trial):
                raise RuntimeError("Teardown boom")

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(), graders={"code": FakeGrader()},
            hooks=[FailingHook()],
        )
        result = runner.run_task(task)

        # Trial should still succeed despite hook failures
        assert len(result.trials) == 1
        assert result.trials[0].grades[0].passed is True

    def test_multiple_hooks_all_called(self):
        calls = []

        class HookA:
            def setup(self, task):
                calls.append("A_setup")
            def teardown(self, task, trial):
                calls.append("A_teardown")

        class HookB:
            def setup(self, task):
                calls.append("B_setup")
            def teardown(self, task, trial):
                calls.append("B_teardown")

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(), graders={"code": FakeGrader()},
            hooks=[HookA(), HookB()],
        )
        runner.run_task(task)

        assert calls == ["A_setup", "B_setup", "A_teardown", "B_teardown"]

    def test_trial_timeout(self):
        import time as _time

        class SlowAgent:
            def run(self, question):
                _time.sleep(5)
                return AgentResponse(
                    outcome="late", transcript=Transcript(task_id="t1"),
                )
            def reset(self):
                pass

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
            metadata={"timeout": 0.1},
        )
        runner = EvalRunner(agent=SlowAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)

        assert result.trials[0].error is not None
        assert "timed out" in result.trials[0].error

    def test_default_timeout(self):
        import time as _time

        class SlowAgent:
            def run(self, question):
                _time.sleep(5)
                return AgentResponse(
                    outcome="late", transcript=Transcript(task_id="t1"),
                )
            def reset(self):
                pass

        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=SlowAgent(), graders={"code": FakeGrader()},
            default_timeout=0.1,
        )
        result = runner.run_task(task)

        assert result.trials[0].error is not None
        assert "timed out" in result.trials[0].error
