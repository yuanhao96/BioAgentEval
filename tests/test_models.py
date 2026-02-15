import pytest
from bioagenteval.models import (
    Task, GraderConfig, EvalSuite, ExpectedOutput, MetricGroup,
    TranscriptEvent, Transcript,
    GradeResult, TrialResult, EvalResult,
    AgentResponse,
)


class TestGraderConfig:
    def test_defaults(self):
        gc = GraderConfig(type="code")
        assert gc.type == "code"
        assert gc.rubric == ""
        assert gc.weight == 1.0
        assert gc.params == {}

    def test_full_config(self):
        gc = GraderConfig(
            type="model",
            rubric="Is the answer complete?",
            weight=0.5,
            params={"model": "claude-sonnet-4-5-20250929"},
        )
        assert gc.weight == 0.5


class TestExpectedOutput:
    def test_minimal(self):
        eo = ExpectedOutput(type="entities", value=["INS", "HLA-DRB1"])
        assert eo.type == "entities"
        assert eo.value == ["INS", "HLA-DRB1"]
        assert eo.params == {}

    def test_with_params(self):
        eo = ExpectedOutput(
            type="numeric_range",
            value={"target": 42, "min": 40, "max": 45},
            params={"tolerance": 0.1},
        )
        assert eo.type == "numeric_range"
        assert eo.params["tolerance"] == 0.1

    def test_mcq_answer(self):
        eo = ExpectedOutput(type="mcq_answer", value="B")
        assert eo.type == "mcq_answer"
        assert eo.value == "B"

    def test_cypher_patterns(self):
        eo = ExpectedOutput(type="cypher_patterns", value=["MATCH.*Gene"])
        assert eo.value == ["MATCH.*Gene"]


class TestMetricGroup:
    def test_minimal(self):
        mg = MetricGroup(type="transcript")
        assert mg.type == "transcript"
        assert mg.metrics == []

    def test_with_metrics(self):
        mg = MetricGroup(type="transcript", metrics=["n_turns", "n_tool_calls"])
        assert len(mg.metrics) == 2
        assert "n_turns" in mg.metrics


class TestTask:
    def test_minimal_task(self):
        t = Task(
            id="t1",
            question="What genes are associated with type 1 diabetes?",
        )
        assert t.id == "t1"
        assert t.question == "What genes are associated with type 1 diabetes?"
        assert t.expected_output == []
        assert t.tags == {}
        assert t.tracked_metrics == []
        assert t.graders == []
        assert t.metadata == {}
        assert t.num_trials == 1

    def test_full_task(self):
        t = Task(
            id="t2",
            question="Tell me about INS gene",
            expected_output=[
                ExpectedOutput(type="entities", value=["INS", "ENSG00000254647"]),
            ],
            tags={"complexity": "simple"},
            tracked_metrics=[
                MetricGroup(type="transcript", metrics=["n_turns"]),
            ],
            graders=[
                GraderConfig(type="code"),
                GraderConfig(type="model", rubric="Is the answer complete?"),
            ],
            metadata={"category": "entity_overview"},
            num_trials=3,
        )
        assert len(t.graders) == 2
        assert t.graders[0].type == "code"
        assert t.num_trials == 3
        assert len(t.expected_output) == 1
        assert t.expected_output[0].type == "entities"
        assert t.tags["complexity"] == "simple"

    def test_task_requires_question(self):
        with pytest.raises(Exception):
            Task(id="t3")


class TestTranscript:
    def test_empty_transcript(self):
        tr = Transcript(task_id="t1")
        assert tr.events == []
        assert tr.task_id == "t1"

    def test_add_events(self):
        tr = Transcript(task_id="t1")
        ev = TranscriptEvent(
            event_type="stream_event",
            event_name="complexity_classified",
            data={"complexity": "simple"},
        )
        tr.events.append(ev)
        assert len(tr.events) == 1
        assert tr.events[0].event_name == "complexity_classified"

    def test_transcript_records_cypher_queries(self):
        tr = Transcript(task_id="t1")
        ev = TranscriptEvent(
            event_type="cypher_query",
            data={"query": "MATCH (g:Gene) RETURN g LIMIT 5"},
        )
        tr.events.append(ev)
        assert tr.events[0].data["query"].startswith("MATCH")


class TestGradeResult:
    def test_grade_result(self):
        g = GradeResult(
            grader_type="code",
            score=0.8,
            passed=True,
            details={"entity_presence": True},
        )
        assert g.score == 0.8
        assert g.passed is True

    def test_score_bounds(self):
        g = GradeResult(grader_type="code", score=0.0, passed=False)
        assert g.score == 0.0
        g2 = GradeResult(grader_type="code", score=1.0, passed=True)
        assert g2.score == 1.0


class TestTrialResult:
    def test_trial_result(self):
        tr = TrialResult(
            task_id="t1",
            trial_num=0,
            outcome="Some response text",
            transcript=Transcript(task_id="t1"),
            grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
            duration_ms=1234.5,
        )
        assert tr.trial_num == 0
        assert tr.duration_ms == 1234.5
        assert tr.metrics == {}

    def test_trial_result_with_metrics(self):
        tr = TrialResult(
            task_id="t1",
            trial_num=0,
            outcome="answer",
            transcript=Transcript(task_id="t1"),
            metrics={"n_turns": 3, "n_tool_calls": 5},
        )
        assert tr.metrics["n_turns"] == 3
        assert tr.metrics["n_tool_calls"] == 5


class TestEvalResult:
    def test_eval_result_aggregation(self):
        trials = [
            TrialResult(
                task_id="t1",
                trial_num=i,
                outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=s, passed=s >= 0.5)],
                duration_ms=1000.0,
            )
            for i, s in enumerate([1.0, 0.0, 1.0])
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_at_k(k=1) > 0.0
        assert er.pass_at_k(k=3) > 0.0
        assert er.mean_score("code") == pytest.approx(2.0 / 3.0)

    def test_pass_at_k_edge_cases(self):
        er = EvalResult(task_id="t1", trials=[])
        assert er.pass_at_k(k=1) == 0.0
        assert er.pass_at_k(k=0) == 0.0

    def test_pass_at_k_all_pass(self):
        trials = [
            TrialResult(
                task_id="t1",
                trial_num=i,
                outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
                duration_ms=100.0,
            )
            for i in range(5)
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_at_k(k=1) == 1.0

    def test_mean_score_missing_grader(self):
        er = EvalResult(task_id="t1", trials=[])
        assert er.mean_score("nonexistent") == 0.0


class TestAgentResponse:
    def test_agent_response(self):
        resp = AgentResponse(
            outcome="INS gene is associated with diabetes",
            transcript=Transcript(task_id="t1"),
        )
        assert resp.outcome == "INS gene is associated with diabetes"
        assert resp.transcript.task_id == "t1"


class TestEvalSuite:
    def test_suite(self):
        s = EvalSuite(
            name="core",
            description="Core capability tests",
            task_ids=["t1", "t2"],
        )
        assert len(s.task_ids) == 2
        assert s.default_tracked_metrics == []

    def test_suite_with_default_tracked_metrics(self):
        s = EvalSuite(
            name="core",
            default_tracked_metrics=[
                MetricGroup(type="transcript", metrics=["n_turns"]),
            ],
        )
        assert len(s.default_tracked_metrics) == 1
        assert s.default_tracked_metrics[0].type == "transcript"
