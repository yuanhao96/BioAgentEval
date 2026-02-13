import pytest
from bioagenteval.models import (
    Task, GraderConfig, EvalSuite,
    TranscriptEvent, Transcript,
    GradeResult, TrialResult, EvalResult,
    AgentResponse,
)


class TestGraderConfig:
    def test_defaults(self):
        gc = GraderConfig(type="code")
        assert gc.type == "code"
        assert gc.checks == []
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


class TestTask:
    def test_minimal_task(self):
        t = Task(
            id="t1",
            question="What genes are associated with type 1 diabetes?",
        )
        assert t.id == "t1"
        assert t.question == "What genes are associated with type 1 diabetes?"
        assert t.expected_entities == []
        assert t.graders == []
        assert t.metadata == {}
        assert t.num_trials == 1

    def test_full_task(self):
        t = Task(
            id="t2",
            question="Tell me about INS gene",
            expected_entities=["INS", "ENSG00000254647"],
            expected_complexity="simple",
            graders=[
                GraderConfig(type="code", checks=["entity_presence"]),
                GraderConfig(type="model", rubric="Is the answer complete?"),
            ],
            metadata={"category": "entity_overview"},
            num_trials=3,
        )
        assert len(t.graders) == 2
        assert t.graders[0].type == "code"
        assert t.num_trials == 3

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
