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


class TestPassHatK:
    def test_all_pass(self):
        trials = [
            TrialResult(
                task_id="t1", trial_num=i, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
                duration_ms=100.0,
            )
            for i in range(5)
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_hat_k(k=1) == 1.0
        assert er.pass_hat_k(k=5) == 1.0

    def test_none_pass(self):
        trials = [
            TrialResult(
                task_id="t1", trial_num=i, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=0.0, passed=False)],
                duration_ms=100.0,
            )
            for i in range(3)
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_hat_k(k=1) == 0.0

    def test_partial_pass(self):
        # 2 out of 3 pass: pass^1 = C(2,1)/C(3,1) = 2/3
        trials = [
            TrialResult(
                task_id="t1", trial_num=i, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=s, passed=s >= 0.5)],
                duration_ms=100.0,
            )
            for i, s in enumerate([1.0, 0.0, 1.0])
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_hat_k(k=1) == pytest.approx(2.0 / 3.0)
        # pass^2 = C(2,2)/C(3,2) = 1/3
        assert er.pass_hat_k(k=2) == pytest.approx(1.0 / 3.0)
        # pass^3 = C(2,3)/C(3,3) = 0
        assert er.pass_hat_k(k=3) == 0.0

    def test_edge_cases(self):
        er = EvalResult(task_id="t1", trials=[])
        assert er.pass_hat_k(k=1) == 0.0
        assert er.pass_hat_k(k=0) == 0.0

    def test_k_greater_than_n(self):
        trials = [
            TrialResult(
                task_id="t1", trial_num=0, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
                duration_ms=100.0,
            )
        ]
        er = EvalResult(task_id="t1", trials=trials)
        # k=5 but n=1, clamps to k=1, c=1: C(1,1)/C(1,1) = 1.0
        assert er.pass_hat_k(k=5) == 1.0


class TestWeightedScore:
    def test_uniform_weights(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=1.0, passed=True, weight=1.0),
                GradeResult(grader_type="model", score=0.5, passed=True, weight=1.0),
            ],
        )
        assert trial.weighted_score() == pytest.approx(0.75)

    def test_different_weights(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=1.0, passed=True, weight=3.0),
                GradeResult(grader_type="model", score=0.0, passed=False, weight=1.0),
            ],
        )
        # (1.0*3 + 0.0*1) / (3+1) = 0.75
        assert trial.weighted_score() == pytest.approx(0.75)

    def test_empty_grades(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
        )
        assert trial.weighted_score() == 0.0

    def test_weighted_passed(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=0.8, passed=True, weight=2.0),
                GradeResult(grader_type="model", score=0.2, passed=False, weight=1.0),
            ],
        )
        # (0.8*2 + 0.2*1) / 3 = 0.6
        assert trial.weighted_passed(threshold=0.5) is True
        assert trial.weighted_passed(threshold=0.7) is False

    def test_default_weight_backward_compatible(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=0.8, passed=True),
                GradeResult(grader_type="model", score=0.6, passed=True),
            ],
        )
        # Default weight=1.0, so (0.8+0.6)/2 = 0.7
        assert trial.weighted_score() == pytest.approx(0.7)


class TestUnknownGradeHandling:
    def test_weighted_score_excludes_unknown(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=1.0, passed=True, weight=1.0),
                GradeResult(grader_type="model", score=0.0, passed=False, weight=1.0,
                            details={"status": "unknown", "reasoning": "Insufficient info."}),
            ],
        )
        # Only the code grade should count: 1.0
        assert trial.weighted_score() == pytest.approx(1.0)

    def test_weighted_passed_excludes_unknown(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=0.8, passed=True, weight=1.0),
                GradeResult(grader_type="model", score=0.0, passed=False, weight=1.0,
                            details={"status": "unknown"}),
            ],
        )
        # Only code grade counts: 0.8 >= 0.5
        assert trial.weighted_passed(threshold=0.5) is True

    def test_all_unknown_returns_zero(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="model", score=0.0, passed=False, weight=1.0,
                            details={"status": "unknown"}),
                GradeResult(grader_type="model", score=0.0, passed=False, weight=1.0,
                            details={"status": "unknown"}),
            ],
        )
        assert trial.weighted_score() == 0.0
        assert trial.weighted_passed() is False

    def test_has_unknown_grades_true(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=1.0, passed=True),
                GradeResult(grader_type="model", score=0.0, passed=False,
                            details={"status": "unknown"}),
            ],
        )
        assert trial.has_unknown_grades() is True

    def test_has_unknown_grades_false(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=1.0, passed=True),
                GradeResult(grader_type="model", score=0.8, passed=True),
            ],
        )
        assert trial.has_unknown_grades() is False

    def test_no_grades_has_unknown_false(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
        )
        assert trial.has_unknown_grades() is False

    def test_mixed_known_unknown_weighted_score(self):
        trial = TrialResult(
            task_id="t1", trial_num=0, outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[
                GradeResult(grader_type="code", score=0.8, passed=True, weight=2.0),
                GradeResult(grader_type="model", score=0.0, passed=False, weight=1.0,
                            details={"status": "unknown"}),
                GradeResult(grader_type="model", score=0.6, passed=True, weight=1.0),
            ],
        )
        # Known grades: code(0.8*2) + model(0.6*1) = 2.2 / 3.0
        assert trial.weighted_score() == pytest.approx(2.2 / 3.0)


class TestConvergenceCheck:
    def test_all_pass_converged(self):
        trials = [
            TrialResult(
                task_id="t1", trial_num=i, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
                duration_ms=100.0,
            )
            for i in range(10)
        ]
        er = EvalResult(task_id="t1", trials=trials)
        conv = er.convergence_check()
        assert conv["pass_rate"] == 1.0
        assert conv["converged"] is True
        assert conv["n_trials"] == 10

    def test_mixed_results(self):
        trials = [
            TrialResult(
                task_id="t1", trial_num=i, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=s, passed=s >= 0.5)],
                duration_ms=100.0,
            )
            for i, s in enumerate([1.0, 0.0, 1.0, 0.0, 1.0])
        ]
        er = EvalResult(task_id="t1", trials=trials)
        conv = er.convergence_check()
        assert conv["pass_rate"] == pytest.approx(0.6)
        assert 0.0 <= conv["ci_lower"] <= conv["ci_upper"] <= 1.0
        assert conv["ci_width"] > 0

    def test_empty_trials(self):
        er = EvalResult(task_id="t1", trials=[])
        conv = er.convergence_check()
        assert conv["converged"] is False
        assert conv["n_trials"] == 0

    def test_single_trial(self):
        trials = [
            TrialResult(
                task_id="t1", trial_num=0, outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
                duration_ms=100.0,
            )
        ]
        er = EvalResult(task_id="t1", trials=trials)
        conv = er.convergence_check()
        assert conv["n_trials"] == 1
        # Wide CI with single trial
        assert conv["ci_width"] > 0


class TestShouldFail:
    def test_should_fail_default(self):
        t = Task(id="t1", question="Q?")
        assert t.should_fail is False

    def test_should_fail_set(self):
        t = Task(id="t1", question="Q?", should_fail=True)
        assert t.should_fail is True


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

    def test_suite_eval_type(self):
        s = EvalSuite(name="core", eval_type="regression")
        assert s.eval_type == "regression"

    def test_suite_eval_type_default(self):
        s = EvalSuite(name="core")
        assert s.eval_type == ""

    def test_suite_eval_type_capability(self):
        s = EvalSuite(name="core", eval_type="capability")
        assert s.eval_type == "capability"

    def test_suite_eval_type_invalid_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvalSuite(name="core", eval_type="invalid_type")

    def test_suite_with_default_tracked_metrics(self):
        s = EvalSuite(
            name="core",
            default_tracked_metrics=[
                MetricGroup(type="transcript", metrics=["n_turns"]),
            ],
        )
        assert len(s.default_tracked_metrics) == 1
        assert s.default_tracked_metrics[0].type == "transcript"
