import json
import pytest
from pathlib import Path

from bioagenteval.models import (
    EvalResult, GradeResult, Transcript, TrialResult,
)
from bioagenteval.reporter import EvalReporter


def _make_result(task_id: str, scores: list[float]) -> EvalResult:
    trials = [
        TrialResult(
            task_id=task_id,
            trial_num=i,
            outcome=f"answer_{i}",
            transcript=Transcript(task_id=task_id),
            grades=[GradeResult(grader_type="code", score=s, passed=s >= 0.5)],
            duration_ms=100.0 * (i + 1),
        )
        for i, s in enumerate(scores)
    ]
    return EvalResult(task_id=task_id, trials=trials)


class TestEvalReporter:
    def test_generate_report_structure(self):
        results = [
            _make_result("t1", [1.0, 0.0, 1.0]),
            _make_result("t2", [0.0]),
        ]
        report = EvalReporter.generate_report("test_suite", results)
        assert report["suite_name"] == "test_suite"
        assert "run_id" in report
        assert "timestamp" in report
        assert len(report["results"]) == 2
        assert "summary" in report

    def test_per_task_metrics(self):
        results = [_make_result("t1", [1.0, 0.0, 1.0])]
        report = EvalReporter.generate_report("s", results)
        task_report = report["results"][0]
        assert task_report["task_id"] == "t1"
        assert "pass_at_1" in task_report
        assert "mean_scores" in task_report
        assert task_report["num_trials"] == 3
        assert len(task_report["trials"]) == 3

    def test_summary_aggregation(self):
        results = [
            _make_result("t1", [1.0, 1.0]),
            _make_result("t2", [0.0, 0.0]),
        ]
        report = EvalReporter.generate_report("s", results)
        summary = report["summary"]
        assert summary["total_tasks"] == 2
        assert 0.0 <= summary["overall_pass_at_1"] <= 1.0

    def test_save_report(self, tmp_path):
        results = [_make_result("t1", [1.0])]
        out_file = tmp_path / "report.json"
        EvalReporter.save_report("s", results, out_file)
        assert out_file.exists()
        loaded = json.loads(out_file.read_text())
        assert loaded["suite_name"] == "s"

    def test_trials_include_outcome_and_grades(self):
        results = [_make_result("t1", [0.8])]
        report = EvalReporter.generate_report("s", results)
        trial = report["results"][0]["trials"][0]
        assert trial["outcome"] == "answer_0"
        assert len(trial["grades"]) == 1
        assert trial["grades"][0]["score"] == 0.8

    def test_summary_includes_pass_hat_1(self):
        results = [
            _make_result("t1", [1.0, 1.0]),
            _make_result("t2", [0.0, 0.0]),
        ]
        report = EvalReporter.generate_report("s", results)
        summary = report["summary"]
        assert "overall_pass_hat_1" in summary
        # t1: pass^1 = 1.0, t2: pass^1 = 0.0, avg = 0.5
        assert summary["overall_pass_hat_1"] == pytest.approx(0.5)

    def test_summary_includes_latency_percentiles(self):
        results = [_make_result("t1", [1.0, 0.5, 1.0])]
        report = EvalReporter.generate_report("s", results)
        summary = report["summary"]
        assert "latency_p50_ms" in summary
        assert "latency_p95_ms" in summary
        # _make_result gives durations 100, 200, 300
        assert summary["latency_p50_ms"] == pytest.approx(200.0)

    def test_per_task_includes_pass_hat_1(self):
        results = [_make_result("t1", [1.0, 0.0, 1.0])]
        report = EvalReporter.generate_report("s", results)
        task_report = report["results"][0]
        assert "pass_hat_1" in task_report
        # 2/3 pass: pass^1 = C(2,1)/C(3,1) = 2/3
        assert task_report["pass_hat_1"] == pytest.approx(2.0 / 3.0)

    def test_saturation_detected(self):
        # All tasks pass → overall_pass_at_1 = 1.0 → saturated
        results = [_make_result("t1", [1.0, 1.0]), _make_result("t2", [1.0, 1.0])]
        report = EvalReporter.generate_report("s", results)
        assert report["summary"]["saturated"] is True
        assert "saturation_note" in report["summary"]
        assert "regression" in report["summary"]["saturation_note"]

    def test_not_saturated(self):
        results = [_make_result("t1", [1.0, 0.0]), _make_result("t2", [0.0, 0.0])]
        report = EvalReporter.generate_report("s", results)
        assert report["summary"]["saturated"] is False
        assert "saturation_note" not in report["summary"]

    def test_eval_type_in_report(self):
        results = [_make_result("t1", [1.0])]
        report = EvalReporter.generate_report("s", results, eval_type="regression")
        assert report["eval_type"] == "regression"

    def test_convergence_in_task_report(self):
        results = [_make_result("t1", [1.0, 0.0, 1.0])]
        report = EvalReporter.generate_report("s", results)
        task_report = report["results"][0]
        assert "convergence" in task_report
        conv = task_report["convergence"]
        assert "pass_rate" in conv
        assert "ci_width" in conv
        assert "converged" in conv

    def test_eval_type_absent_when_empty(self):
        results = [_make_result("t1", [1.0])]
        report = EvalReporter.generate_report("s", results)
        assert "eval_type" not in report

    def test_trials_include_metrics(self):
        trial = TrialResult(
            task_id="t1",
            trial_num=0,
            outcome="answer",
            transcript=Transcript(task_id="t1"),
            grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
            duration_ms=500.0,
            metrics={"n_turns": 3, "n_tool_calls": 2},
        )
        result = EvalResult(task_id="t1", trials=[trial])
        report = EvalReporter.generate_report("s", [result])
        trial_dict = report["results"][0]["trials"][0]
        assert "metrics" in trial_dict
        assert trial_dict["metrics"]["n_turns"] == 3
        assert trial_dict["metrics"]["n_tool_calls"] == 2
