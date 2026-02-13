import json
import pytest
from pathlib import Path

from pankeval.models import (
    EvalResult, GradeResult, Transcript, TrialResult,
)
from pankeval.reporter import EvalReporter


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
