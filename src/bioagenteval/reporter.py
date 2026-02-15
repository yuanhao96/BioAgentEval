"""Structured JSON report generation for evaluation results."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bioagenteval.models import EvalResult


class EvalReporter:
    """Generates structured JSON reports from evaluation results."""

    @staticmethod
    def generate_report(
        suite_name: str,
        results: list[EvalResult],
    ) -> dict[str, Any]:
        """Generate a full evaluation report as a dict."""
        task_reports = []
        pass_at_1_values = []

        for result in results:
            p1 = result.pass_at_k(k=1)
            pass_at_1_values.append(p1)

            grader_types = {
                g.grader_type for t in result.trials for g in t.grades
            }
            mean_scores = {
                gt: result.mean_score(gt) for gt in sorted(grader_types)
            }

            trial_dicts = []
            for trial in result.trials:
                trial_dicts.append({
                    "trial_num": trial.trial_num,
                    "outcome": trial.outcome,
                    "grades": [g.model_dump() for g in trial.grades],
                    "transcript": trial.transcript.model_dump(mode="json"),
                    "duration_ms": trial.duration_ms,
                    "error": trial.error,
                    "metrics": trial.metrics,
                })

            task_reports.append({
                "task_id": result.task_id,
                "pass_at_1": p1,
                "mean_scores": mean_scores,
                "num_trials": len(result.trials),
                "trials": trial_dicts,
            })

        overall_pass_at_1 = (
            sum(pass_at_1_values) / len(pass_at_1_values)
            if pass_at_1_values
            else 0.0
        )

        return {
            "suite_name": suite_name,
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": task_reports,
            "summary": {
                "total_tasks": len(results),
                "overall_pass_at_1": overall_pass_at_1,
            },
        }

    @staticmethod
    def save_report(
        suite_name: str,
        results: list[EvalResult],
        path: str | Path,
    ) -> None:
        """Generate and save a report to a JSON file."""
        report = EvalReporter.generate_report(suite_name, results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
