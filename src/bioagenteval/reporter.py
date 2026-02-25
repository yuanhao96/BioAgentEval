"""Structured JSON report generation for evaluation results."""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bioagenteval.models import EvalResult


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a sorted list of values."""
    if not sorted_values:
        return 0.0
    k = (p / 100) * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


class EvalReporter:
    """Generates structured JSON reports from evaluation results."""

    SATURATION_THRESHOLD = 0.95

    @staticmethod
    def generate_report(
        suite_name: str,
        results: list[EvalResult],
        eval_type: str = "",
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
                "pass_hat_1": result.pass_hat_k(k=1),
                "mean_scores": mean_scores,
                "num_trials": len(result.trials),
                "convergence": result.convergence_check(),
                "trials": trial_dicts,
            })

        overall_pass_at_1 = (
            sum(pass_at_1_values) / len(pass_at_1_values)
            if pass_at_1_values
            else 0.0
        )

        pass_hat_1_values = [r.pass_hat_k(k=1) for r in results]
        overall_pass_hat_1 = (
            sum(pass_hat_1_values) / len(pass_hat_1_values)
            if pass_hat_1_values
            else 0.0
        )

        all_durations = sorted(
            t.duration_ms
            for r in results
            for t in r.trials
        )

        summary: dict[str, Any] = {
            "total_tasks": len(results),
            "overall_pass_at_1": overall_pass_at_1,
            "overall_pass_hat_1": overall_pass_hat_1,
        }
        if all_durations:
            summary["latency_p50_ms"] = _percentile(all_durations, 50)
            summary["latency_p95_ms"] = _percentile(all_durations, 95)

        # Saturation detection
        saturated = overall_pass_at_1 >= EvalReporter.SATURATION_THRESHOLD
        summary["saturated"] = saturated
        if saturated:
            summary["saturation_note"] = (
                f"Suite pass@1 ({overall_pass_at_1:.0%}) is near 100%. "
                "Consider promoting to a regression suite and creating "
                "harder diagnostic tasks."
            )

        report: dict[str, Any] = {
            "suite_name": suite_name,
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": task_reports,
            "summary": summary,
        }
        if eval_type:
            report["eval_type"] = eval_type

        return report

    @staticmethod
    def save_report(
        suite_name: str,
        results: list[EvalResult],
        path: str | Path,
        eval_type: str = "",
    ) -> None:
        """Generate and save a report to a JSON file."""
        report = EvalReporter.generate_report(suite_name, results, eval_type=eval_type)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
