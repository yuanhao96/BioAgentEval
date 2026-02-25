"""Suite lifecycle management: promote, generate tasks from failures, balance check."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def promote_suite(suite_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Promote a capability suite to regression if it is saturated.

    Reads the report JSON to check saturation (overall pass@1 >= 95%),
    then updates the YAML eval_type from "capability" to "regression".

    Returns a status dict with keys: promoted (bool), reason (str),
    and optionally old_eval_type / new_eval_type.
    """
    suite_path = Path(suite_path)
    report_path = Path(report_path)

    # Load suite YAML
    with open(suite_path) as f:
        suite_data = yaml.safe_load(f)

    eval_type = suite_data.get("eval_type", "")

    if eval_type != "capability":
        return {
            "promoted": False,
            "reason": f"Suite eval_type is {eval_type!r}, not 'capability'. Only capability suites can be promoted.",
        }

    # Load report JSON
    with open(report_path) as f:
        report = json.load(f)

    summary = report.get("summary", {})
    overall_pass_at_1 = summary.get("overall_pass_at_1", 0.0)

    if overall_pass_at_1 < 0.95:
        return {
            "promoted": False,
            "reason": f"Suite is not saturated (pass@1 = {overall_pass_at_1:.2%}, threshold = 95%).",
        }

    # Promote: update YAML
    suite_data["eval_type"] = "regression"
    with open(suite_path, "w") as f:
        yaml.dump(suite_data, f, default_flow_style=False, sort_keys=False)

    return {
        "promoted": True,
        "reason": f"Suite promoted: pass@1 = {overall_pass_at_1:.2%} >= 95%.",
        "old_eval_type": "capability",
        "new_eval_type": "regression",
    }


def generate_tasks_from_failures(
    report_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Generate YAML task stubs from failed tasks in a report.

    Finds tasks with pass@1 < 1.0 and creates placeholder task entries
    that can be edited into new, harder evaluation tasks.

    Returns a status dict with keys: generated (int), task_ids (list[str]).
    """
    report_path = Path(report_path)
    output_path = Path(output_path)

    with open(report_path) as f:
        report = json.load(f)

    results = report.get("results", [])
    failed_tasks = [r for r in results if r.get("pass_at_1", 1.0) < 1.0]

    if not failed_tasks:
        return {"generated": 0, "task_ids": [], "reason": "All tasks passed."}

    stubs: list[dict[str, Any]] = []
    generated_ids: list[str] = []
    for task_result in failed_tasks:
        task_id = task_result["task_id"]
        new_id = f"{task_id}_v2"
        generated_ids.append(new_id)

        stub: dict[str, Any] = {
            "id": new_id,
            "question": f"TODO: revised question based on {task_id}",
            "expected_output": [
                {"type": "entities", "value": ["TODO"]},
            ],
            "tags": {"complexity": "complex"},
            "graders": [{"type": "code"}],
            "num_trials": 3,
            "metadata": {
                "generated_from": task_id,
                "original_pass_at_1": task_result.get("pass_at_1", 0.0),
            },
        }
        stubs.append(stub)

    output_data = {
        "name": f"{report.get('suite_name', 'unknown')}_failures",
        "description": "Auto-generated task stubs from failures. Edit before use.",
        "tasks": stubs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    return {
        "generated": len(stubs),
        "task_ids": generated_ids,
        "output_path": str(output_path),
    }


def check_suite_balance(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze tag distributions across tasks and flag imbalances.

    For each tag key, computes value counts and flags imbalances where
    any tag value has <10% or >50% of tasks when there are 3+ distinct values.

    Args:
        tasks: List of task dicts, each with a "tags" key mapping to a dict.

    Returns:
        A dict with keys: balanced (bool), tag_distributions (dict),
        warnings (list[str]).
    """
    total = len(tasks)
    if total == 0:
        return {"balanced": True, "tag_distributions": {}, "warnings": []}

    # Collect all tag keys
    all_tag_keys: set[str] = set()
    for task in tasks:
        all_tag_keys.update(task.get("tags", {}).keys())

    tag_distributions: dict[str, dict[str, int]] = {}
    warnings: list[str] = []

    for key in sorted(all_tag_keys):
        counter: Counter[str] = Counter()
        for task in tasks:
            value = task.get("tags", {}).get(key)
            if value is not None:
                counter[str(value)] += 1

        tag_distributions[key] = dict(counter)

        # Only flag imbalance if 3+ distinct values
        if len(counter) >= 3:
            for value, count in counter.items():
                fraction = count / total
                if fraction < 0.10:
                    warnings.append(
                        f"Tag '{key}': value '{value}' is underrepresented "
                        f"({count}/{total} = {fraction:.0%})"
                    )
                elif fraction > 0.50:
                    warnings.append(
                        f"Tag '{key}': value '{value}' is overrepresented "
                        f"({count}/{total} = {fraction:.0%})"
                    )

    return {
        "balanced": len(warnings) == 0,
        "tag_distributions": tag_distributions,
        "warnings": warnings,
    }
