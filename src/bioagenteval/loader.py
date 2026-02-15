"""Load evaluation suites and tasks from YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bioagenteval.models import (
    EvalSuite, ExpectedOutput, GraderConfig, MetricGroup, Task,
)


def load_suite(path: str | Path) -> tuple[EvalSuite, list[Task]]:
    """Load an evaluation suite from a YAML file.

    Returns (EvalSuite, list[Task]).  Tasks whose YAML block omits
    ``num_trials`` inherit ``default_num_trials`` from the suite.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    default_trials = raw.get("default_num_trials", 1)

    # Parse suite-level default_tracked_metrics
    default_tracked_metrics = [
        MetricGroup(**mg)
        for mg in raw.get("default_tracked_metrics", [])
    ]

    tasks: list[Task] = []
    task_ids: list[str] = []
    for t_raw in raw.get("tasks", []):
        # Apply default num_trials if not explicitly set
        if "num_trials" not in t_raw:
            t_raw["num_trials"] = default_trials

        # Parse grader configs
        grader_dicts = t_raw.pop("graders", [])
        graders = [GraderConfig(**g) for g in grader_dicts]

        # Parse expected_output
        eo_dicts = t_raw.pop("expected_output", [])
        expected_output = [ExpectedOutput(**eo) for eo in eo_dicts]

        # Parse tracked_metrics
        tm_dicts = t_raw.pop("tracked_metrics", [])
        tracked_metrics = [MetricGroup(**mg) for mg in tm_dicts]

        task = Task(
            **t_raw,
            graders=graders,
            expected_output=expected_output,
            tracked_metrics=tracked_metrics,
        )

        # Apply suite-level default_tracked_metrics if task has none
        if not task.tracked_metrics and default_tracked_metrics:
            task.tracked_metrics = default_tracked_metrics

        tasks.append(task)
        task_ids.append(task.id)

    suite = EvalSuite(
        name=raw["name"],
        description=raw.get("description", ""),
        task_ids=task_ids,
        default_num_trials=default_trials,
        default_tracked_metrics=default_tracked_metrics,
    )
    return suite, tasks
