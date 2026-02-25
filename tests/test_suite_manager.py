"""Tests for suite_manager: promote, generate tasks, balance check."""
from __future__ import annotations

import json
import textwrap

import pytest
import yaml

from bioagenteval.suite_manager import (
    check_suite_balance,
    generate_tasks_from_failures,
    promote_suite,
)


# ---------- helpers ----------

def _write_suite(tmp_path, eval_type="capability", name="test_suite"):
    """Write a minimal suite YAML and return the path."""
    suite_path = tmp_path / "suite.yaml"
    data = {
        "name": name,
        "eval_type": eval_type,
        "tasks": [
            {"id": "t1", "question": "Q1", "graders": [{"type": "code"}]},
        ],
    }
    suite_path.write_text(yaml.dump(data, default_flow_style=False))
    return suite_path


def _write_report(tmp_path, task_results, overall_pass_at_1=None, suite_name="test_suite"):
    """Write a report JSON and return the path."""
    report_path = tmp_path / "report.json"
    if overall_pass_at_1 is None:
        vals = [r.get("pass_at_1", 0.0) for r in task_results]
        overall_pass_at_1 = sum(vals) / len(vals) if vals else 0.0

    report = {
        "suite_name": suite_name,
        "results": task_results,
        "summary": {
            "total_tasks": len(task_results),
            "overall_pass_at_1": overall_pass_at_1,
            "saturated": overall_pass_at_1 >= 0.95,
        },
    }
    report_path.write_text(json.dumps(report))
    return report_path


# ---------- TestPromoteSuite ----------

class TestPromoteSuite:
    """Tests for promote_suite()."""

    def test_saturated_capability_promoted(self, tmp_path):
        suite_path = _write_suite(tmp_path, eval_type="capability")
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 1.0}],
            overall_pass_at_1=1.0,
        )

        result = promote_suite(suite_path, report_path)

        assert result["promoted"] is True
        assert result["old_eval_type"] == "capability"
        assert result["new_eval_type"] == "regression"

        # Verify YAML was updated
        with open(suite_path) as f:
            updated = yaml.safe_load(f)
        assert updated["eval_type"] == "regression"

    def test_non_saturated_not_promoted(self, tmp_path):
        suite_path = _write_suite(tmp_path, eval_type="capability")
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 0.5}],
            overall_pass_at_1=0.5,
        )

        result = promote_suite(suite_path, report_path)

        assert result["promoted"] is False
        assert "not saturated" in result["reason"]

    def test_non_capability_rejected(self, tmp_path):
        suite_path = _write_suite(tmp_path, eval_type="regression")
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 1.0}],
            overall_pass_at_1=1.0,
        )

        result = promote_suite(suite_path, report_path)

        assert result["promoted"] is False
        assert "not 'capability'" in result["reason"]

    def test_empty_eval_type_rejected(self, tmp_path):
        suite_path = _write_suite(tmp_path, eval_type="")
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 1.0}],
            overall_pass_at_1=1.0,
        )

        result = promote_suite(suite_path, report_path)

        assert result["promoted"] is False

    def test_exactly_at_threshold(self, tmp_path):
        """95% exactly should be promoted."""
        suite_path = _write_suite(tmp_path, eval_type="capability")
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 0.95}],
            overall_pass_at_1=0.95,
        )

        result = promote_suite(suite_path, report_path)

        assert result["promoted"] is True


# ---------- TestGenerateTasksFromFailures ----------

class TestGenerateTasksFromFailures:
    """Tests for generate_tasks_from_failures()."""

    def test_generates_stubs_for_failures(self, tmp_path):
        report_path = _write_report(
            tmp_path,
            [
                {"task_id": "t1", "pass_at_1": 1.0},
                {"task_id": "t2", "pass_at_1": 0.5},
                {"task_id": "t3", "pass_at_1": 0.0},
            ],
        )
        output_path = tmp_path / "stubs.yaml"

        result = generate_tasks_from_failures(report_path, output_path)

        assert result["generated"] == 2
        assert "t2_v2" in result["task_ids"]
        assert "t3_v2" in result["task_ids"]

        # Check output YAML
        with open(output_path) as f:
            stubs = yaml.safe_load(f)

        assert len(stubs["tasks"]) == 2
        assert stubs["tasks"][0]["id"] == "t2_v2"
        assert stubs["tasks"][0]["metadata"]["generated_from"] == "t2"

    def test_all_pass_no_stubs(self, tmp_path):
        report_path = _write_report(
            tmp_path,
            [
                {"task_id": "t1", "pass_at_1": 1.0},
                {"task_id": "t2", "pass_at_1": 1.0},
            ],
        )
        output_path = tmp_path / "stubs.yaml"

        result = generate_tasks_from_failures(report_path, output_path)

        assert result["generated"] == 0
        assert not output_path.exists()

    def test_preserves_original_pass_at_1(self, tmp_path):
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 0.33}],
        )
        output_path = tmp_path / "stubs.yaml"

        result = generate_tasks_from_failures(report_path, output_path)

        with open(output_path) as f:
            stubs = yaml.safe_load(f)

        assert stubs["tasks"][0]["metadata"]["original_pass_at_1"] == pytest.approx(0.33)

    def test_output_dir_created(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        report_path = _write_report(
            tmp_path,
            [{"task_id": "t1", "pass_at_1": 0.0}],
        )
        output_path = tmp_path / "subdir" / "stubs.yaml"

        result = generate_tasks_from_failures(report_path, output_path)

        assert result["generated"] == 1
        assert output_path.exists()


# ---------- TestCheckSuiteBalance ----------

class TestCheckSuiteBalance:
    """Tests for check_suite_balance()."""

    def test_balanced_suite(self):
        tasks = [
            {"tags": {"complexity": "simple", "domain": "gene"}},
            {"tags": {"complexity": "complex", "domain": "pathway"}},
            {"tags": {"complexity": "simple", "domain": "disease"}},
        ]
        result = check_suite_balance(tasks)

        assert result["balanced"] is True
        assert len(result["warnings"]) == 0
        assert result["tag_distributions"]["complexity"] == {"simple": 2, "complex": 1}

    def test_imbalanced_suite(self):
        """One value has >50% when 3+ distinct values."""
        tasks = [
            {"tags": {"level": "a"}},
            {"tags": {"level": "a"}},
            {"tags": {"level": "a"}},
            {"tags": {"level": "a"}},
            {"tags": {"level": "a"}},
            {"tags": {"level": "a"}},
            {"tags": {"level": "b"}},
            {"tags": {"level": "c"}},
        ]
        result = check_suite_balance(tasks)

        assert result["balanced"] is False
        assert any("overrepresented" in w for w in result["warnings"])

    def test_underrepresented_value(self):
        """One value has <10% when 3+ distinct values."""
        tasks = (
            [{"tags": {"cat": "x"}}] * 5
            + [{"tags": {"cat": "y"}}] * 5
            + [{"tags": {"cat": "z"}}] * 1  # 1/11 ≈ 9% < 10%
        )
        result = check_suite_balance(tasks)

        assert result["balanced"] is False
        assert any("underrepresented" in w for w in result["warnings"])

    def test_two_values_no_imbalance_flag(self):
        """With only 2 distinct values, no imbalance is flagged (need 3+)."""
        tasks = [
            {"tags": {"complexity": "simple"}},
            {"tags": {"complexity": "simple"}},
            {"tags": {"complexity": "simple"}},
            {"tags": {"complexity": "complex"}},
        ]
        result = check_suite_balance(tasks)

        assert result["balanced"] is True

    def test_empty_tasks(self):
        result = check_suite_balance([])

        assert result["balanced"] is True
        assert result["tag_distributions"] == {}

    def test_tasks_without_tags(self):
        tasks = [{"id": "t1"}, {"id": "t2"}]
        result = check_suite_balance(tasks)

        assert result["balanced"] is True
        assert result["tag_distributions"] == {}

    def test_multiple_tag_keys(self):
        """Each tag key is analyzed independently."""
        tasks = [
            {"tags": {"complexity": "simple", "domain": "gene"}},
            {"tags": {"complexity": "complex", "domain": "gene"}},
            {"tags": {"complexity": "simple", "domain": "pathway"}},
            {"tags": {"complexity": "complex", "domain": "disease"}},
        ]
        result = check_suite_balance(tasks)

        assert "complexity" in result["tag_distributions"]
        assert "domain" in result["tag_distributions"]
