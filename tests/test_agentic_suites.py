"""Tests for agentic benchmark task suite YAML files."""
from pathlib import Path

import pytest

from bioagenteval.graders.code_grader import CodeGrader
from bioagenteval.loader import load_suite, filter_tasks_by_tags
from bioagenteval.models import GraderConfig, Transcript, TranscriptEvent


TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


# ── Helpers ───────────────────────────────────────────────────────────────

def _grade_task(task, outcome: str, transcript: Transcript | None = None) -> float:
    """Run CodeGrader on a task with a canned outcome and optional transcript."""
    grader = CodeGrader()
    config = GraderConfig(type="code")
    if transcript is None:
        transcript = Transcript(task_id=task.id)
    result = grader.grade(task, outcome, transcript, config)
    return result.score


def _make_transcript_with_tools(task_id: str, tool_names: list[str]) -> Transcript:
    """Create a transcript with tool_call events for the given tool names."""
    events = [
        TranscriptEvent(event_type="tool_call", event_name=name)
        for name in tool_names
    ]
    return Transcript(task_id=task_id, events=events)


# ── Biomni-Eval1 Suite ────────────────────────────────────────────────────

class TestBiomniEval1Suite:
    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "biomni_eval1.yaml")
        assert suite.name == "biomni_eval1"
        assert suite.eval_type == "capability"
        assert len(tasks) == 3

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "biomni_eval1.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_has_drylab_and_wetlab_categories(self):
        _, tasks = load_suite(TASKS_DIR / "biomni_eval1.yaml")
        categories = {t.tags["category"] for t in tasks}
        assert "DryLab" in categories
        assert "WetLab" in categories

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "biomni_eval1.yaml")
        required = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_all_tasks_have_benchmark_tag(self):
        _, tasks = load_suite(TASKS_DIR / "biomni_eval1.yaml")
        for task in tasks:
            assert task.tags["benchmark"] == "Biomni-Eval1"

    def test_tool_calls_grading(self):
        _, tasks = load_suite(TASKS_DIR / "biomni_eval1.yaml")
        task = tasks[0]  # drylab gene expression — expects run_analysis + statistical_test
        transcript = _make_transcript_with_tools(task.id, ["run_analysis", "statistical_test"])
        outcome = "```python\nimport pandas as pd\n```\nfold change adjusted p-value upregulated differential expression"
        score = _grade_task(task, outcome, transcript)
        assert score > 0.5


# ── BixBench Suite ────────────────────────────────────────────────────────

class TestBixBenchSuite:
    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "bixbench.yaml")
        assert suite.name == "bixbench"
        assert suite.eval_type == "capability"
        assert len(tasks) == 3

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "bixbench.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_code_valid(self):
        _, tasks = load_suite(TASKS_DIR / "bixbench.yaml")
        for task in tasks:
            code_eos = [eo for eo in task.expected_output if eo.type == "code_valid"]
            assert len(code_eos) >= 1, f"Task {task.id} missing code_valid expected_output"

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "bixbench.yaml")
        required = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_code_valid_grading(self):
        _, tasks = load_suite(TASKS_DIR / "bixbench.yaml")
        task = tasks[0]  # GC content — expects valid Python
        outcome = "```python\nwith open('input.fasta') as f:\n    data = f.read()\nprint('GC content sorted')\n```"
        score = _grade_task(task, outcome)
        assert score > 0.5

    def test_invalid_code_grading(self):
        _, tasks = load_suite(TASKS_DIR / "bixbench.yaml")
        task = tasks[0]
        outcome = "```python\ndef broken(\n```"
        score = _grade_task(task, outcome)
        assert score < 1.0


# ── BioML-Bench Suite ─────────────────────────────────────────────────────

class TestBioMLBenchSuite:
    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "bioml_bench.yaml")
        assert suite.name == "bioml_bench"
        assert suite.eval_type == "capability"
        assert len(tasks) == 3

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "bioml_bench.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_has_diverse_categories(self):
        _, tasks = load_suite(TASKS_DIR / "bioml_bench.yaml")
        categories = {t.tags["category"] for t in tasks}
        assert len(categories) == 3  # DrugResponse, ProteinClassification, FeatureSelection

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "bioml_bench.yaml")
        required = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_numeric_tolerance_grading(self):
        _, tasks = load_suite(TASKS_DIR / "bioml_bench.yaml")
        task = tasks[0]  # drug response — expects RMSE ~0.85
        outcome = "```python\nprint('cross-validation feature importance RMSE')\n```\nThe RMSE is 0.82"
        score = _grade_task(task, outcome)
        assert score > 0.5


# ── SpatialBench Suite ────────────────────────────────────────────────────

class TestSpatialBenchSuite:
    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "spatialbench.yaml")
        assert suite.name == "spatialbench"
        assert suite.eval_type == "capability"
        assert len(tasks) == 3

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "spatialbench.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_has_set_similarity_checks(self):
        _, tasks = load_suite(TASKS_DIR / "spatialbench.yaml")
        for task in tasks:
            ss_eos = [eo for eo in task.expected_output if eo.type == "set_similarity"]
            assert len(ss_eos) >= 1, f"Task {task.id} missing set_similarity expected_output"

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "spatialbench.yaml")
        required = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_set_similarity_grading(self):
        _, tasks = load_suite(TASKS_DIR / "spatialbench.yaml")
        task = tasks[0]  # domain identification — expects cortical layers
        outcome = "Layer 1, Layer 2/3, Layer 4, Layer 5, Layer 6, White Matter\nspatial domain marker gene cortex clustering"
        score = _grade_task(task, outcome)
        assert score > 0.5


# ── scBench Suite ─────────────────────────────────────────────────────────

class TestScBenchSuite:
    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "scbench.yaml")
        assert suite.name == "scbench"
        assert suite.eval_type == "capability"
        assert len(tasks) == 3

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "scbench.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_has_diverse_categories(self):
        _, tasks = load_suite(TASKS_DIR / "scbench.yaml")
        categories = {t.tags["category"] for t in tasks}
        assert categories == {"CellTypeAnnotation", "TrajectoryInference", "BatchCorrection"}

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "scbench.yaml")
        required = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_cell_type_set_similarity(self):
        _, tasks = load_suite(TASKS_DIR / "scbench.yaml")
        task = tasks[0]  # cell type annotation
        outcome = (
            "```python\nimport scanpy as sc\n```\n"
            "CD4+ T cells, CD8+ T cells, B cells, NK cells, Monocytes, Dendritic cells"
        )
        score = _grade_task(task, outcome)
        assert score > 0.5


# ── BioAgent Bench Suite ──────────────────────────────────────────────────

class TestBioAgentBenchSuite:
    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "bioagent_bench.yaml")
        assert suite.name == "bioagent_bench"
        assert suite.eval_type == "capability"
        assert len(tasks) == 3

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "bioagent_bench.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_tool_calls(self):
        _, tasks = load_suite(TASKS_DIR / "bioagent_bench.yaml")
        for task in tasks:
            tc_eos = [eo for eo in task.expected_output if eo.type == "tool_calls"]
            assert len(tc_eos) >= 1, f"Task {task.id} missing tool_calls expected_output"

    def test_all_tasks_have_code_valid(self):
        _, tasks = load_suite(TASKS_DIR / "bioagent_bench.yaml")
        for task in tasks:
            cv_eos = [eo for eo in task.expected_output if eo.type == "code_valid"]
            assert len(cv_eos) >= 1, f"Task {task.id} missing code_valid expected_output"

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "bioagent_bench.yaml")
        required = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_pipeline_grading_with_tools(self):
        _, tasks = load_suite(TASKS_DIR / "bioagent_bench.yaml")
        task = tasks[0]  # RNA-seq pipeline
        transcript = _make_transcript_with_tools(
            task.id,
            ["run_fastqc", "run_trimming", "run_alignment", "run_quantification"]
        )
        outcome = (
            "```python\nimport deseq2\n```\n"
            "quality control alignment differential expression fold change adjusted p-value"
        )
        score = _grade_task(task, outcome, transcript)
        assert score > 0.5


# ── Cross-Suite Consistency ───────────────────────────────────────────────

class TestAgenticCrossSuiteConsistency:
    """Verify consistency across all agentic benchmark suites."""

    AGENTIC_SUITES = [
        "biomni_eval1.yaml", "bixbench.yaml", "bioml_bench.yaml",
        "spatialbench.yaml", "scbench.yaml", "bioagent_bench.yaml",
    ]

    def test_no_duplicate_ids_across_agentic_suites(self):
        all_ids = []
        for name in self.AGENTIC_SUITES:
            _, tasks = load_suite(TASKS_DIR / name)
            all_ids.extend(t.id for t in tasks)
        assert len(all_ids) == len(set(all_ids)), "Duplicate task IDs across agentic suites"

    def test_all_agentic_suites_load(self):
        for name in self.AGENTIC_SUITES:
            suite, tasks = load_suite(TASKS_DIR / name)
            assert suite.name, f"Suite in {name} has no name"
            assert len(tasks) == 3, f"Expected 3 tasks in {name}, got {len(tasks)}"
            assert suite.eval_type == "capability"

    def test_no_duplicate_ids_across_all_suites(self):
        """Verify no ID collisions across ALL suites (text + agentic)."""
        all_ids = []
        for yaml_file in TASKS_DIR.glob("*.yaml"):
            _, tasks = load_suite(yaml_file)
            all_ids.extend(t.id for t in tasks)
        assert len(all_ids) == len(set(all_ids)), "Duplicate task IDs across all suites"
