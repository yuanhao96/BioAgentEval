"""Tests for benchmark task suite YAML files — validates loading, grading, and structure."""
from pathlib import Path

import pytest

from bioagenteval.graders.code_grader import CodeGrader
from bioagenteval.loader import load_suite, filter_tasks_by_tags
from bioagenteval.models import GraderConfig, Transcript


TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


# ── Helper ────────────────────────────────────────────────────────────────

def _grade_task(task, outcome: str) -> float:
    """Run CodeGrader on a task with a canned outcome string."""
    grader = CodeGrader()
    config = GraderConfig(type="code")
    transcript = Transcript(task_id=task.id)
    result = grader.grade(task, outcome, transcript, config)
    return result.score


# ── HLE-Bio Suite ─────────────────────────────────────────────────────────

class TestHLEBioSuite:
    """Verify the existing HLE-Bio suite loads and grades correctly."""

    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "hle_bio_chem.yaml")
        assert suite.name == "hle_bio_chem"
        assert len(tasks) == 149

    def test_all_tasks_have_ids(self):
        _, tasks = load_suite(TASKS_DIR / "hle_bio_chem.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_all_tasks_have_expected_output(self):
        _, tasks = load_suite(TASKS_DIR / "hle_bio_chem.yaml")
        for task in tasks:
            assert len(task.expected_output) > 0, f"Task {task.id} has no expected_output"

    def test_mcq_grading_correct(self):
        _, tasks = load_suite(TASKS_DIR / "hle_bio_chem.yaml")
        # Find a task with mcq_answer expected_output
        mcq_tasks = [t for t in tasks if any(eo.type == "mcq_answer" for eo in t.expected_output)]
        assert len(mcq_tasks) > 0, "No MCQ tasks found in HLE-Bio suite"
        task = mcq_tasks[0]
        answer = next(eo.value for eo in task.expected_output if eo.type == "mcq_answer")
        score = _grade_task(task, f"The answer is {answer}")
        assert score == 1.0

    def test_mcq_grading_incorrect(self):
        _, tasks = load_suite(TASKS_DIR / "hle_bio_chem.yaml")
        mcq_tasks = [t for t in tasks if any(eo.type == "mcq_answer" for eo in t.expected_output)]
        task = mcq_tasks[0]
        answer = next(eo.value for eo in task.expected_output if eo.type == "mcq_answer")
        wrong = "Z" if answer != "Z" else "Y"
        score = _grade_task(task, f"The answer is {wrong}")
        assert score == 0.0

    def test_all_tasks_have_tags(self):
        _, tasks = load_suite(TASKS_DIR / "hle_bio_chem.yaml")
        for task in tasks:
            assert "category" in task.tags, f"Task {task.id} missing 'category' tag"


# ── FrontierScience-Bio Suite ─────────────────────────────────────────────

class TestFrontierScienceBioSuite:
    """Validate the FrontierScience-Bio task suite."""

    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        assert suite.name == "frontierscience_bio"
        assert suite.eval_type == "capability"
        assert len(tasks) == 6

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_has_olympiad_and_research_categories(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        categories = {t.tags["category"] for t in tasks}
        assert "Olympiad" in categories
        assert "Research" in categories

    def test_olympiad_tasks_count(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        olympiad = filter_tasks_by_tags(tasks, {"category": "Olympiad"})
        assert len(olympiad) == 3

    def test_research_tasks_count(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        research = filter_tasks_by_tags(tasks, {"category": "Research"})
        assert len(research) == 3

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        required_tags = {"benchmark", "category", "subject", "answer_type", "difficulty"}
        for task in tasks:
            missing = required_tags - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_all_tasks_have_benchmark_tag(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        for task in tasks:
            assert task.tags["benchmark"] == "FrontierScience-Bio"

    def test_olympiad_exact_match_grading(self):
        """Olympiad tasks with exact_match should grade correctly."""
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        olympiad = filter_tasks_by_tags(tasks, {"category": "Olympiad"})
        # Take first Olympiad task (enzyme kinetics: answer=30)
        task = olympiad[0]
        assert any(eo.type == "exact_match" for eo in task.expected_output)
        score = _grade_task(task, "The apparent Km is 30 mM")
        assert score >= 0.5  # Should match via exact_match and/or numeric_tolerance

    def test_olympiad_wrong_answer(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        olympiad = filter_tasks_by_tags(tasks, {"category": "Olympiad"})
        task = olympiad[0]
        score = _grade_task(task, "The apparent Km is 999 mM")
        assert score == 0.0

    def test_research_keyword_coverage_grading(self):
        """Research tasks with keyword_coverage should grade correctly."""
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        research = filter_tasks_by_tags(tasks, {"category": "Research"})
        task = research[0]
        assert any(eo.type == "keyword_coverage" for eo in task.expected_output)
        # Provide an answer containing the required and optional keywords
        outcome = (
            "CRISPR-Cas9 off-target effects occur due to PAM-proximal mismatch tolerance "
            "in the guide RNA binding. High-fidelity Cas9 variants and GUIDE-seq validation "
            "are two strategies to minimize off-targets. Truncated guide RNAs also help."
        )
        score = _grade_task(task, outcome)
        assert score > 0.5

    def test_research_poor_answer(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        research = filter_tasks_by_tags(tasks, {"category": "Research"})
        task = research[0]
        score = _grade_task(task, "I don't know anything about CRISPR.")
        assert score < 0.5

    def test_all_tasks_have_graders(self):
        _, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        for task in tasks:
            assert len(task.graders) >= 1, f"Task {task.id} has no graders"

    def test_default_num_trials_applied(self):
        suite, tasks = load_suite(TASKS_DIR / "frontierscience_bio.yaml")
        assert suite.default_num_trials == 3
        for task in tasks:
            assert task.num_trials == 3


# ── LAB-Bench Suite ───────────────────────────────────────────────────────

class TestLABBenchSuite:
    """Validate the LAB-Bench task suite."""

    def test_loads_successfully(self):
        suite, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        assert suite.name == "lab_bench"
        assert suite.eval_type == "capability"
        assert len(tasks) == 12

    def test_all_task_ids_unique(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_has_all_four_subcategories(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        categories = {t.tags["category"] for t in tasks}
        assert categories == {"LitQA2", "CloningScenarios", "ProtocolQA", "SeqQA"}

    def test_three_tasks_per_subcategory(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        for cat in ["LitQA2", "CloningScenarios", "ProtocolQA", "SeqQA"]:
            cat_tasks = filter_tasks_by_tags(tasks, {"category": cat})
            assert len(cat_tasks) == 3, f"Expected 3 tasks for {cat}, got {len(cat_tasks)}"

    def test_all_tasks_have_required_tags(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        required_tags = {"benchmark", "category", "subject", "answer_type"}
        for task in tasks:
            missing = required_tags - set(task.tags.keys())
            assert not missing, f"Task {task.id} missing tags: {missing}"

    def test_all_tasks_have_benchmark_tag(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        for task in tasks:
            assert task.tags["benchmark"] == "LAB-Bench"

    def test_all_tasks_are_mcq(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        for task in tasks:
            mcq_eos = [eo for eo in task.expected_output if eo.type == "mcq_answer"]
            assert len(mcq_eos) == 1, f"Task {task.id} should have exactly one mcq_answer"

    def test_mcq_correct_grading(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        for task in tasks:
            answer = next(eo.value for eo in task.expected_output if eo.type == "mcq_answer")
            score = _grade_task(task, f"The answer is {answer}")
            assert score == 1.0, f"Task {task.id} should score 1.0 for correct answer {answer}"

    def test_mcq_incorrect_grading(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        task = tasks[0]
        answer = next(eo.value for eo in task.expected_output if eo.type == "mcq_answer")
        wrong = "Z" if answer != "Z" else "Y"
        score = _grade_task(task, f"The answer is {wrong}")
        assert score == 0.0

    def test_all_tasks_have_graders(self):
        _, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        for task in tasks:
            assert len(task.graders) >= 1, f"Task {task.id} has no graders"

    def test_default_num_trials_applied(self):
        suite, tasks = load_suite(TASKS_DIR / "lab_bench.yaml")
        assert suite.default_num_trials == 3
        for task in tasks:
            assert task.num_trials == 3


# ── Cross-Suite Checks ────────────────────────────────────────────────────

class TestCrossSuiteConsistency:
    """Verify consistency across all benchmark suites."""

    def test_no_duplicate_ids_across_suites(self):
        all_ids = []
        for yaml_file in TASKS_DIR.glob("*.yaml"):
            _, tasks = load_suite(yaml_file)
            all_ids.extend(t.id for t in tasks)
        assert len(all_ids) == len(set(all_ids)), "Duplicate task IDs across suites"

    def test_all_suites_load_without_error(self):
        yaml_files = list(TASKS_DIR.glob("*.yaml"))
        assert len(yaml_files) >= 3, "Expected at least 3 suite files"
        for yaml_file in yaml_files:
            suite, tasks = load_suite(yaml_file)
            assert suite.name, f"Suite in {yaml_file.name} has no name"
            assert len(tasks) > 0, f"Suite in {yaml_file.name} has no tasks"
