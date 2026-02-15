import pytest
from pathlib import Path

from bioagenteval.loader import load_suite
from bioagenteval.models import EvalSuite, Task


MINIMAL_YAML = """\
name: test_suite
description: A test suite
tasks:
  - id: t1
    question: "What is gene X?"
"""

FULL_YAML = """\
name: full_suite
description: Full featured suite
default_num_trials: 5
default_tracked_metrics:
  - type: transcript
    metrics:
      - n_turns
      - n_tool_calls
tasks:
  - id: t1
    question: "What genes cause diabetes?"
    expected_output:
      - type: entities
        value:
          - INS
          - HLA-DRB1
    tags:
      complexity: complex
    graders:
      - type: code
      - type: model
        rubric: "Is it correct?"
    num_trials: 3
  - id: t2
    question: "Tell me about INS."
    expected_output:
      - type: entities
        value:
          - INS
    num_trials: 2
"""

MCQ_YAML = """\
name: mcq_suite
tasks:
  - id: mcq1
    question: "Which answer is correct? A, B, C, D"
    expected_output:
      - type: mcq_answer
        value: B
    graders:
      - type: code
"""

CYPHER_YAML = """\
name: cypher_suite
tasks:
  - id: cypher1
    question: "Describe the insulin pathway."
    expected_output:
      - type: entities
        value:
          - INSR
      - type: cypher_patterns
        value:
          - "MATCH.*insulin"
    tags:
      complexity: complex
    graders:
      - type: code
"""


class TestLoadSuite:
    def test_load_minimal(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(MINIMAL_YAML)
        suite, tasks = load_suite(f)
        assert isinstance(suite, EvalSuite)
        assert suite.name == "test_suite"
        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        assert tasks[0].question == "What is gene X?"

    def test_load_full(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(FULL_YAML)
        suite, tasks = load_suite(f)
        assert suite.name == "full_suite"
        assert suite.default_num_trials == 5
        assert len(tasks) == 2
        assert len(tasks[0].expected_output) == 1
        assert tasks[0].expected_output[0].type == "entities"
        assert tasks[0].expected_output[0].value == ["INS", "HLA-DRB1"]
        assert tasks[0].tags == {"complexity": "complex"}
        assert len(tasks[0].graders) == 2
        assert tasks[0].num_trials == 3
        assert tasks[1].num_trials == 2
        assert suite.task_ids == ["t1", "t2"]

    def test_load_default_tracked_metrics(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(FULL_YAML)
        suite, tasks = load_suite(f)
        assert len(suite.default_tracked_metrics) == 1
        assert suite.default_tracked_metrics[0].type == "transcript"
        # Tasks without own tracked_metrics get suite defaults
        assert len(tasks[0].tracked_metrics) == 1
        assert "n_turns" in tasks[0].tracked_metrics[0].metrics

    def test_load_applies_default_num_trials(self, tmp_path):
        yaml_content = """\
name: defaults
default_num_trials: 7
tasks:
  - id: t1
    question: "Q?"
"""
        f = tmp_path / "suite.yaml"
        f.write_text(yaml_content)
        suite, tasks = load_suite(f)
        assert tasks[0].num_trials == 7

    def test_task_explicit_trials_override_default(self, tmp_path):
        yaml_content = """\
name: override
default_num_trials: 7
tasks:
  - id: t1
    question: "Q?"
    num_trials: 2
"""
        f = tmp_path / "suite.yaml"
        f.write_text(yaml_content)
        _, tasks = load_suite(f)
        assert tasks[0].num_trials == 2

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_suite(Path("/nonexistent/path.yaml"))

    def test_load_string_path(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(MINIMAL_YAML)
        suite, tasks = load_suite(str(f))
        assert suite.name == "test_suite"

    def test_load_mcq_task(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(MCQ_YAML)
        _, tasks = load_suite(f)
        assert len(tasks[0].expected_output) == 1
        assert tasks[0].expected_output[0].type == "mcq_answer"
        assert tasks[0].expected_output[0].value == "B"

    def test_load_cypher_patterns(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(CYPHER_YAML)
        _, tasks = load_suite(f)
        eo_types = {eo.type for eo in tasks[0].expected_output}
        assert "entities" in eo_types
        assert "cypher_patterns" in eo_types
        cypher_eo = [eo for eo in tasks[0].expected_output if eo.type == "cypher_patterns"][0]
        assert cypher_eo.value == ["MATCH.*insulin"]
        assert tasks[0].tags["complexity"] == "complex"
