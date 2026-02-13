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
tasks:
  - id: t1
    question: "What genes cause diabetes?"
    expected_entities:
      - INS
      - HLA-DRB1
    expected_complexity: complex
    graders:
      - type: code
        checks:
          - entity_presence
      - type: model
        rubric: "Is it correct?"
    num_trials: 3
  - id: t2
    question: "Tell me about INS."
    expected_entities:
      - INS
    num_trials: 2
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
        assert tasks[0].expected_entities == ["INS", "HLA-DRB1"]
        assert len(tasks[0].graders) == 2
        assert tasks[0].num_trials == 3
        assert tasks[1].num_trials == 2
        assert suite.task_ids == ["t1", "t2"]

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
