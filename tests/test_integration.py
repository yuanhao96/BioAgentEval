"""End-to-end integration test: load suite -> run agent -> grade -> report."""
import json
import pytest

from bioagenteval.graders import CodeGrader
from bioagenteval.loader import load_suite
from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent
from bioagenteval.reporter import EvalReporter
from bioagenteval.runner import EvalRunner


SUITE_YAML = """\
name: integration_test
description: Minimal integration test suite
default_num_trials: 2

tasks:
  - id: gene_lookup
    question: "What genes are associated with type 1 diabetes?"
    expected_output:
      - type: entities
        value:
          - INS
          - HLA-DRB1
    tags:
      complexity: complex
    graders:
      - type: code
    num_trials: 2
  - id: pathway_check
    question: "Describe the insulin signaling pathway."
    expected_output:
      - type: entities
        value:
          - INSR
          - IRS1
      - type: cypher_patterns
        value:
          - "MATCH.*insulin"
    graders:
      - type: code
    num_trials: 1
"""


class StubAgent:
    """Agent that returns a canned response with Cypher queries."""
    def run(self, question: str) -> AgentResponse:
        return AgentResponse(
            outcome="INS and HLA-DRB1 are key genes. INSR and IRS1 in pathway.",
            transcript=Transcript(
                task_id="stub",
                events=[
                    TranscriptEvent(
                        event_type="cypher_query",
                        data={"query": "MATCH (p:Pathway {name:'insulin signaling'}) RETURN p"},
                    ),
                    TranscriptEvent(
                        event_type="llm_response",
                        data={"answer": "INS and HLA-DRB1..."},
                    ),
                ],
            ),
        )

    def reset(self) -> None:
        pass


class JsonStubAgent:
    """Agent that returns a JSON response for json_schema grading tests."""
    def __init__(self, response_json: str):
        self._response = response_json

    def run(self, question: str) -> AgentResponse:
        return AgentResponse(
            outcome=self._response,
            transcript=Transcript(task_id="stub"),
        )

    def reset(self) -> None:
        pass


class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        # 1. Write and load suite
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(SUITE_YAML)
        suite, tasks = load_suite(suite_file)
        assert len(tasks) == 2

        # 2. Run evaluation
        runner = EvalRunner(
            agent=StubAgent(),
            graders={"code": CodeGrader()},
        )
        results = runner.run_suite(tasks)
        assert len(results) == 2

        # 3. Check results
        # Task 1: INS and HLA-DRB1 should both be found
        assert results[0].pass_at_k(k=1) == 1.0
        # Task 2: entities found, cypher pattern matched
        assert results[1].trials[0].grades[0].passed is True

        # 4. Generate and save report
        report_path = tmp_path / "report.json"
        EvalReporter.save_report(suite.name, results, report_path)

        report = json.loads(report_path.read_text())
        assert report["suite_name"] == "integration_test"
        assert report["summary"]["total_tasks"] == 2
        assert report["summary"]["overall_pass_at_1"] == 1.0
        assert len(report["results"][0]["trials"]) == 2
        assert len(report["results"][1]["trials"]) == 1

    def test_json_schema_pipeline(self, tmp_path):
        """Test full pipeline with a json_schema expected output."""
        yaml_str = """\
name: json_schema_test
tasks:
  - id: gene_json
    question: "Return a JSON object for TP53."
    expected_output:
      - type: json_schema
        value:
          type: object
          properties:
            symbol:
              type: string
            name:
              type: string
          required:
            - symbol
            - name
    graders:
      - type: code
"""
        f = tmp_path / "json_schema.yaml"
        f.write_text(yaml_str)
        suite, tasks = load_suite(f)
        assert len(tasks) == 1
        assert tasks[0].expected_output[0].type == "json_schema"

        agent = JsonStubAgent(json.dumps({"symbol": "TP53", "name": "Tumor protein p53"}))
        runner = EvalRunner(agent=agent, graders={"code": CodeGrader()})
        results = runner.run_suite(tasks)

        trial = results[0].trials[0]
        assert trial.grades[0].passed is True
        assert trial.grades[0].score == 1.0
        assert "validation_errors" not in trial.grades[0].details

    def test_json_schema_pipeline_failure(self, tmp_path):
        """Test pipeline with invalid JSON response against schema."""
        yaml_str = """\
name: json_schema_fail
tasks:
  - id: gene_json_bad
    question: "Return a JSON object for TP53."
    expected_output:
      - type: json_schema
        value:
          type: object
          required:
            - symbol
    graders:
      - type: code
"""
        f = tmp_path / "json_fail.yaml"
        f.write_text(yaml_str)
        _, tasks = load_suite(f)

        agent = JsonStubAgent("This is not JSON at all")
        runner = EvalRunner(agent=agent, graders={"code": CodeGrader()})
        results = runner.run_suite(tasks)

        trial = results[0].trials[0]
        assert trial.grades[0].passed is False
        assert trial.grades[0].score == 0.0
        assert "parse_error" in trial.grades[0].details

    def test_partial_match_pipeline(self, tmp_path):
        """Test that partial entity matches produce fractional scores."""
        yaml = """\
name: partial
tasks:
  - id: t1
    question: "Genes for diabetes?"
    expected_output:
      - type: entities
        value:
          - INS
          - MISSING_GENE
          - ANOTHER_MISSING
    graders:
      - type: code
"""
        f = tmp_path / "partial.yaml"
        f.write_text(yaml)
        _, tasks = load_suite(f)

        runner = EvalRunner(
            agent=StubAgent(),
            graders={"code": CodeGrader()},
        )
        results = runner.run_suite(tasks)
        # Only 1/3 entities found -> score ~ 0.33, passed = False
        trial = results[0].trials[0]
        assert trial.grades[0].score == pytest.approx(1.0 / 3.0)
        assert trial.grades[0].passed is False
