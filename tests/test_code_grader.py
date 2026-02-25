import json

import pytest
from bioagenteval.graders.base import BaseGrader
from bioagenteval.graders.code_grader import CodeGrader
from bioagenteval.models import (
    ExpectedOutput, GraderConfig, GradeResult, Task, Transcript, TranscriptEvent,
)


class TestCodeGraderIsBaseGrader:
    def test_inherits_base(self):
        grader = CodeGrader()
        assert isinstance(grader, BaseGrader)


class TestEntityPresence:
    def make_task(self, entities):
        return Task(
            id="t1",
            question="Q?",
            expected_output=[ExpectedOutput(type="entities", value=entities)],
            graders=[GraderConfig(type="code")],
        )

    def test_all_entities_present(self):
        task = self.make_task(["INS", "HLA-DRB1"])
        outcome = "The INS gene and HLA-DRB1 are associated with diabetes."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0
        assert result.grader_type == "code"

    def test_partial_entities(self):
        task = self.make_task(["INS", "HLA-DRB1", "PTPN22"])
        outcome = "The INS gene is important."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == pytest.approx(1.0 / 3.0)

    def test_no_entities_present(self):
        task = self.make_task(["INS", "HLA-DRB1"])
        outcome = "I don't know."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive_match(self):
        task = self.make_task(["ins", "hla-drb1"])
        outcome = "The INS gene and HLA-DRB1 are relevant."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True

    def test_no_expected_entities(self):
        task = self.make_task([])
        config = GraderConfig(type="code")
        result = CodeGrader().grade(task, "anything", Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0


class TestCypherPattern:
    def make_task_with_cypher(self, patterns):
        return Task(
            id="t1",
            question="Q?",
            expected_output=[ExpectedOutput(type="cypher_patterns", value=patterns)],
            graders=[GraderConfig(type="code")],
        )

    def test_cypher_pattern_found(self):
        task = self.make_task_with_cypher(["MATCH.*Gene"])
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": "MATCH (g:Gene) RETURN g"},
                )
            ],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "answer", transcript, config)
        assert result.passed is True
        assert result.score == 1.0

    def test_cypher_pattern_not_found(self):
        task = self.make_task_with_cypher(["MATCH.*Pathway"])
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": "MATCH (g:Gene) RETURN g"},
                )
            ],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "answer", transcript, config)
        assert result.passed is False

    def test_no_cypher_events(self):
        task = self.make_task_with_cypher(["MATCH.*Gene"])
        transcript = Transcript(task_id="t1", events=[])
        config = task.graders[0]
        result = CodeGrader().grade(task, "answer", transcript, config)
        assert result.passed is False
        assert result.score == 0.0


class TestMultipleChecks:
    def test_combined_checks(self):
        task = Task(
            id="t1",
            question="Q?",
            expected_output=[
                ExpectedOutput(type="entities", value=["INS"]),
                ExpectedOutput(type="cypher_patterns", value=["MATCH.*Gene"]),
            ],
            graders=[GraderConfig(type="code")],
        )
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": "MATCH (g:Gene {name:'INS'}) RETURN g"},
                )
            ],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "INS is a gene", transcript, config)
        assert result.passed is True
        assert result.score == 1.0


class TestMcqAnswer:
    def make_mcq_task(self, answer):
        return Task(
            id="t1",
            question="Which is correct? A, B, C, D",
            expected_output=[ExpectedOutput(type="mcq_answer", value=answer)],
            graders=[GraderConfig(type="code")],
        )

    def test_exact_match(self):
        task = self.make_mcq_task("B")
        config = task.graders[0]
        result = CodeGrader().grade(task, "B", Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0

    def test_answer_is_pattern(self):
        task = self.make_mcq_task("B")
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "The answer is B", Transcript(task_id="t1"), config
        )
        assert result.passed is True

    def test_parenthesized(self):
        task = self.make_mcq_task("C")
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "I think (C) is correct", Transcript(task_id="t1"), config
        )
        assert result.passed is True

    def test_wrong_answer(self):
        task = self.make_mcq_task("B")
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "The answer is definitely A", Transcript(task_id="t1"), config
        )
        # "B" doesn't appear standalone in "The answer is definitely A"
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive(self):
        task = self.make_mcq_task("b")
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "Answer: B", Transcript(task_id="t1"), config
        )
        assert result.passed is True


class TestNumericRange:
    def make_numeric_task(self, value):
        return Task(
            id="t1",
            question="What is the value?",
            expected_output=[ExpectedOutput(type="numeric_range", value=value)],
            graders=[GraderConfig(type="code")],
        )

    def test_exact_target(self):
        task = self.make_numeric_task({"target": 42.0})
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "The value is 42.0", Transcript(task_id="t1"), config
        )
        assert result.passed is True

    def test_within_range(self):
        task = self.make_numeric_task({"min": 10, "max": 50})
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "The answer is 25", Transcript(task_id="t1"), config
        )
        assert result.passed is True

    def test_outside_range(self):
        task = self.make_numeric_task({"min": 10, "max": 50})
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "The answer is 100", Transcript(task_id="t1"), config
        )
        assert result.passed is False

    def test_no_numbers_in_outcome(self):
        task = self.make_numeric_task({"target": 42.0})
        config = task.graders[0]
        result = CodeGrader().grade(
            task, "I have no idea", Transcript(task_id="t1"), config
        )
        assert result.passed is False
        assert result.score == 0.0


class TestJsonSchema:
    SIMPLE_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }

    def make_task(self, schema):
        return Task(
            id="t1",
            question="Return a JSON object.",
            expected_output=[ExpectedOutput(type="json_schema", value=schema)],
            graders=[GraderConfig(type="code")],
        )

    def test_valid_json_matching_schema(self):
        task = self.make_task(self.SIMPLE_SCHEMA)
        outcome = json.dumps({"name": "Alice", "age": 30})
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0
        assert "validation_errors" not in result.details
        assert "parse_error" not in result.details

    def test_schema_violation(self):
        task = self.make_task(self.SIMPLE_SCHEMA)
        outcome = json.dumps({"age": 30})  # missing required "name"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == 0.0
        assert "validation_errors" in result.details
        assert any("'name' is a required property" in e for e in result.details["validation_errors"])

    def test_multiple_validation_errors(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        task = self.make_task(schema)
        outcome = json.dumps({})  # missing both required fields
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert len(result.details["validation_errors"]) == 2

    def test_malformed_json(self):
        task = self.make_task(self.SIMPLE_SCHEMA)
        config = task.graders[0]
        result = CodeGrader().grade(task, "not json at all", Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == 0.0
        assert "parse_error" in result.details

    def test_empty_string(self):
        task = self.make_task(self.SIMPLE_SCHEMA)
        config = task.graders[0]
        result = CodeGrader().grade(task, "", Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == 0.0
        assert "parse_error" in result.details

    def test_wrong_type(self):
        task = self.make_task(self.SIMPLE_SCHEMA)
        outcome = json.dumps({"name": 123})  # name should be string
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert "validation_errors" in result.details

    def test_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "gene": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "organism": {"type": "string"},
                    },
                    "required": ["symbol"],
                },
            },
            "required": ["gene"],
        }
        task = self.make_task(schema)
        outcome = json.dumps({"gene": {"symbol": "TP53", "organism": "human"}})
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0

    def test_invalid_schema(self):
        task = self.make_task({"type": "not_a_real_type"})
        outcome = json.dumps({"a": 1})
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == 0.0
        assert "schema_error" in result.details

    def test_combined_with_entities(self):
        task = Task(
            id="t1",
            question="Return JSON with gene info.",
            expected_output=[
                ExpectedOutput(type="entities", value=["TP53"]),
                ExpectedOutput(type="json_schema", value={
                    "type": "object",
                    "properties": {"gene": {"type": "string"}},
                    "required": ["gene"],
                }),
            ],
            graders=[GraderConfig(type="code")],
        )
        outcome = json.dumps({"gene": "TP53"})
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        # entities: "TP53" is in the outcome string '{"gene": "TP53"}' → 1.0
        # json_schema: valid → 1.0
        # average = 1.0
        assert result.passed is True
        assert result.score == 1.0

    def test_array_schema(self):
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
        task = self.make_task(schema)
        outcome = json.dumps(["gene1", "gene2"])
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0


class TestNoExpectedOutput:
    def test_empty_expected_output(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "anything", Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0
