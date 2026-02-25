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


class TestToolCalls:
    def _transcript_with_tools(self, tool_events):
        events = []
        for name, data in tool_events:
            events.append(TranscriptEvent(
                event_type="tool_call", event_name=name, data=data,
            ))
        return Transcript(task_id="t1", events=events)

    def make_task(self, tool_calls):
        return Task(
            id="t1",
            question="Query the graph.",
            expected_output=[ExpectedOutput(type="tool_calls", value=tool_calls)],
            graders=[GraderConfig(type="code")],
        )

    def test_all_tools_found(self):
        task = self.make_task([{"tool_name": "cypher_query"}, {"tool_name": "search"}])
        transcript = self._transcript_with_tools([
            ("cypher_query", {}),
            ("search", {}),
        ])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0
        assert result.passed is True
        assert "missing_tool_calls" not in result.details

    def test_partial_tools_found(self):
        task = self.make_task([{"tool_name": "cypher_query"}, {"tool_name": "fetch"}])
        transcript = self._transcript_with_tools([("cypher_query", {})])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 0.5
        assert "missing_tool_calls" in result.details
        assert "fetch" in result.details["missing_tool_calls"]

    def test_no_tools_found(self):
        task = self.make_task([{"tool_name": "cypher_query"}])
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 0.0
        assert result.passed is False

    def test_with_params_match(self):
        task = self.make_task([{
            "tool_name": "cypher_query",
            "params": {"query": "MATCH (g:Gene) RETURN g"},
        }])
        transcript = self._transcript_with_tools([
            ("cypher_query", {"query": "MATCH (g:Gene) RETURN g", "db": "neo4j"}),
        ])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0

    def test_with_params_mismatch(self):
        task = self.make_task([{
            "tool_name": "cypher_query",
            "params": {"query": "MATCH (g:Gene) RETURN g"},
        }])
        transcript = self._transcript_with_tools([
            ("cypher_query", {"query": "MATCH (d:Disease) RETURN d"}),
        ])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 0.0

    def test_empty_expected(self):
        task = self.make_task([])
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0

    def test_tool_name_from_data(self):
        """Tool name can come from event.data['tool_name'] if event_name is empty."""
        task = self.make_task([{"tool_name": "search"}])
        events = [TranscriptEvent(
            event_type="tool_use", data={"tool_name": "search"},
        )]
        transcript = Transcript(task_id="t1", events=events)
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0


class TestTurnLimit:
    def make_task(self, max_turns):
        return Task(
            id="t1",
            question="Q?",
            expected_output=[ExpectedOutput(type="turn_limit", value={"max_turns": max_turns})],
            graders=[GraderConfig(type="code")],
        )

    def _transcript_with_turns(self, n):
        events = [TranscriptEvent(event_type="llm_call") for _ in range(n)]
        return Transcript(task_id="t1", events=events)

    def test_within_limit(self):
        task = self.make_task(5)
        transcript = self._transcript_with_turns(3)
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0
        assert result.passed is True
        assert result.details["actual_turns"] == 3

    def test_at_limit(self):
        task = self.make_task(3)
        transcript = self._transcript_with_turns(3)
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0

    def test_over_limit(self):
        task = self.make_task(2)
        transcript = self._transcript_with_turns(5)
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 0.0
        assert result.passed is False
        assert result.details["actual_turns"] == 5
        assert result.details["max_turns"] == 2

    def test_zero_turns(self):
        task = self.make_task(5)
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0


class TestTrajectoryPattern:
    def make_task(self, patterns):
        return Task(
            id="t1",
            question="Q?",
            expected_output=[ExpectedOutput(type="trajectory_pattern", value=patterns)],
            graders=[GraderConfig(type="code")],
        )

    def _transcript_with_events(self, event_types):
        events = [TranscriptEvent(event_type=et) for et in event_types]
        return Transcript(task_id="t1", events=events)

    def test_all_patterns_match(self):
        task = self.make_task(["llm_call", "tool_call", "llm_call"])
        transcript = self._transcript_with_events([
            "llm_call", "tool_call", "llm_call",
        ])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0

    def test_partial_match(self):
        task = self.make_task(["llm_call", "cypher_query", "tool_call"])
        transcript = self._transcript_with_events(["llm_call", "cypher_query"])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == pytest.approx(2.0 / 3.0)

    def test_no_match(self):
        task = self.make_task(["cypher_query"])
        transcript = self._transcript_with_events(["llm_call"])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 0.0

    def test_order_matters(self):
        task = self.make_task(["tool_call", "llm_call"])
        transcript = self._transcript_with_events(["llm_call", "tool_call"])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        # "tool_call" doesn't appear before "llm_call" in "llm_call tool_call"
        # Actually: event_sequence = "llm_call tool_call"
        # Pattern "tool_call" matches at position 9, then "llm_call" needs to match
        # after that — but there's no llm_call after tool_call, so only 1/2 match
        assert result.score == 0.5

    def test_regex_patterns(self):
        task = self.make_task(["llm_.*", "tool_.*"])
        transcript = self._transcript_with_events([
            "llm_call", "tool_use",
        ])
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0

    def test_empty_patterns(self):
        task = self.make_task([])
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 1.0

    def test_empty_transcript(self):
        task = self.make_task(["llm_call"])
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "result", transcript, config)
        assert result.score == 0.0


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


# ---------------------------------------------------------------------------
# Agent-type-specific checks
# ---------------------------------------------------------------------------


class TestCodeValid:
    def make_task(self, value):
        return Task(
            id="t1",
            question="Write a Python function.",
            expected_output=[ExpectedOutput(type="code_valid", value=value)],
            graders=[GraderConfig(type="code")],
        )

    def test_valid_python(self):
        task = self.make_task({"language": "python"})
        outcome = "def hello():\n    return 42"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0
        assert result.passed is True

    def test_valid_python_in_markdown(self):
        task = self.make_task({"language": "python"})
        outcome = '```python\ndef greet(name):\n    return f"Hello {name}"\n```'
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_invalid_python(self):
        task = self.make_task({"language": "python"})
        outcome = "def broken(\n    return"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.0
        assert result.passed is False
        assert "syntax_errors" in result.details

    def test_default_language_is_python(self):
        task = self.make_task({})
        outcome = "x = 1 + 2"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_unsupported_language(self):
        task = self.make_task({"language": "rust"})
        outcome = "fn main() {}"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.0
        assert "unsupported language" in result.details.get("error", "")

    def test_multiple_code_blocks(self):
        task = self.make_task({"language": "python"})
        outcome = '```python\nx = 1\n```\nSome text\n```python\ny = 2\n```'
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_one_invalid_block_fails(self):
        task = self.make_task({"language": "python"})
        outcome = '```python\nx = 1\n```\n```python\ndef bad(\n```'
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.0


class TestTestResults:
    def _transcript_with_tests(self, test_results):
        events = [
            TranscriptEvent(
                event_type="test_result",
                data={"test_name": name, "passed": passed},
            )
            for name, passed in test_results
        ]
        return Transcript(task_id="t1", events=events)

    def make_task(self, value):
        return Task(
            id="t1",
            question="Fix the bug.",
            expected_output=[ExpectedOutput(type="test_results", value=value)],
            graders=[GraderConfig(type="code")],
        )

    def test_all_expected_tests_pass(self):
        task = self.make_task({"expected_tests": ["test_a", "test_b"]})
        transcript = self._transcript_with_tests([("test_a", True), ("test_b", True)])
        config = task.graders[0]
        result = CodeGrader().grade(task, "fixed", transcript, config)
        assert result.score == 1.0
        assert result.passed is True

    def test_partial_expected_tests(self):
        task = self.make_task({"expected_tests": ["test_a", "test_b"]})
        transcript = self._transcript_with_tests([("test_a", True), ("test_b", False)])
        config = task.graders[0]
        result = CodeGrader().grade(task, "fixed", transcript, config)
        assert result.score == 0.5
        assert "test_a" in result.details["passed_tests"]
        assert "test_b" in result.details["failed_tests"]

    def test_no_test_events(self):
        task = self.make_task({"expected_tests": ["test_a"]})
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "fixed", transcript, config)
        assert result.score == 0.0
        assert "error" in result.details

    def test_min_pass_rate_met(self):
        task = self.make_task({"min_pass_rate": 0.8})
        transcript = self._transcript_with_tests([
            ("t1", True), ("t2", True), ("t3", True), ("t4", True), ("t5", False),
        ])
        config = task.graders[0]
        result = CodeGrader().grade(task, "fixed", transcript, config)
        assert result.score == 1.0
        assert result.details["pass_rate"] == pytest.approx(0.8)

    def test_min_pass_rate_not_met(self):
        task = self.make_task({"min_pass_rate": 1.0})
        transcript = self._transcript_with_tests([("t1", True), ("t2", False)])
        config = task.graders[0]
        result = CodeGrader().grade(task, "fixed", transcript, config)
        assert result.score == 0.0

    def test_default_min_pass_rate_is_1(self):
        task = self.make_task({})
        transcript = self._transcript_with_tests([("t1", True), ("t2", False)])
        config = task.graders[0]
        result = CodeGrader().grade(task, "fixed", transcript, config)
        assert result.score == 0.0  # Default threshold is 1.0


class TestGroundedness:
    def make_task(self, value):
        return Task(
            id="t1",
            question="Summarize the research.",
            expected_output=[ExpectedOutput(type="groundedness", value=value)],
            graders=[GraderConfig(type="code")],
        )

    def test_required_sources_all_found(self):
        task = self.make_task({"required_sources": [
            "https://pubmed.ncbi.nlm.nih.gov/12345",
            "10.1038/nature12345",
        ]})
        outcome = (
            "According to https://pubmed.ncbi.nlm.nih.gov/12345, "
            "the gene is involved (10.1038/nature12345)."
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_required_sources_partial(self):
        task = self.make_task({"required_sources": [
            "https://pubmed.ncbi.nlm.nih.gov/12345",
            "https://example.com/missing",
        ]})
        outcome = "See https://pubmed.ncbi.nlm.nih.gov/12345 for details."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.5

    def test_min_citations_met(self):
        task = self.make_task({"min_citations": 2})
        outcome = "Studies [1] and [2] show that the protein is essential."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0
        assert result.details["detected_count"] >= 2

    def test_min_citations_not_met(self):
        task = self.make_task({"min_citations": 3})
        outcome = "The gene is important [1]."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.0

    def test_detects_url_citations(self):
        task = self.make_task({"min_citations": 1})
        outcome = "Source: https://www.nature.com/articles/s41586"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_detects_doi_citations(self):
        task = self.make_task({"min_citations": 1})
        outcome = "Published as 10.1038/s41586-021-03819-2"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_detects_author_year_citations(self):
        task = self.make_task({"min_citations": 1})
        outcome = "As shown by Smith et al., 2023, the pathway is active."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_no_citations(self):
        task = self.make_task({"min_citations": 1})
        outcome = "The gene is important."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.0

    def test_default_mode_any_citation(self):
        task = self.make_task({})
        outcome = "See [1] for details."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0


class TestKeywordCoverage:
    def make_task(self, value):
        return Task(
            id="t1",
            question="Explain the pathway.",
            expected_output=[ExpectedOutput(type="keyword_coverage", value=value)],
            graders=[GraderConfig(type="code")],
        )

    def test_all_keywords_found(self):
        task = self.make_task({"keywords": ["insulin", "glucose", "pancreas"]})
        outcome = "Insulin is produced by the pancreas and regulates glucose."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_partial_keywords(self):
        task = self.make_task({"keywords": ["insulin", "glucose", "liver"]})
        outcome = "Insulin regulates glucose levels."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == pytest.approx(2.0 / 3.0)

    def test_no_keywords_found(self):
        task = self.make_task({"keywords": ["BRCA1", "mutation"]})
        outcome = "The weather is nice today."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 0.0

    def test_case_insensitive_substring(self):
        task = self.make_task({"keywords": ["INSULIN", "GLUCOSE"]})
        outcome = "insulin and glucose are related."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_regex_mode(self):
        task = self.make_task({
            "keywords": [r"insulin\s+signaling", r"type\s+[12]\s+diabetes"],
            "match_mode": "regex",
        })
        outcome = "Insulin signaling is disrupted in type 2 diabetes."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0

    def test_empty_keywords(self):
        task = self.make_task({"keywords": []})
        outcome = "anything"
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.score == 1.0


class TestStateCheck:
    def _transcript_with_state(self, state_data):
        events = [
            TranscriptEvent(event_type="state_snapshot", data=state_data),
        ]
        return Transcript(task_id="t1", events=events)

    def make_task(self, value):
        return Task(
            id="t1",
            question="Update the record.",
            expected_output=[ExpectedOutput(type="state_check", value=value)],
            graders=[GraderConfig(type="code")],
        )

    def test_all_assertions_pass(self):
        task = self.make_task({"assertions": {"status": "resolved", "priority": "high"}})
        transcript = self._transcript_with_state({"status": "resolved", "priority": "high"})
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 1.0
        assert result.passed is True

    def test_partial_assertions(self):
        task = self.make_task({"assertions": {"status": "resolved", "priority": "low"}})
        transcript = self._transcript_with_state({"status": "resolved", "priority": "high"})
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 0.5
        assert result.details["failed_assertions"][0]["key"] == "priority"

    def test_no_state_events(self):
        task = self.make_task({"assertions": {"status": "done"}})
        transcript = Transcript(task_id="t1")
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 0.0
        assert "error" in result.details

    def test_nested_dot_notation(self):
        task = self.make_task({"assertions": {"user.email": "test@example.com"}})
        transcript = self._transcript_with_state({
            "user": {"email": "test@example.com", "name": "Test"},
        })
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 1.0

    def test_nested_key_missing(self):
        task = self.make_task({"assertions": {"user.phone": "555-1234"}})
        transcript = self._transcript_with_state({
            "user": {"email": "test@example.com"},
        })
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 0.0
        assert result.details["failed_assertions"][0]["actual"] is None

    def test_uses_last_snapshot(self):
        """Multiple state_snapshot events — should use the last one."""
        events = [
            TranscriptEvent(event_type="state_snapshot", data={"status": "open"}),
            TranscriptEvent(event_type="state_snapshot", data={"status": "resolved"}),
        ]
        task = self.make_task({"assertions": {"status": "resolved"}})
        transcript = Transcript(task_id="t1", events=events)
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 1.0

    def test_empty_assertions(self):
        task = self.make_task({"assertions": {}})
        transcript = self._transcript_with_state({"anything": True})
        config = task.graders[0]
        result = CodeGrader().grade(task, "done", transcript, config)
        assert result.score == 1.0
