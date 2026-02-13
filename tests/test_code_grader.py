import pytest
from pankeval.graders.base import BaseGrader
from pankeval.graders.code_grader import CodeGrader
from pankeval.models import (
    GraderConfig, GradeResult, Task, Transcript, TranscriptEvent,
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
            expected_entities=entities,
            graders=[GraderConfig(type="code", checks=["entity_presence"])],
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
        config = GraderConfig(type="code", checks=["entity_presence"])
        result = CodeGrader().grade(task, "anything", Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0


class TestCypherPattern:
    def make_task_with_cypher(self, patterns):
        return Task(
            id="t1",
            question="Q?",
            expected_cypher_patterns=patterns,
            graders=[GraderConfig(type="code", checks=["cypher_pattern"])],
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
            expected_entities=["INS"],
            expected_cypher_patterns=["MATCH.*Gene"],
            graders=[
                GraderConfig(
                    type="code",
                    checks=["entity_presence", "cypher_pattern"],
                )
            ],
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
