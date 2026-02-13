"""Deterministic code-based grader: entity presence, Cypher pattern matching."""
from __future__ import annotations

import re

from pankeval.graders.base import BaseGrader
from pankeval.models import GradeResult, GraderConfig, Task, Transcript


class CodeGrader(BaseGrader):
    """Deterministic grader using string matching and pattern checks."""

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        check_results: dict[str, float] = {}

        for check in config.checks:
            if check == "entity_presence":
                check_results[check] = self._check_entity_presence(task, outcome)
            elif check == "cypher_pattern":
                check_results[check] = self._check_cypher_pattern(task, transcript)

        if not check_results:
            score = 1.0
        else:
            score = sum(check_results.values()) / len(check_results)

        return GradeResult(
            grader_type="code",
            score=score,
            passed=score >= 0.5,
            details=check_results,
        )

    def _check_entity_presence(self, task: Task, outcome: str) -> float:
        if not task.expected_entities:
            return 1.0
        outcome_lower = outcome.lower()
        found = sum(
            1 for e in task.expected_entities if e.lower() in outcome_lower
        )
        return found / len(task.expected_entities)

    def _check_cypher_pattern(self, task: Task, transcript: Transcript) -> float:
        if not task.expected_cypher_patterns:
            return 1.0
        cypher_queries = [
            ev.data.get("query", "")
            for ev in transcript.events
            if ev.event_type == "cypher_query"
        ]
        all_cypher = " ".join(cypher_queries)
        matched = sum(
            1
            for pat in task.expected_cypher_patterns
            if re.search(pat, all_cypher, re.IGNORECASE)
        )
        return matched / len(task.expected_cypher_patterns)
