"""Deterministic code-based grader: dispatches on expected_output types."""
from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from bioagenteval.graders.base import BaseGrader
from bioagenteval.models import (
    ExpectedOutput, GradeResult, GraderConfig, Task, Transcript,
)


class CodeGrader(BaseGrader):
    """Deterministic grader that iterates over task.expected_output items."""

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        if not task.expected_output:
            return GradeResult(
                grader_type="code", score=1.0, passed=True, details={},
            )

        check_results: dict[str, float] = {}
        extra_details: dict[str, Any] = {}
        for eo in task.expected_output:
            if eo.type == "entities":
                check_results["entities"] = _check_entities(eo.value, outcome)
            elif eo.type == "cypher_patterns":
                check_results["cypher_patterns"] = _check_cypher_patterns(
                    eo.value, transcript,
                )
            elif eo.type == "mcq_answer":
                check_results["mcq_answer"] = _check_mcq_answer(eo.value, outcome)
            elif eo.type == "numeric_range":
                check_results["numeric_range"] = _check_numeric_range(
                    eo.value, outcome,
                )
            elif eo.type == "json_schema":
                js_score, js_details = _check_json_schema(eo.value, outcome)
                check_results["json_schema"] = js_score
                extra_details.update(js_details)

        if not check_results:
            score = 1.0
        else:
            score = sum(check_results.values()) / len(check_results)

        details: dict[str, Any] = {**check_results, **extra_details}
        return GradeResult(
            grader_type="code",
            score=score,
            passed=score >= 0.5,
            details=details,
        )


def _check_entities(entities: list[str], outcome: str) -> float:
    """Check what fraction of expected entities appear in the outcome."""
    if not entities:
        return 1.0
    outcome_lower = outcome.lower()
    found = sum(1 for e in entities if e.lower() in outcome_lower)
    return found / len(entities)


def _check_cypher_patterns(patterns: list[str], transcript: Transcript) -> float:
    """Check what fraction of expected Cypher patterns appear in the transcript."""
    if not patterns:
        return 1.0
    cypher_queries = [
        ev.data.get("query", "")
        for ev in transcript.events
        if ev.event_type == "cypher_query"
    ]
    all_cypher = " ".join(cypher_queries)
    matched = sum(
        1
        for pat in patterns
        if re.search(pat, all_cypher, re.IGNORECASE)
    )
    return matched / len(patterns)


def _check_mcq_answer(expected: str, outcome: str) -> float:
    """Check if the expected MCQ answer appears in the outcome.

    Supports exact match and flexible patterns like "The answer is B",
    "(B)", "Answer: B".
    """
    expected_upper = expected.strip().upper()
    outcome_upper = outcome.upper()

    # Exact match: the answer letter appears standalone
    if expected_upper in outcome_upper:
        return 1.0

    # Flexible patterns
    patterns = [
        rf"\b{re.escape(expected_upper)}\b",
        rf"answer\s*(?:is|:)\s*{re.escape(expected_upper)}",
        rf"\({re.escape(expected_upper)}\)",
    ]
    for pat in patterns:
        if re.search(pat, outcome_upper):
            return 1.0

    return 0.0


def _check_json_schema(
    schema: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Validate that the outcome is valid JSON conforming to a JSON schema.

    Returns (score, details) where score is 1.0 if valid, 0.0 otherwise,
    and details contains error information on failure.
    """
    try:
        data = json.loads(outcome)
    except (json.JSONDecodeError, TypeError) as exc:
        return 0.0, {"parse_error": str(exc)}

    try:
        jsonschema.Draft7Validator.check_schema(schema)
        validator = jsonschema.Draft7Validator(schema)
    except jsonschema.SchemaError as exc:
        return 0.0, {"schema_error": str(exc)}

    try:
        errors = list(validator.iter_errors(data))
    except Exception as exc:
        return 0.0, {"schema_error": str(exc)}

    if not errors:
        return 1.0, {}

    return 0.0, {
        "validation_errors": [e.message for e in errors],
    }


def _check_numeric_range(value: dict[str, Any], outcome: str) -> float:
    """Check if a numeric answer falls within the expected range.

    value should contain 'target' and optionally 'min'/'max'.
    """
    # Extract numbers from outcome
    numbers = re.findall(r"-?\d+\.?\d*", outcome)
    if not numbers:
        return 0.0

    target = value.get("target")
    min_val = value.get("min")
    max_val = value.get("max")

    for num_str in numbers:
        num = float(num_str)
        if min_val is not None and num < min_val:
            continue
        if max_val is not None and num > max_val:
            continue
        if target is not None and num == target:
            return 1.0
        if min_val is not None or max_val is not None:
            return 1.0

    return 0.0
