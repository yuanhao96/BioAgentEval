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
            elif eo.type == "tool_calls":
                tc_score, tc_details = _check_tool_calls(eo.value, transcript)
                check_results["tool_calls"] = tc_score
                extra_details.update(tc_details)
            elif eo.type == "turn_limit":
                tl_score, tl_details = _check_turn_limit(eo.value, transcript)
                check_results["turn_limit"] = tl_score
                extra_details.update(tl_details)
            elif eo.type == "trajectory_pattern":
                check_results["trajectory_pattern"] = _check_trajectory_pattern(
                    eo.value, transcript,
                )

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


def _check_tool_calls(
    expected_calls: list[dict[str, Any]], transcript: Transcript,
) -> tuple[float, dict[str, Any]]:
    """Verify that expected tool calls appear in the transcript.

    Each expected call has 'tool_name' (required) and optional 'params' dict.
    Returns (score, details) where score is fraction of expected calls matched.
    """
    if not expected_calls:
        return 1.0, {}

    tool_events = [
        ev for ev in transcript.events
        if ev.event_type in {"tool_call", "tool_use", "cypher_query"}
    ]

    matched = 0
    missing: list[str] = []
    for expected in expected_calls:
        tool_name = expected.get("tool_name", "")
        expected_params = expected.get("params", {})
        found = False
        for ev in tool_events:
            ev_name = ev.event_name or ev.data.get("tool_name", "")
            if ev_name != tool_name:
                continue
            if expected_params:
                if all(ev.data.get(k) == v for k, v in expected_params.items()):
                    found = True
                    break
            else:
                found = True
                break
        if found:
            matched += 1
        else:
            missing.append(tool_name)

    score = matched / len(expected_calls)
    details: dict[str, Any] = {}
    if missing:
        details["missing_tool_calls"] = missing
    return score, details


def _check_turn_limit(
    value: dict[str, Any], transcript: Transcript,
) -> tuple[float, dict[str, Any]]:
    """Check if the number of LLM turns is within the allowed limit.

    Returns (score, details) where score is 1.0 if within limit, 0.0 if exceeded.
    """
    max_turns = value.get("max_turns", 0)
    actual_turns = sum(1 for ev in transcript.events if ev.event_type == "llm_call")
    details = {"max_turns": max_turns, "actual_turns": actual_turns}
    if actual_turns <= max_turns:
        return 1.0, details
    return 0.0, details


def _check_trajectory_pattern(
    patterns: list[str], transcript: Transcript,
) -> float:
    """Check that regex patterns match event_types in order.

    Each pattern is matched against individual event_types, consuming events
    in order. Returns fraction of patterns matched.
    """
    if not patterns:
        return 1.0

    event_types = [ev.event_type for ev in transcript.events]

    matched = 0
    event_idx = 0
    for pat in patterns:
        while event_idx < len(event_types):
            if re.fullmatch(pat, event_types[event_idx]):
                matched += 1
                event_idx += 1
                break
            event_idx += 1

    return matched / len(patterns)


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
