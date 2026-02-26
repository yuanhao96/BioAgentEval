"""Deterministic code-based grader: dispatches on expected_output types."""
from __future__ import annotations

import ast
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
            elif eo.type == "code_valid":
                cv_score, cv_details = _check_code_valid(eo.value, outcome)
                check_results["code_valid"] = cv_score
                extra_details.update(cv_details)
            elif eo.type == "test_results":
                tr_score, tr_details = _check_test_results(
                    eo.value, transcript,
                )
                check_results["test_results"] = tr_score
                extra_details.update(tr_details)
            elif eo.type == "groundedness":
                gr_score, gr_details = _check_groundedness(eo.value, outcome)
                check_results["groundedness"] = gr_score
                extra_details.update(gr_details)
            elif eo.type == "keyword_coverage":
                check_results["keyword_coverage"] = _check_keyword_coverage(
                    eo.value, outcome,
                )
            elif eo.type == "state_check":
                sc_score, sc_details = _check_state(eo.value, transcript)
                check_results["state_check"] = sc_score
                extra_details.update(sc_details)
            elif eo.type == "exact_match":
                check_results["exact_match"] = _check_exact_match(
                    eo.value, outcome,
                )
            elif eo.type == "set_similarity":
                ss_score, ss_details = _check_set_similarity(
                    eo.value, outcome,
                )
                check_results["set_similarity"] = ss_score
                extra_details.update(ss_details)
            elif eo.type == "precision_at_k":
                pk_score, pk_details = _check_precision_at_k(
                    eo.value, outcome,
                )
                check_results["precision_at_k"] = pk_score
                extra_details.update(pk_details)
            elif eo.type == "numeric_tolerance":
                nt_score, nt_details = _check_numeric_tolerance(
                    eo.value, outcome,
                )
                check_results["numeric_tolerance"] = nt_score
                extra_details.update(nt_details)

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


# ---------------------------------------------------------------------------
# Agent-type-specific checks
# ---------------------------------------------------------------------------


def _extract_code_blocks(outcome: str) -> list[str]:
    """Extract code from markdown fenced blocks or return full outcome."""
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", outcome, re.DOTALL)
    return blocks if blocks else [outcome]


def _check_code_valid(
    value: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Check that code in the outcome is syntactically valid.

    Coding-agent check.  Currently supports Python via ``ast.parse()``.
    value: {"language": "python"} (default: python).
    """
    language = value.get("language", "python") if isinstance(value, dict) else "python"
    blocks = _extract_code_blocks(outcome)

    if language != "python":
        return 0.0, {"error": f"unsupported language: {language}"}

    errors: list[str] = []
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as exc:
            errors.append(f"line {exc.lineno}: {exc.msg}")

    if not errors:
        return 1.0, {}
    return 0.0, {"syntax_errors": errors}


def _check_test_results(
    value: dict[str, Any], transcript: Transcript,
) -> tuple[float, dict[str, Any]]:
    """Verify test outcomes from transcript events.

    Coding-agent check.  Looks for ``event_type == "test_result"`` events
    with ``data.passed`` (bool) and ``data.test_name`` (str).

    value formats:
    - {"expected_tests": ["test_a", "test_b"]}  -> fraction of listed tests that passed
    - {"min_pass_rate": 0.9}                     -> 1.0 if pass-rate >= threshold, else 0.0
    """
    test_events = [
        ev for ev in transcript.events if ev.event_type == "test_result"
    ]

    if not test_events:
        return 0.0, {"error": "no test_result events in transcript"}

    expected_tests = value.get("expected_tests")
    min_pass_rate = value.get("min_pass_rate")

    if expected_tests:
        results_map = {
            ev.data.get("test_name", ""): ev.data.get("passed", False)
            for ev in test_events
        }
        matched = sum(1 for t in expected_tests if results_map.get(t))
        score = matched / len(expected_tests)
        details: dict[str, Any] = {
            "expected_tests": expected_tests,
            "passed_tests": [t for t in expected_tests if results_map.get(t)],
            "failed_tests": [t for t in expected_tests if not results_map.get(t)],
        }
        return score, details

    # min_pass_rate mode
    total = len(test_events)
    passed = sum(1 for ev in test_events if ev.data.get("passed"))
    rate = passed / total
    threshold = min_pass_rate if min_pass_rate is not None else 1.0
    score = 1.0 if rate >= threshold else 0.0
    return score, {"pass_rate": rate, "threshold": threshold, "total": total, "passed": passed}


# Regex patterns for detecting citations in text
_CITATION_PATTERNS = [
    re.compile(r"https?://\S+"),                          # URLs
    re.compile(r"\b10\.\d{4,}/\S+"),                      # DOIs
    re.compile(r"\[\d+\]"),                                # Numbered refs [1]
    re.compile(r"[A-Z][a-z]+\s+et\s+al\.?,?\s*\(?\d{4}"), # Author et al., 2023
]


def _find_citations(text: str) -> list[str]:
    """Return all citation-like strings found in text."""
    found: list[str] = []
    for pat in _CITATION_PATTERNS:
        found.extend(pat.findall(text))
    return found


def _check_groundedness(
    value: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Check that the outcome contains citations or references.

    Research-agent check.

    value formats:
    - {"required_sources": ["url_or_doi", ...]}  -> fraction of sources found
    - {"min_citations": 3}                       -> 1.0 if >= N citations detected
    """
    required = value.get("required_sources")
    min_cit = value.get("min_citations")

    detected = _find_citations(outcome)

    if required:
        outcome_lower = outcome.lower()
        matched = sum(1 for src in required if src.lower() in outcome_lower)
        return matched / len(required), {
            "required_sources": required,
            "matched_count": matched,
            "detected_citations": detected,
        }

    if min_cit is not None:
        n = len(detected)
        score = 1.0 if n >= min_cit else 0.0
        return score, {
            "min_citations": min_cit,
            "detected_count": n,
            "detected_citations": detected,
        }

    # Default: just report whether any citations exist
    return (1.0 if detected else 0.0), {"detected_citations": detected}


def _check_keyword_coverage(
    value: dict[str, Any], outcome: str,
) -> float:
    """Check that required keywords/topics are covered in the outcome.

    Research-agent check.

    value: {"keywords": ["topic1", "pattern2"], "match_mode": "substring"|"regex"}
    Default match_mode: "substring" (case-insensitive).
    """
    keywords = value.get("keywords", [])
    if not keywords:
        return 1.0

    mode = value.get("match_mode", "substring")
    outcome_lower = outcome.lower()
    matched = 0

    for kw in keywords:
        if mode == "regex":
            if re.search(kw, outcome, re.IGNORECASE):
                matched += 1
        else:
            if kw.lower() in outcome_lower:
                matched += 1

    return matched / len(keywords)


def _resolve_nested(data: dict[str, Any], dotted_key: str) -> Any:
    """Resolve a dot-notation key against a nested dict."""
    parts = dotted_key.split(".")
    current: Any = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _check_state(
    value: dict[str, Any], transcript: Transcript,
) -> tuple[float, dict[str, Any]]:
    """Verify key-value assertions against the last state snapshot in the transcript.

    Conversational / computer-use agent check.

    value: {"assertions": {"key": "expected", "nested.key": "expected"}}
    Looks for the last ``state_snapshot`` event in transcript.events.
    Supports dot-notation for nested access.
    """
    assertions = value.get("assertions", {})
    if not assertions:
        return 1.0, {}

    # Find the last state_snapshot event
    state_events = [
        ev for ev in transcript.events if ev.event_type == "state_snapshot"
    ]
    if not state_events:
        return 0.0, {"error": "no state_snapshot events in transcript"}

    state_data = state_events[-1].data
    matched = 0
    failed: list[dict[str, Any]] = []

    for key, expected in assertions.items():
        actual = _resolve_nested(state_data, key)
        if actual == expected:
            matched += 1
        else:
            failed.append({"key": key, "expected": expected, "actual": actual})

    score = matched / len(assertions)
    details: dict[str, Any] = {"total_assertions": len(assertions), "matched": matched}
    if failed:
        details["failed_assertions"] = failed
    return score, details


# ---------------------------------------------------------------------------
# Benchmark-specific checks
# ---------------------------------------------------------------------------


def _normalize_text(text: str, case_sensitive: bool = False, strip_whitespace: bool = True) -> str:
    """Normalize text for comparison."""
    if strip_whitespace:
        text = " ".join(text.split())
    if not case_sensitive:
        text = text.lower()
    return text.strip()


def _check_exact_match(value: dict[str, Any] | str, outcome: str) -> float:
    """Check if the outcome exactly matches the expected answer.

    Used by HLE-Bio (short answer), FrontierScience-Bio (Olympiad).

    value formats:
    - str: the expected answer
    - dict: {"answer": str, "case_sensitive": bool, "strip_whitespace": bool}
    """
    if isinstance(value, str):
        expected = value
        case_sensitive = False
        strip_ws = True
    else:
        expected = value.get("answer", "")
        case_sensitive = value.get("case_sensitive", False)
        strip_ws = value.get("strip_whitespace", True)

    norm_expected = _normalize_text(expected, case_sensitive, strip_ws)
    norm_outcome = _normalize_text(outcome, case_sensitive, strip_ws)

    if norm_expected == norm_outcome:
        return 1.0

    # Also check if the expected answer appears as the final line or after "Answer:"
    for pattern in [
        rf"(?:answer|response)\s*[:=]\s*{re.escape(norm_expected)}\s*$",
        rf"^{re.escape(norm_expected)}\s*$",
    ]:
        if re.search(pattern, norm_outcome, re.MULTILINE | (0 if case_sensitive else re.IGNORECASE)):
            return 1.0

    return 0.0


def _check_set_similarity(
    value: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Compute Jaccard similarity between expected and predicted sets.

    Used by SpatialBench/scBench for label set comparisons.

    value: {"expected": ["label1", "label2", ...], "separator": ","}
    The outcome is split by separator (default: newline or comma) to extract
    predicted items.
    """
    expected_items = value.get("expected", [])
    if not expected_items:
        return 1.0, {}

    separator = value.get("separator", r"[,\n]")
    predicted_raw = re.split(separator, outcome)
    predicted_items = [item.strip().lower() for item in predicted_raw if item.strip()]

    expected_set = {item.lower() for item in expected_items}
    predicted_set = set(predicted_items)

    intersection = expected_set & predicted_set
    union = expected_set | predicted_set

    if not union:
        return 1.0, {}

    jaccard = len(intersection) / len(union)

    return jaccard, {
        "expected_count": len(expected_set),
        "predicted_count": len(predicted_set),
        "intersection_count": len(intersection),
        "jaccard": jaccard,
    }


def _check_precision_at_k(
    value: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Compute precision@K for ranked lists.

    Used by SpatialBench/scBench for gene list evaluation.

    value: {"expected": ["gene1", "gene2", ...], "k": 10, "separator": ","}
    The outcome is split to extract a ranked list; only the top K items
    are evaluated against the expected set.
    """
    expected_items = value.get("expected", [])
    k = value.get("k", len(expected_items))
    separator = value.get("separator", r"[,\n]")

    if not expected_items:
        return 1.0, {}

    predicted_raw = re.split(separator, outcome)
    predicted_items = [item.strip().lower() for item in predicted_raw if item.strip()]
    top_k = predicted_items[:k]

    if not top_k:
        return 0.0, {"error": "no predicted items found in outcome"}

    expected_set = {item.lower() for item in expected_items}
    hits = sum(1 for item in top_k if item in expected_set)
    precision = hits / len(top_k)

    return precision, {
        "k": k,
        "top_k_count": len(top_k),
        "hits": hits,
        "precision_at_k": precision,
    }


def _check_numeric_tolerance(
    value: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Check if a numeric answer is within tolerance of the expected value.

    Used by SpatialBench/scBench for numeric grading, BioML-Bench for
    ML metric evaluation (AUROC, Spearman, etc.).

    value: {"expected": float, "abs_tol": float, "rel_tol": float}
    At least one of abs_tol or rel_tol must be specified.
    If both are specified, the answer passes if EITHER tolerance is met.
    """
    expected = value.get("expected")
    abs_tol = value.get("abs_tol")
    rel_tol = value.get("rel_tol")

    if expected is None:
        return 0.0, {"error": "missing 'expected' value"}

    # Extract numbers from outcome
    numbers = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", outcome)
    if not numbers:
        return 0.0, {"error": "no numbers found in outcome"}

    expected_f = float(expected)

    # Check each extracted number
    for num_str in numbers:
        num = float(num_str)
        within_abs = False
        within_rel = False

        if abs_tol is not None:
            within_abs = abs(num - expected_f) <= abs_tol
        if rel_tol is not None and expected_f != 0:
            within_rel = abs(num - expected_f) / abs(expected_f) <= rel_tol

        if within_abs or within_rel:
            return 1.0, {
                "expected": expected_f,
                "actual": num,
                "abs_diff": abs(num - expected_f),
            }

    # Return the closest match details
    parsed = [float(n) for n in numbers]
    closest = min(parsed, key=lambda x: abs(x - expected_f))
    return 0.0, {
        "expected": expected_f,
        "closest": closest,
        "abs_diff": abs(closest - expected_f),
    }
