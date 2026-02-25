# Agent-Type Evaluation Patterns

**Date**: 2026-02-25
**Milestone**: Agent-Type Evaluation Patterns

## Summary

Add 5 new `expected_output` check types to CodeGrader for agent-type-specific evaluation: `code_valid` and `test_results` (coding agents), `groundedness` and `keyword_coverage` (research agents), `state_check` (conversational + computer-use agents). Add example tasks and documentation.

## Files to Modify

1. `src/bioagenteval/graders/code_grader.py` — Add 5 new check functions + dispatch entries
2. `tests/test_code_grader.py` — Add test classes for each new check type
3. `tasks/biomedical_core.yaml` — Add example tasks demonstrating new check types

## Implementation Steps

### Step 1: Add `code_valid` check to CodeGrader

Check type for **coding agents**. Validates that code in the outcome parses correctly.

- Value format: `{"language": "python"}` (default: python)
- Extract code from markdown fences (```python...```) or treat full outcome as code
- Use `ast.parse()` for Python
- Score: 1.0 if valid, 0.0 if syntax error
- Details: include error message on failure

### Step 2: Add `test_results` check to CodeGrader

Check type for **coding agents**. Verifies test outcomes from transcript events.

- Value format: `{"min_pass_rate": 1.0}` or `{"expected_tests": ["test_a", "test_b"]}`
- Look for events with `event_type == "test_result"` in transcript
- Each event.data has: `test_name`, `passed` (bool), optional `message`
- If `expected_tests`: score = fraction of expected tests that passed
- If `min_pass_rate`: score = 1.0 if (passed_count / total_count) >= min_pass_rate, else 0.0
- If no test events found: score = 0.0

### Step 3: Add `groundedness` check to CodeGrader

Check type for **research agents**. Verifies citations/sources are present.

- Value format: `{"required_sources": ["url_or_doi"]}` or `{"min_citations": 3}`
- Citation patterns detected: URLs (http/https), DOIs (10.xxxx/...), numbered refs ([1]), author-year (Name et al., YYYY)
- If `required_sources`: score = fraction of required sources found in outcome
- If `min_citations`: score = 1.0 if detected_count >= min_citations, 0.0 otherwise
- Details: list of detected citations

### Step 4: Add `keyword_coverage` check to CodeGrader

Check type for **research agents**. Verifies required topics are covered.

- Value format: `{"keywords": ["topic1", "regex2"], "match_mode": "substring"|"regex"}`
- Default `match_mode`: "substring" (case-insensitive)
- Score: fraction of keywords matched
- Different from `entities`: supports regex mode and is explicitly for topic coverage

### Step 5: Add `state_check` check to CodeGrader

Check type for **conversational + computer-use agents**. Verifies backend/environment state.

- Value format: `{"assertions": {"key": "expected_value", "nested.key": "val"}}`
- Look for last `state_snapshot` event in transcript.events
- Support dot-notation for nested key access (e.g., "user.email")
- Score: fraction of assertions that match
- Details: list of failed assertions

### Step 6: Add dispatch entries in CodeGrader.grade()

Wire up all 5 new check types in the existing dispatch block.

### Step 7: Write tests for all new checks

For each check type, test:
- Happy path (all expected items present)
- Partial match
- No match / empty input
- Edge cases specific to each type

### Step 8: Add example tasks to biomedical_core.yaml

Add 2-3 example tasks showing:
- A coding agent task with `code_valid` + `test_results`
- A research agent task with `groundedness` + `keyword_coverage`
- A conversational/computer-use task with `state_check`

### Step 9: Run full test suite

Verify all 224 existing tests + new tests pass.

## Dependency Order

Steps 1-5 are independent (each adds one check function)
Step 6 depends on Steps 1-5
Step 7 can be written alongside Steps 1-5
Step 8 depends on Step 6
Step 9 is final
