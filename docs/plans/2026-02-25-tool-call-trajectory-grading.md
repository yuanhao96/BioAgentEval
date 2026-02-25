# Plan: Tool Call & Trajectory Grading

**Date**: 2026-02-25
**Milestone**: Tool Call & Trajectory Grading

## Overview

Add three new expected_output types to CodeGrader for grading agent trajectories:
1. `tool_calls` — verify expected tools were used
2. `turn_limit` — enforce max interaction turns
3. `trajectory_pattern` — ordered regex matching on event sequences

## Steps

### Step 1: _check_tool_calls (code_grader.py)

Add dispatch + check function:
- value: list of dicts with `tool_name` (required) and `params` (optional dict)
- Search transcript events with event_type in {"tool_call", "tool_use", "cypher_query"}
- For each expected tool call, check if a matching event exists (event_name or data.tool_name matches)
- If params specified, verify they are a subset of event.data
- Return fraction of expected tool calls found

### Step 2: _check_turn_limit (code_grader.py)

Add dispatch + check function:
- value: dict with `max_turns` (int)
- Count llm_call events in transcript
- Return 1.0 if count <= max_turns, 0.0 otherwise

### Step 3: _check_trajectory_pattern (code_grader.py)

Add dispatch + check function:
- value: list of regex pattern strings
- Build ordered sequence of event_types from transcript
- For each pattern, search for a match in the remaining event_type sequence (preserving order)
- Return fraction of patterns matched

### Step 4: Example tasks in biomedical_core.yaml

Add 1-2 tasks demonstrating the new types.

### Step 5: Tests in test_code_grader.py

Test classes: TestToolCalls, TestTurnLimit, TestTrajectoryPattern

### Step 6: Update docs

CLAUDE.md and README.md — mention new expected_output types.

## Files Modified

1. src/bioagenteval/code_grader.py — 3 new check functions + dispatch
2. tasks/biomedical_core.yaml — example tasks
3. tests/test_code_grader.py — new test classes
4. CLAUDE.md — docs
5. README.md — docs
