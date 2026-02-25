# Robustness, Isolation & Observability

**Date**: 2026-02-25
**Milestone**: Robustness, Isolation & Observability

## Summary

Add trial isolation hooks, transcript analysis utilities, grader calibration workflow, and trial timeout support.

## Files to Create

1. `src/bioagenteval/transcript_analysis.py` — Pure functions for transcript summarization, tool sequence extraction, retry detection
2. `src/bioagenteval/calibration.py` — Grader calibration: accuracy, precision, recall, confusion matrix
3. `tests/test_transcript_analysis.py` — Tests for transcript analysis
4. `tests/test_calibration.py` — Tests for grader calibration

## Files to Modify

1. `src/bioagenteval/harness.py` — Add TrialHook Protocol
2. `src/bioagenteval/runner.py` — Integrate TrialHook, add trial timeout
3. `tests/test_runner.py` — Tests for hook integration and timeout

## Implementation Steps

### Step 1: Add TrialHook Protocol to harness.py

- Define `TrialHook` Protocol with `setup(task: Task) -> None` and `teardown(task: Task, trial: TrialResult) -> None`
- Runtime-checkable, same pattern as AgentHarness

### Step 2: Integrate TrialHook into EvalRunner

- Add optional `hooks: list[TrialHook]` parameter to EvalRunner.__init__
- Call `hook.setup(task)` before agent.run() in _run_trial
- Call `hook.teardown(task, trial)` after grading completes in _run_trial
- Wrap hook calls in try/except to not break trial execution

### Step 3: Add trial timeout to EvalRunner

- Add `default_timeout: float | None = None` to EvalRunner.__init__
- In _run_trial, check task.metadata.get("timeout", self.default_timeout)
- If timeout set, wrap agent.run() in concurrent.futures with timeout
- On timeout, set error="Trial timed out after {N}s"

### Step 4: Create transcript_analysis.py

- `summarize_transcript(transcript) -> dict`: event counts by type, total events, duration if available
- `extract_tool_sequence(transcript) -> list[str]`: ordered tool/event names
- `detect_retries(transcript, threshold=2) -> list[dict]`: find consecutive identical tool calls

### Step 5: Create calibration.py

- `CalibrationResult` dataclass: accuracy, precision, recall, confusion_matrix, details
- `calibrate_grader(grader, examples) -> CalibrationResult`: runs grader on labeled examples
- Each example: dict with task, outcome, transcript, config, expected_passed
- Computes TP/FP/TN/FN and derives accuracy/precision/recall

### Step 6: Write tests

- test_runner.py: hook setup/teardown called, hook error doesn't break trial, timeout triggers error
- test_transcript_analysis.py: summarize, extract_tool_sequence, detect_retries
- test_calibration.py: perfect grader, imperfect grader, empty examples

### Step 7: Run full test suite

Verify all 277 existing tests + new tests pass.
