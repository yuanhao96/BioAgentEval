# Current Context

## Active Milestone

**Name**: Robustness, Isolation & Observability
**Goal**: Add trial isolation hooks, transcript analysis utilities, grader calibration, and trial timeout.

## Current Phase

**Phase**: execute
**Started**: 2026-02-25

## Key Decisions

- TrialHook as Protocol (same pattern as AgentHarness) with setup(task) and teardown(task, trial) [R1]
- Hooks passed as list to EvalRunner, wrapped in try/except so failures don't break trials [R1]
- Transcript analysis: 3 pure functions in transcript_analysis.py (summarize, extract_tool_sequence, detect_retries) [R1]
- Grader calibration: accuracy/precision/recall + confusion matrix, no Cohen's kappa [R2]
- Trial timeout: default_timeout on runner + per-task override via task.metadata["timeout"] [R2]
- Thread timeout limitation accepted: thread continues but runner unblocks [R2]

## Blockers

<!-- None currently. -->

## Plan Reference

Full plan: `docs/plans/2026-02-25-robustness-observability.md`

### Steps

1. [ ] Add TrialHook Protocol to harness.py
2. [ ] Integrate TrialHook into EvalRunner
3. [ ] Add trial timeout to EvalRunner
4. [ ] Create transcript_analysis.py
5. [ ] Create calibration.py
6. [ ] Write tests
7. [ ] Run full test suite

## Notes

- Lesson from M1: Protocol-based abstraction is clean and testable
- Lesson from M3: single-module approach keeps things simple
- Hook implementations must be thread-safe if max_concurrency > 1
