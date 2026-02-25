# Lessons Learned

## Milestone: Core Metrics & Grader Weight Support (2026-02-25)

### What Worked
- Adding sibling methods (pass_hat_k alongside pass_at_k) keeps the API consistent and intuitive
- Propagating weight from config → result in the runner is clean and backward-compatible (default 1.0)
- Putting percentiles in the reporter summary rather than per-trial metrics was the right call — percentiles need multiple data points

### What Didn't Work
- Nothing significant — milestone was well-scoped

### Patterns to Reuse
- Default values for backward compatibility: adding new fields with sensible defaults (weight=1.0) avoids breaking existing code
- Test helper patterns: _make_result in test_reporter.py is reusable across test files

### Patterns to Avoid
- None identified

## Milestone: Tool Call & Trajectory Grading (2026-02-25)

### What Worked
- Extending CodeGrader dispatch rather than creating new grader classes — consistent with existing architecture
- Using fullmatch on individual event types for trajectory_pattern avoids greedy regex issues
- The transcript already has all the data needed — no model changes required

### What Didn't Work
- Initial trajectory_pattern implementation used joined string matching, which caused greedy regex issues — switched to per-event fullmatch

### Patterns to Reuse
- When adding new expected_output types, just add dispatch + _check_* function — zero changes to runner, reporter, or models
- fullmatch on individual items instead of search on joined strings prevents regex boundary issues

### Patterns to Avoid
- Joining list items into a single string for regex matching — greedy quantifiers cause unexpected behavior

## Milestone: Eval Suite Management & CLI Enhancements (2026-02-25)

### What Worked
- filter_tasks_by_tags as a standalone function (not a method) makes it testable and reusable
- Saturation detection as a simple threshold check in the reporter — minimal code, maximum signal
- The diff command reads raw JSON, doesn't need model objects — simple and robust

### What Didn't Work
- Nothing significant — straightforward feature additions

### Patterns to Reuse
- CLI filtering as post-load step: load full suite, then filter — avoids loader complexity
- Optional report fields: only include eval_type if non-empty, avoids null pollution

### Patterns to Avoid
- None identified

## Milestone: Robustness & Concurrent Execution (2026-02-25)

### What Worked
- ThreadPoolExecutor for parallelism — no async required, agents stay synchronous
- Grade inversion for should_fail — simple score/passed flip, no changes to pass_at_k
- Wilson CI for convergence — well-studied formula, gives actionable CI width

### What Didn't Work
- Wilson CI is conservative at p=0 and p=1 boundaries — needed threshold=0.3 instead of 0.2

### Patterns to Reuse
- Retry with exponential backoff as a wrapper method (_grade_with_retry) keeps the main flow clean
- Preserving task order in parallel results using dict keyed by task.id

### Patterns to Avoid
- Don't use tight convergence thresholds with Wilson CI at boundary pass rates
