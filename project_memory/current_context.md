# Current Context

## Active Milestone

**Name**: Robustness & Concurrent Execution
**Goal**: Add async parallel execution, retry logic, convergence analysis, and negative test support for production-grade eval robustness

## Current Phase

**Phase**: execute
**Started**: 2026-02-25

## Key Decisions

- Parallel: ThreadPoolExecutor with max_concurrency param on EvalRunner
- Retry: exponential backoff in grading loop, max_retries param on runner
- Convergence: Wilson score CI width per task in EvalResult
- Negative tests: should_fail bool on Task, invert pass/fail when True

## Blockers

## Plan Reference

### Steps

1. [ ] Add should_fail to Task model + loader support
2. [ ] Add max_concurrency and max_retries to EvalRunner
3. [ ] Implement parallel run_suite with ThreadPoolExecutor
4. [ ] Implement retry logic in grading loop
5. [ ] Handle should_fail in pass_at_k and pass_hat_k
6. [ ] Add convergence_check to EvalResult
7. [ ] Add convergence info to reporter
8. [ ] Write tests
9. [ ] Update docs

## Notes
