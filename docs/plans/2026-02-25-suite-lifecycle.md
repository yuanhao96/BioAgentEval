# Suite Lifecycle & Eval-Driven Development

**Date**: 2026-02-25
**Milestone**: Suite Lifecycle & Eval-Driven Development

## Summary

Add suite lifecycle management: validate eval_type values, CLI `promote` command for graduating saturated suites, CLI `generate-task` command for scaffolding tasks from failures, and suite balance checking.

## Files to Create

1. `src/bioagenteval/suite_manager.py` — Suite lifecycle logic (promote, generate tasks, balance check)
2. `tests/test_suite_manager.py` — Tests for suite manager

## Files to Modify

1. `src/bioagenteval/models.py` — Add Literal validation for eval_type
2. `src/bioagenteval/__main__.py` — Add promote, generate-task, check-balance CLI commands
3. `tests/test_models.py` — Test eval_type validation

## Implementation Steps

### Step 1: Add eval_type validation to EvalSuite model

Update `models.py`:
- Change `eval_type: str = ""` to `eval_type: Literal["", "capability", "regression"] = ""`
- Ensure backward compatibility with YAML loading

### Step 2: Create suite_manager.py with promote logic

Create `src/bioagenteval/suite_manager.py`:
- `promote_suite(suite_path, report_path) -> dict`: Reads report, checks saturation, updates YAML eval_type from "capability" to "regression". Returns status dict with success/failure info.
- Validation: suite must be "capability" type, report must show saturation (pass@1 >= 95%)

### Step 3: Add generate-task logic to suite_manager.py

- `generate_tasks_from_failures(report_path, output_path)`: Reads report JSON, finds tasks with pass@1 < 1.0, generates YAML stubs for new tasks.
- Each stub includes: id (original_id + "_v2"), question (from failed task), placeholder expected_output and graders, metadata noting the original failure.

### Step 4: Add balance checker to suite_manager.py

- `check_suite_balance(tasks) -> dict`: Analyzes tag distributions across tasks.
- For each tag key, compute value counts and flag imbalances.
- Imbalance: any tag value with <10% or >50% of tasks when there are 3+ distinct values.
- Returns structured report with warnings.

### Step 5: Add CLI commands

Add to `__main__.py`:
- `promote` command: takes suite_path and report_path, calls promote_suite()
- `generate-task` command: takes report_path and output_path, calls generate_tasks_from_failures()
- `check-balance` command: takes suite_path, calls check_suite_balance()

### Step 6: Write tests

- test_suite_manager.py:
  - TestPromoteSuite: saturated capability → regression, non-saturated stays, non-capability rejected
  - TestGenerateTasksFromFailures: generates stubs for failed tasks, handles all-pass report
  - TestCheckSuiteBalance: balanced suite, imbalanced suite, single-tag suite

### Step 7: Run full test suite

Verify all 259 existing tests + new tests pass.
