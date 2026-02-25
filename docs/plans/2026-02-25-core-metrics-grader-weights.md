# Plan: Core Metrics & Grader Weight Support

**Date**: 2026-02-25
**Milestone**: Core Metrics & Grader Weight Support

## Overview

Close four core metrics gaps from the Anthropic eval guide:
1. pass^k metric (consistency measurement)
2. Grader weight aggregation (weighted scoring)
3. Cost metric (token pricing estimate)
4. Latency percentiles (p50, p95 in report summary)

## Steps

### Step 1: Add pass^k to EvalResult (models.py)

**File**: `src/bioagenteval/models.py`

Add `pass_hat_k(k)` method to `EvalResult`:
- Formula: `C(c, k) / C(n, k)` where c = passing trials, n = total
- Edge cases: n=0 → 0.0, k<=0 → 0.0, k>n → use k=n, c<k → 0.0

### Step 2: Add weight field to GradeResult, weighted_score to TrialResult (models.py)

**File**: `src/bioagenteval/models.py`

- Add `weight: float = 1.0` to `GradeResult`
- Add `weighted_score() -> float` method to `TrialResult`: weighted average of grade scores using grade weights
- Add `weighted_passed(threshold: float = 0.5) -> bool` to `TrialResult`

### Step 3: Propagate weight from GraderConfig to GradeResult in runner (runner.py)

**File**: `src/bioagenteval/runner.py`

After `grader.grade(...)`, copy `grader_config.weight` onto the returned `GradeResult.weight`.

### Step 4: Add cost metric to registry (metrics.py)

**File**: `src/bioagenteval/metrics.py`

Register `estimated_cost` metric:
- Default pricing: prompt=$3/1M tokens, completion=$15/1M tokens (GPT-4o-class)
- Sum prompt_tokens × prompt_price + completion_tokens × completion_price
- Returns float (USD estimate)

### Step 5: Add latency percentiles to reporter summary (reporter.py)

**File**: `src/bioagenteval/reporter.py`

In `generate_report()`, add to summary:
- `latency_p50_ms`: median duration_ms across all trials
- `latency_p95_ms`: 95th percentile duration_ms
- `overall_pass_hat_1`: pass^1 averaged across tasks (mirrors overall_pass_at_1)

### Step 6: Write tests

**Files**:
- `tests/test_models.py`: Tests for `pass_hat_k`, `weighted_score`, `weighted_passed`
- `tests/test_metrics.py`: Tests for `estimated_cost` metric
- `tests/test_reporter.py`: Tests for latency percentiles and pass^1 in summary
- `tests/test_runner.py`: Test that weight propagates from config to grade result

### Step 7: Update documentation

**Files**:
- `CLAUDE.md`: Mention pass^k, grader weights, cost metric
- `README.md`: Update metrics table, add pass^k and cost metric docs

## Files Modified

1. `src/bioagenteval/models.py` — pass^k, GradeResult.weight, TrialResult.weighted_score
2. `src/bioagenteval/runner.py` — propagate weight
3. `src/bioagenteval/metrics.py` — estimated_cost metric
4. `src/bioagenteval/reporter.py` — latency percentiles, pass^1 summary
5. `tests/test_models.py` — new test classes
6. `tests/test_metrics.py` — cost metric tests
7. `tests/test_reporter.py` — percentile + pass^1 tests
8. `tests/test_runner.py` — weight propagation test
9. `CLAUDE.md` — docs update
10. `README.md` — docs update

## Testing Strategy

- All new methods get dedicated test classes
- Edge cases: empty trials, k>n, all pass, all fail, zero tokens
- Backward compatibility: default weight=1.0 produces same results as before
- Run full `pytest` at the end to verify no regressions
