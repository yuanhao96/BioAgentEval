# Lessons Learned

## Milestone: Multi-Provider Model Grader + LLM Judge Enhancements (2026-02-25)

### What Worked
- Provider abstraction (LLMClient protocol) was the right approach — clean, testable, extensible
- Using Protocol with @runtime_checkable instead of ABC for LLMClient avoids forcing both providers into inheritance
- Lazy imports of openai/anthropic inside __init__ prevent import failures when only one provider is installed
- Sharing a single LLMClient instance across ModelGrader/PairwiseGrader/ConsensusGrader in CLI avoids redundant client creation
- _MockLLMClient test helper made tests much simpler than mocking SDK internals

### What Didn't Work
- Initially duplicated _format_expected_output across model_grader.py and pairwise_grader.py — caught in code review
- Plan specified deterministic scoring for PairwiseGrader (1.0/0.5/0.0) but LLM-flexible scoring is better — plan should have been updated

### Patterns to Reuse
- Protocol-based abstraction for external service clients: keeps implementations independent, simplifies testing
- ConsensusGrader as a decorator/wrapper pattern: any BaseGrader can be wrapped for consensus without modification

### Patterns to Avoid
- Duplicating utility functions across grader files: extract to base.py or a utils module early

## Milestone: Agent-Type Evaluation Patterns (2026-02-25)

### What Worked
- Extending existing CodeGrader dispatch pattern was clean — 5 new checks added without modifying any existing check logic
- Each check function is fully independent (~30-50 lines) — easy to understand and test
- _extract_code_blocks helper reusable for any check that needs to parse markdown fences
- Dot-notation key resolution (_resolve_nested) was a good investment for state_check

### What Didn't Work
- Nothing major — the pattern was well-established from the initial 8 check types

### Patterns to Reuse
- Adding new expected_output types: just add a _check_X function and a dispatch entry in CodeGrader.grade()
- Citation detection regex patterns: reusable for any research-agent evaluation

### Patterns to Avoid
- code_grader.py is now ~450 lines — if it grows beyond ~600, consider splitting into domain-specific modules

## Milestone: Suite Lifecycle & Eval-Driven Development (2026-02-25)

### What Worked
- Single suite_manager.py module for all lifecycle logic kept things simple — 3 functions, ~140 lines total
- Using plain dicts (not Pydantic models) for suite_manager functions simplified YAML reading/writing
- Literal type validation on eval_type catches invalid values at model construction time
- Lazy imports in CLI commands avoid loading suite_manager when not needed

### What Didn't Work
- Nothing major — straightforward feature set, well-scoped milestone

### Patterns to Reuse
- YAML round-trip via yaml.safe_load / yaml.dump for updating suite files in-place
- Helper functions in tests (_write_suite, _write_report) with tmp_path for clean fixture creation
- Imbalance detection threshold (10%/50% with 3+ distinct values) is a reasonable heuristic

### Patterns to Avoid
- Nothing specific to flag for this milestone

## Milestone: Robustness, Isolation & Observability (2026-02-25)

### What Worked
- TrialHook Protocol follows established AgentHarness pattern — consistent with codebase conventions
- Wrapping hook calls in try/except ensures hooks never break trial execution
- Transcript analysis as pure functions (no state, no side effects) makes testing trivial
- CalibrationResult as a dataclass (not Pydantic) keeps it lightweight for a utility output
- Per-task timeout via task.metadata avoids adding model fields while staying flexible

### What Didn't Work
- Nothing major — well-established patterns from prior milestones applied cleanly

### Patterns to Reuse
- Protocol + try/except wrapper for optional lifecycle hooks
- Pure functions on immutable data for analysis utilities
- Confusion matrix approach (TP/FP/TN/FN) for any binary classifier evaluation

### Patterns to Avoid
- Thread-based timeout has a known limitation (thread continues after timeout) — document this for users
