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

## Milestone: Specialized Grader Types for Benchmark Support (2026-02-25)

### What Worked
- CodeGrader dispatch pattern continues to scale cleanly — 4 new checks added without touching existing logic
- Helper function `_normalize_text()` shared across exact_match and set_similarity avoids duplication
- Jaccard coefficient is the right metric for set_similarity — simple, well-understood, handles edge cases
- Supporting both absolute and relative tolerance in numeric_tolerance covers most benchmark needs
- `_extract_numbers()` regex handles scientific notation (1.5e-3) which is common in biology benchmarks

### What Didn't Work
- Initial `_check_numeric_tolerance` had a SyntaxError: `min(generator, key=...)` requires parenthesized generator expression — caught by test run
- Fixed by splitting into explicit list comprehension then min() call

### Patterns to Reuse
- `_normalize_text(text, case_sensitive, strip_whitespace)` helper — reusable for any text comparison check
- `_extract_numbers()` regex for finding numeric values in free-text agent output
- Separator-based set splitting (comma, semicolon, newline) for parsing list-style outputs

### Patterns to Avoid
- code_grader.py is now ~550 lines — approaching the ~600 line threshold from M2 lesson. Consider splitting if M6/M7 add more checks

## Milestone: Text-Based Benchmark Suites (2026-02-25)

### What Worked
- Using existing HLE-Bio suite as template made authoring new suites fast
- Separate YAML file per benchmark keeps suites independently manageable
- Validation tests that load real YAML files catch schema mismatches immediately
- Cross-suite consistency test (no duplicate IDs) prevents conflicts

### What Didn't Work
- Initially used wrong value key names: `expected` instead of `answer` for exact_match, `required_keywords`/`optional_keywords` instead of `keywords` for keyword_coverage
- Caught by integration tests — validates the importance of testing against actual grader code, not just YAML structure

### Patterns to Reuse
- `_grade_task()` test helper: loads real task from YAML and runs CodeGrader with canned outcome — quick integration validation
- Standard tag set (benchmark, category, subject, answer_type, difficulty) for all benchmark suites
- Per-subcategory task count assertions (e.g., `test_three_tasks_per_subcategory`) verify suite completeness

### Patterns to Avoid
- Don't assume grader value schemas from their names — always check the actual `_check_X` function signature before writing YAML

## Milestone: Agentic Benchmark Suites (2026-02-25)

### What Worked
- Creating all 6 YAML files in parallel was efficient — no dependencies between benchmarks
- Lesson from M6 (verify value schemas first) paid off — zero schema-related failures
- _make_transcript_with_tools test helper made agentic test writing quick
- Cross-suite ID uniqueness test across ALL suites catches conflicts early

### What Didn't Work
- Nothing major — the M6 lesson about value schemas prevented the main pitfall

### Patterns to Reuse
- Standard benchmark suite template: name, description, eval_type, default_graders, default_tracked_metrics, tasks
- _make_transcript_with_tools: creates mock transcripts for tool_calls testing
- 3 tasks per benchmark is a good representative sample size for harness validation
- Cross-suite tests should always check ALL suites, not just the new ones

### Patterns to Avoid
- Nothing specific — well-established patterns from M6 applied cleanly
