# Project Progress

## Goal Summary

Extend BioAgentEval's features (new graders, metrics) and improve robustness to support all agent evaluation applications described in Anthropic's "Demystifying Evals for AI Agents" guide. This includes: multi-provider model grading (Anthropic + OpenAI), LLM judge enhancements (pairwise comparison, multi-judge consensus, uncertainty handling), agent-type-specific evaluation patterns (coding, research, conversational, computer-use), suite lifecycle management (capability vs regression classification, saturation-based promotion), and robustness/observability improvements (trial isolation, transcript analysis, grader calibration).

## Completed Milestones

### Milestone: Multi-Provider Model Grader + LLM Judge Enhancements
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added LLMClient abstraction supporting both OpenAI and Anthropic providers, LLM judge "Unknown" escape hatch, PairwiseGrader for comparing agent outputs, and ConsensusGrader for multi-judge majority vote aggregation.
- **Acceptance criteria met**:
  - [x] Model grader supports both `anthropic` and `openai` providers via config
  - [x] LLM judge can return "Unknown" when insufficient info (not penalizing agent unfairly)
  - [x] Pairwise comparison grader compares two agent outputs and picks the better one
  - [x] Multi-judge consensus runs N LLM judges and aggregates via majority vote
  - [x] All new features have tests; existing 181 tests still pass (now 224 total)

### Milestone: Agent-Type Evaluation Patterns
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added 5 new CodeGrader check types for agent-type-specific evaluation: code_valid and test_results (coding agents), groundedness and keyword_coverage (research agents), state_check (conversational/computer-use agents). Added 3 example tasks and 35 new tests.
- **Acceptance criteria met**:
  - [x] New `expected_output` types for agent-type-specific checks with corresponding CodeGrader implementations
  - [x] Example tasks demonstrating each agent type's evaluation pattern
  - [x] Documentation of when to use each check type
  - [x] All new checks have dedicated tests (259 total, 0 regressions)

### Milestone: Suite Lifecycle & Eval-Driven Development
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added eval_type Literal validation, suite_manager.py with promote_suite(), generate_tasks_from_failures(), and check_suite_balance(). Added 3 CLI commands (promote, generate-task, check-balance). 18 new tests, 277 total, 0 regressions.
- **Acceptance criteria met**:
  - [x] `EvalSuite` model validates `eval_type` field via `Literal["", "capability", "regression"]`
  - [x] CLI `promote` command graduates saturated capability suites to regression
  - [x] CLI `generate-task` command scaffolds new tasks from failure reports
  - [x] Suite balance checker warns about class imbalance in tags/difficulty
  - [x] Tests for all new functionality (18 new tests)

### Milestone: Robustness, Isolation & Observability
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added TrialHook Protocol for trial isolation (setup/teardown), transcript analysis utilities (summarize, extract tool sequence, detect retries), grader calibration workflow (accuracy/precision/recall/confusion matrix), and trial timeout support (per-task and runner-level). 25 new tests, 302 total, 0 regressions.
- **Acceptance criteria met**:
  - [x] Trial isolation with environment setup/teardown hooks (TrialHook Protocol)
  - [x] Transcript analysis utilities for debugging agent trajectories (3 functions)
  - [x] Grader calibration workflow for validating grader accuracy (calibrate_grader)
  - [x] Trial timeout support (per-task via metadata, runner-level default)
  - [x] Tests for all new functionality (25 new tests)

## Project Complete

All 4 milestones completed. 302 tests pass. The eval harness now supports all major patterns from Anthropic's "Demystifying Evals for AI Agents" guide.
