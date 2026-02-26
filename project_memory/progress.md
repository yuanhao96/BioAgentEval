# Project Progress

## Goal Summary

Extend BioAgentEval's features (new graders, metrics) and improve robustness to support all agent evaluation applications described in Anthropic's "Demystifying Evals for AI Agents" guide. Additionally, the harness should successfully support all biology benchmarks (each as an individual task suite): HLE-Bio, FrontierScience-Bio, LAB-Bench, Biomni-Eval1, BixBench, BioML-Bench, SpatialBench/scBench, BioAgent Bench.

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

### Milestone: Specialized Grader Types for Benchmark Support
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added 4 new CodeGrader check types for benchmark evaluation: exact_match (normalized string comparison), set_similarity (Jaccard coefficient), precision_at_k (top-K precision for ranked lists), numeric_tolerance (absolute/relative tolerance). 27 new tests, 329 total, 0 regressions.
- **Acceptance criteria met**:
  - [x] `exact_match` check type: normalized string comparison with case/whitespace options
  - [x] `set_similarity` check type: Jaccard coefficient between predicted and expected sets
  - [x] `precision_at_k` check type: top-K precision for ranked gene/feature lists
  - [x] `numeric_tolerance` check type: numeric answer within absolute/relative tolerance
  - [x] Tests for all new check types (27 new tests)

### Milestone: Text-Based Benchmark Suites (FrontierScience-Bio, LAB-Bench)
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Created task suite YAMLs for FrontierScience-Bio (6 tasks: 3 Olympiad with exact_match+numeric_tolerance, 3 Research with keyword_coverage) and LAB-Bench (12 tasks: 3 per subcategory with mcq_answer). Verified HLE-Bio suite loads and grades correctly. 32 new tests, 361 total, 0 regressions.
- **Acceptance criteria met**:
  - [x] FrontierScience-Bio task suite YAML with 6 representative sample tasks (3 Olympiad + 3 Research)
  - [x] LAB-Bench task suite YAML with 12 representative sample tasks (3 LitQA2, 3 CloningScenarios, 3 ProtocolQA, 3 SeqQA)
  - [x] HLE-Bio suite verified to work with current graders (6 tests)
  - [x] Each suite uses appropriate grader configurations (exact_match, numeric_tolerance, keyword_coverage, mcq_answer, model judge)
  - [x] Suite validation passes for all suites (32 new tests)

### Milestone: Agentic Benchmark Suites (Biomni-Eval1, BixBench, BioML-Bench, SpatialBench/scBench, BioAgent Bench)
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Created 6 task suite YAMLs for agentic benchmarks: Biomni-Eval1 (3 tasks: DryLab+WetLab), BixBench (3 tasks: code execution), BioML-Bench (3 tasks: ML pipelines), SpatialBench (3 tasks: spatial transcriptomics), scBench (3 tasks: single-cell analysis), BioAgent Bench (3 tasks: end-to-end pipelines). 36 new tests, 397 total, 0 regressions.
- **Acceptance criteria met**:
  - [x] Biomni-Eval1 task suite YAML with 3 representative tasks (tool_calls + keyword_coverage + model judge)
  - [x] BixBench task suite YAML with 3 representative tasks (code_valid + keyword_coverage)
  - [x] BioML-Bench task suite YAML with 3 representative tasks (code_valid + numeric_tolerance + set_similarity)
  - [x] SpatialBench task suite YAML with 3 representative tasks (set_similarity + keyword_coverage)
  - [x] scBench task suite YAML with 3 representative tasks (set_similarity + code_valid + numeric_tolerance)
  - [x] BioAgent Bench task suite YAML with 3 representative tasks (tool_calls + code_valid + keyword_coverage)
  - [x] All suites use appropriate grader configurations
  - [x] Validation tests pass for all new suites (36 new tests)

## Current Milestone

None — all milestones for the current goal are complete.

## Upcoming Milestones

None.
