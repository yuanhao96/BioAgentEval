# Project Progress

## Goal Summary

Extend BioAgentEval's features (new graders, metrics) and improve robustness to support all agent evaluation applications described in Anthropic's "Demystifying Evals for AI Agents" guide — including coding agents, conversational agents, research agents, and computer-use agents. This means closing gaps in metrics (pass^k, cost, latency percentiles), grader capabilities (tool call verification, trajectory grading, turn limits), eval suite management (tag filtering, regression/capability classification, run comparison), and execution robustness (concurrency, retries, convergence analysis).

## Completed Milestones

### Milestone: Core Metrics & Grader Weight Support
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added pass^k metric, grader weight aggregation, estimated_cost metric, and latency percentiles (p50/p95) to report summary. 18 new tests, 138 total.
- **Acceptance criteria met**:
  - [x] pass^k metric implemented with tests
  - [x] GraderConfig.weight honored in score aggregation with tests
  - [x] Cost metric (token pricing estimate) added to registry
  - [x] Latency percentile metrics (p50, p95) added to registry
  - [x] Documentation updated (README, CLAUDE.md)

### Milestone: Tool Call & Trajectory Grading
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added tool_calls, turn_limit, and trajectory_pattern expected_output types to CodeGrader. 18 new tests, 156 total.
- **Acceptance criteria met**:
  - [x] tool_call_verification grader implemented with tests
  - [x] turn_limit grader implemented with tests
  - [x] trajectory_pattern grader implemented with tests
  - [x] Example tasks added to biomedical_core.yaml
  - [x] Documentation updated

### Milestone: Eval Suite Management & CLI Enhancements
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added tag filtering (--tags), eval_type classification, saturation detection, and diff CLI command. 13 new tests, 169 total.
- **Acceptance criteria met**:
  - [x] Tag-based task filtering in CLI (--tags complexity=simple)
  - [x] Regression vs. capability eval_type classification
  - [x] Saturation detection in reporting
  - [x] Run comparison (diff two JSON reports)
  - [x] Documentation updated

### Milestone: Robustness & Concurrent Execution
- **Status**: completed
- **Date completed**: 2026-02-25
- **Summary**: Added parallel execution (ThreadPoolExecutor), retry with backoff, Wilson CI convergence analysis, and should_fail negative test support. 12 new tests, 181 total.
- **Acceptance criteria met**:
  - [x] Parallel task execution via max_concurrency
  - [x] Retry logic with exponential backoff for API-backed graders
  - [x] Convergence analysis (Wilson score CI)
  - [x] Negative test case support (should_fail flag)
  - [x] Documentation updated

## Current Milestone

(All planned milestones completed)

## Upcoming Milestones

(None — project goal satisfied)
