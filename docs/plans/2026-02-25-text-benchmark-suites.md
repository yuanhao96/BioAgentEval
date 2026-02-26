# Plan: Text-Based Benchmark Suites (FrontierScience-Bio, LAB-Bench)

## Objective

Create task suite YAML files for FrontierScience-Bio and LAB-Bench benchmarks,
and verify the existing HLE-Bio suite works with current graders.

## Steps

1. **Create FrontierScience-Bio task suite** (`tasks/frontierscience_bio.yaml`)
   - 3 Olympiad-style tasks (short answer, exact_match + model judge)
   - 3 Research-style tasks (free-form, model judge + keyword_coverage)
   - Suite metadata: name, description, eval_type, default graders
   - Tags: benchmark, category (Olympiad/Research), subject, answer_type, difficulty

2. **Create LAB-Bench task suite** (`tasks/lab_bench.yaml`)
   - 3 LitQA2 tasks (literature-based MCQ)
   - 3 CloningScenarios tasks (DNA cloning MCQ)
   - 3 ProtocolQA tasks (lab protocol MCQ)
   - 3 SeqQA tasks (sequence analysis MCQ)
   - All use mcq_answer expected_output + model grader rubric
   - Tags: benchmark, category (LitQA2/CloningScenarios/ProtocolQA/SeqQA), subject, answer_type

3. **Add validation tests** (`tests/test_benchmark_suites.py`)
   - Test: each YAML file loads and validates as EvalSuite + Tasks
   - Test: CodeGrader runs correctly on sample task from each suite
   - Test: HLE-Bio suite loads and grader runs on a sample task
   - Test: all tasks have required tags (benchmark, category, answer_type)

4. **Run full test suite** to verify 0 regressions

## Files to create/modify

- `tasks/frontierscience_bio.yaml` (new)
- `tasks/lab_bench.yaml` (new)
- `tests/test_benchmark_suites.py` (new)

## Dependencies

- Milestone 5 (Specialized Grader Types) — completed
- Existing grader types: mcq_answer, exact_match, keyword_coverage, model judge

## Acceptance criteria

- [ ] FrontierScience-Bio task suite YAML with 6 representative sample tasks
- [ ] LAB-Bench task suite YAML with 12 representative sample tasks
- [ ] HLE-Bio suite verified to work with current graders
- [ ] Each suite uses appropriate grader configurations
- [ ] Suite validation passes (tests)
