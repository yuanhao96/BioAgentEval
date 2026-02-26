# Current Context

## Active Milestone

**Name**: Agentic Benchmark Suites (Biomni-Eval1, BixBench, BioML-Bench, SpatialBench/scBench, BioAgent Bench)
**Goal**: Create task suite YAMLs for 5 agentic biology benchmarks (6 YAML files total).

## Current Phase

**Phase**: execute
**Started**: 2026-02-25

## Key Decisions

- 3 tasks per benchmark (15 + 3 BioAgent Bench = 18 total tasks, 6 YAML files) [R1]
- SpatialBench and scBench as separate YAML files [R1]
- Grader combos: Biomni (tool_calls+model), BixBench (code_valid+test_results+model), BioML (numeric_tolerance+set_similarity+code_valid), SpatialBench/scBench (set_similarity+numeric_tolerance+exact_match), BioAgent (code_valid+tool_calls+model) [R1]
- No new grader types needed — existing 17 check types cover all patterns [R1]
- Use >- folded block scalars for long questions [R2]
- Verify value schemas against _check_X functions before writing YAML [R2]

## Blockers

<!-- None currently. -->

## Plan Reference

Full plan: `docs/plans/2026-02-25-agentic-benchmark-suites.md`

### Steps

1. [ ] Create Biomni-Eval1 suite (3 tasks)
2. [ ] Create BixBench suite (3 tasks)
3. [ ] Create BioML-Bench suite (3 tasks)
4. [ ] Create SpatialBench suite (3 tasks)
5. [ ] Create scBench suite (3 tasks)
6. [ ] Create BioAgent Bench suite (3 tasks)
7. [ ] Add validation tests
8. [ ] Run full test suite

## Notes

- Lesson from M6: YAML value keys must match _check_X function signatures exactly
- Standard tags: benchmark, category, subject, answer_type, difficulty
