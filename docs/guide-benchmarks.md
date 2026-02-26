# Customizing Benchmarks

This guide explains how to create, configure, and manage evaluation task suites in BioAgentEval. A benchmark is defined as a YAML task suite file in the `tasks/` directory. Each file is self-contained and can be loaded, validated, and run independently.

For configuring graders referenced in your tasks, see [Customizing Graders](guide-graders.md). For building agents to run against your benchmarks, see [Customizing Agent Baselines](guide-agents.md).

---

## Task Suite YAML Structure

Every benchmark is a single YAML file with this structure:

```yaml
name: my_benchmark                     # Required — unique suite name
description: >-                        # Optional — human-readable description
  What this benchmark evaluates.
eval_type: capability                  # Optional — "" | "capability" | "regression"
default_num_trials: 3                  # Optional — default trials per task
default_graders:                       # Optional — applied to all tasks
  - type: code
  - type: model
    rubric: Default rubric text
    weight: 1.0
default_tracked_metrics:               # Optional — default metrics for all tasks
  - type: transcript
    metrics: [n_turns, n_tool_calls]
  - type: latency
    metrics: [time_to_last_token]

tasks:
  - id: unique_task_id                 # Required — unique across all suites
    question: >-                       # Required — the prompt sent to the agent
      Ask the agent to do something.
    expected_output:                   # Optional — deterministic checks
      - type: mcq_answer
        value: B
    graders:                           # Optional — overrides default_graders
      - type: code
      - type: model
        rubric: Task-specific rubric
    tags:                              # Optional — arbitrary key-value metadata
      benchmark: My Benchmark
      category: Subcategory
      difficulty: medium
    num_trials: 5                      # Optional — overrides default_num_trials
    metadata:                          # Optional — arbitrary metadata
      timeout: 120                     #   per-task timeout in seconds
      source: original_dataset
    should_fail: false                 # Optional — invert pass/fail (default false)
    tracked_metrics:                   # Optional — overrides default_tracked_metrics
      - type: cost
        metrics: [estimated_cost]
```

### Field Inheritance

Fields cascade from suite level to task level:

| Field | Suite Default | Task Override |
|---|---|---|
| `num_trials` | `default_num_trials` | `num_trials` on the task |
| `graders` | `default_graders` | `graders` on the task |
| `tracked_metrics` | `default_tracked_metrics` | `tracked_metrics` on the task |

If a task defines its own `graders`, the suite `default_graders` are **not** applied to that task. The task's graders completely replace the defaults.

---

## Expected Output Types

The `expected_output` list defines deterministic checks run by the `CodeGrader`. Each item has a `type` and a `value` whose format depends on the type. The final code grader score is the average across all expected output checks.

### Text Matching

**`mcq_answer`** — Multiple-choice answer validation.

```yaml
expected_output:
  - type: mcq_answer
    value: B                          # The correct answer letter
```

Accepts flexible formats in the agent's output: "B", "The answer is B", "(B)", "Answer: B".

**`exact_match`** — Normalized string comparison.

```yaml
expected_output:
  - type: exact_match
    value:
      answer: "42"                    # The expected answer string
      case_sensitive: false           # Default: false
      strip_whitespace: true          # Default: true
```

Also matches patterns like "Answer: 42" or the answer on its own line. When `value` is a plain string instead of a dict, it is treated as the answer with default options.

**`keyword_coverage`** — Topic/keyword presence check.

```yaml
expected_output:
  - type: keyword_coverage
    value:
      keywords:                       # List of terms to search for
        - apoptosis
        - caspase
        - mitochondria
      match_mode: substring           # "substring" (default) or "regex"
```

Returns the fraction of keywords found (case-insensitive substring match by default). A score of 0.5 means half the keywords appeared.

### Numeric Checks

**`numeric_range`** — Value within a min/max range.

```yaml
expected_output:
  - type: numeric_range
    value:
      target: 3.14                    # Exact match (optional)
      min: 3.0                        # Lower bound (optional)
      max: 3.5                        # Upper bound (optional)
```

Extracts all numbers from the agent's output and checks if any fall within bounds.

**`numeric_tolerance`** — Value within absolute or relative tolerance.

```yaml
expected_output:
  - type: numeric_tolerance
    value:
      expected: 0.85                  # Target value
      abs_tol: 0.1                    # Absolute tolerance (optional)
      rel_tol: 0.05                   # Relative tolerance (optional)
```

Extracts numbers (including scientific notation like `1.5e-3`) and finds the closest match. If either tolerance is satisfied, the check passes.

### Structured Output

**`json_schema`** — JSON Schema validation.

```yaml
expected_output:
  - type: json_schema
    value:
      type: object
      properties:
        gene:
          type: string
        organism:
          type: string
      required: [gene]
```

Parses the agent's output as JSON and validates against the schema (Draft 7).

**`entities`** — Entity mention detection.

```yaml
expected_output:
  - type: entities
    value:
      - TP53
      - BRCA1
      - EGFR
```

Returns the fraction of entities found in the outcome (case-insensitive substring match).

**`set_similarity`** — Jaccard similarity between sets.

```yaml
expected_output:
  - type: set_similarity
    value:
      expected:                       # Ground truth items
        - CD4+ T cells
        - B cells
        - NK cells
      separator: ","                  # How to split the outcome (default: comma/newline)
```

Splits the agent's output by the separator to extract predicted items. Returns the Jaccard coefficient: `|intersection| / |union|`.

**`precision_at_k`** — Precision in the top-K of a ranked list.

```yaml
expected_output:
  - type: precision_at_k
    value:
      expected: [TP53, BRCA1, EGFR]  # Ground truth items
      k: 5                            # Top-K cutoff (default: len(expected))
      separator: ","                  # How to split the outcome
```

### Transcript-Based Checks

These checks examine the agent's execution trajectory, not just the final output.

**`tool_calls`** — Verify expected tools were used.

```yaml
expected_output:
  - type: tool_calls
    value:
      - tool_name: search_literature
      - tool_name: run_analysis
        params:                       # Optional parameter matching
          method: deseq2
```

Looks for `tool_call`, `tool_use`, or `cypher_query` events in the transcript. Returns the fraction of expected tools found.

**`turn_limit`** — Enforce maximum LLM turns.

```yaml
expected_output:
  - type: turn_limit
    value:
      max_turns: 5
```

Counts `llm_call` events. Returns 1.0 if within limit, 0.0 if exceeded.

**`trajectory_pattern`** — Ordered event type matching.

```yaml
expected_output:
  - type: trajectory_pattern
    value:
      - llm_call
      - cypher_query
      - llm_call
```

Matches event types in order against the transcript, consuming events as they match.

**`cypher_patterns`** — Regex matching on Cypher queries.

```yaml
expected_output:
  - type: cypher_patterns
    value:
      - "MATCH.*Gene.*TP53"
      - "RETURN.*name"
```

Concatenates all `cypher_query` events and matches each regex. Returns the fraction matched.

### Code Checks

**`code_valid`** — Syntax validation.

```yaml
expected_output:
  - type: code_valid
    value:
      language: python                # Currently only "python" is supported
```

Extracts code from markdown fenced blocks (` ```python ... ``` `) and validates with `ast.parse()`.

**`test_results`** — Test execution outcomes.

```yaml
expected_output:
  - type: test_results
    value:
      expected_tests:                 # Specific tests that must pass
        - test_loading
        - test_analysis
      # OR:
      min_pass_rate: 0.9              # Minimum fraction of tests passing
```

Looks for `test_result` events in the transcript with `data.passed` and `data.test_name`.

### Research Checks

**`groundedness`** — Citation/reference presence.

```yaml
expected_output:
  - type: groundedness
    value:
      min_citations: 3                # Minimum number of citations
      # OR:
      required_sources:               # Specific URLs/DOIs to find
        - "10.1038/s41586-025-09962-4"
```

Detects URLs, DOIs, numbered references (`[1]`), and "Author et al., YYYY" patterns.

### Agent State Checks

**`state_check`** — Verify assertions against state snapshots.

```yaml
expected_output:
  - type: state_check
    value:
      assertions:
        patient.name: "John Doe"
        ui.screen: "results"
```

Reads the last `state_snapshot` event in the transcript. Supports dot-notation for nested access.

---

## Combining Multiple Expected Outputs

A task can have multiple expected outputs. The code grader averages all check scores:

```yaml
expected_output:
  - type: exact_match              # Check 1: did they get the number right?
    value:
      answer: "30"
  - type: numeric_tolerance        # Check 2: is the number close enough?
    value:
      expected: 30
      abs_tol: 0.5
  - type: keyword_coverage         # Check 3: did they explain the reasoning?
    value:
      keywords: [competitive, inhibitor, Km]
```

If exact_match scores 1.0, numeric_tolerance scores 1.0, and keyword_coverage scores 0.67, the final code grader score is `(1.0 + 1.0 + 0.67) / 3 = 0.89`. The pass threshold is 0.5.

---

## Configuring Graders Per Task

Each task specifies which graders to apply. The two most common are `code` (deterministic) and `model` (LLM-based):

```yaml
graders:
  - type: code                      # Runs CodeGrader against expected_output
  - type: model                     # Runs ModelGrader with the rubric
    rubric: >-
      Expected answer: B.
      Scoring: 1.0 = correct, 0.5 = partially correct, 0.0 = incorrect.
    weight: 2.0                      # This grader counts double
```

Grader weights affect the final `weighted_score()` for each trial. A trial passes when `weighted_score() >= 0.5`.

For a full reference of grader types and how to create custom ones, see [Customizing Graders](guide-graders.md).

---

## Tags and Filtering

Tags are arbitrary key-value pairs for classifying and filtering tasks:

```yaml
tags:
  benchmark: LAB-Bench
  category: LitQA2
  subject: Cell Biology
  answer_type: mcq
  difficulty: medium
```

Filter tasks at the CLI:

```bash
# Run only easy MCQ tasks
bioagenteval run tasks/lab_bench.yaml \
  -a myagent:MyAgent \
  --tags difficulty=easy \
  --tags answer_type=mcq
```

Or in Python:

```python
from bioagenteval.loader import load_suite, filter_tasks_by_tags

suite, tasks = load_suite("tasks/lab_bench.yaml")
easy_tasks = filter_tasks_by_tags(tasks, {"difficulty": "easy"})
```

**Recommended tags** for biology benchmarks:

| Tag | Purpose | Example Values |
|---|---|---|
| `benchmark` | Source benchmark | `LAB-Bench`, `BixBench` |
| `category` | Subcategory | `LitQA2`, `SequenceAnalysis` |
| `subject` | Scientific domain | `Cell Biology`, `Genomics` |
| `answer_type` | Response format | `mcq`, `short_answer`, `code`, `pipeline` |
| `difficulty` | Estimated difficulty | `easy`, `medium`, `hard` |

---

## Tracked Metrics

Configure which metrics to compute per trial:

```yaml
default_tracked_metrics:
  - type: transcript
    metrics:
      - n_turns                      # Count of LLM calls
      - n_tool_calls                 # Count of tool/query events
      - n_total_tokens               # Sum of all token counts
  - type: latency
    metrics:
      - time_to_first_token          # ms to first LLM response event
      - output_tokens_per_sec        # Completion tokens / second
      - time_to_last_token           # Total duration in ms
  - type: cost
    metrics:
      - estimated_cost               # USD estimate from token counts
```

Metrics are computed from the agent's transcript and stored in `TrialResult.metrics`. You can register custom metrics — see [Customizing Graders](guide-graders.md#custom-metrics).

---

## Suite Lifecycle

### Capability vs. Regression

Set `eval_type` to classify your suite:

- **`capability`** — Active evaluation suite. Tasks are diagnostic and may have low pass rates. Use this while developing.
- **`regression`** — Saturated suite (near 100% pass rate). Guards against regressions. Promote a capability suite to regression when it's no longer diagnostic.

### Promoting Suites

When your agent consistently passes a capability suite (pass@1 >= 95%), promote it:

```bash
bioagenteval promote tasks/my_suite.yaml eval_report.json
```

This changes `eval_type` from `capability` to `regression` in the YAML file.

### Checking Suite Balance

Verify your suite has balanced coverage across tags:

```bash
bioagenteval check-balance tasks/my_suite.yaml
```

Flags warnings when any tag value represents less than 10% or more than 50% of tasks (when there are 3+ distinct values).

### Generating Tasks from Failures

After a run, generate task stubs for failed cases:

```bash
bioagenteval generate-task eval_report.json tasks/new_tasks.yaml
```

Creates placeholder tasks with `_v2` suffixed IDs for tasks with pass@1 < 1.0.

---

## Negative Test Cases

Use `should_fail: true` for tasks that test error detection or rejection:

```yaml
tasks:
  - id: invalid_query_rejection
    question: "Find the gene that causes happiness."
    should_fail: true                  # Agent should refuse or flag uncertainty
    expected_output:
      - type: keyword_coverage
        value:
          keywords: [uncertain, cannot, no evidence]
    graders:
      - type: code
```

When `should_fail` is true, the runner inverts the score: `score = 1.0 - score` and `passed = not passed`. This means a high original score (agent matched the keywords) becomes a low inverted score — and vice versa. Design the expected_output so that a "correct" failure response matches the checks.

---

## Validation and Testing

### CLI Validation

```bash
bioagenteval validate tasks/my_suite.yaml
```

Loads the YAML, parses all models, and lists each task with its graders, expected outputs, and tags.

### Programmatic Loading

```python
from bioagenteval.loader import load_suite

suite, tasks = load_suite("tasks/my_suite.yaml")
print(f"Suite: {suite.name}, Tasks: {len(tasks)}")
for task in tasks:
    print(f"  {task.id}: {[eo.type for eo in task.expected_output]}")
```

### Writing Tests for Your Suite

Create a test file to verify your suite loads and grades correctly:

```python
from pathlib import Path
from bioagenteval.graders.code_grader import CodeGrader
from bioagenteval.loader import load_suite
from bioagenteval.models import GraderConfig, Transcript

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"

def test_my_suite_loads():
    suite, tasks = load_suite(TASKS_DIR / "my_suite.yaml")
    assert suite.name == "my_suite"
    assert len(tasks) > 0

def test_correct_answer_scores_high():
    _, tasks = load_suite(TASKS_DIR / "my_suite.yaml")
    task = tasks[0]
    grader = CodeGrader()
    config = GraderConfig(type="code")
    transcript = Transcript(task_id=task.id)
    result = grader.grade(task, "The answer is B", transcript, config)
    assert result.score == 1.0
```

---

## Complete Example: Creating a New Benchmark Suite

Here is a minimal but complete example of creating a benchmark for protein function prediction:

```yaml
# tasks/protein_function.yaml
name: protein_function
description: Predict protein function from sequence descriptions.
eval_type: capability
default_num_trials: 3
default_graders:
  - type: code
  - type: model
    rubric: >-
      Evaluate whether the predicted function is correct and well-justified.
      1.0 = correct function with supporting evidence.
      0.5 = partially correct or correct without justification.
      0.0 = incorrect prediction.
default_tracked_metrics:
  - type: transcript
    metrics: [n_turns, n_tool_calls]
  - type: latency
    metrics: [time_to_last_token]

tasks:
  - id: pf_kinase_identification
    question: >-
      Given a protein with the conserved motif HRDLKxxN in its catalytic domain,
      what is the most likely protein family and enzymatic function?
    expected_output:
      - type: exact_match
        value:
          answer: kinase
          case_sensitive: false
      - type: keyword_coverage
        value:
          keywords: [phosphorylation, ATP, catalytic, serine/threonine]
    tags:
      benchmark: ProteinFunction
      category: FamilyPrediction
      subject: Proteomics
      answer_type: short_answer
      difficulty: medium

  - id: pf_enzyme_commission
    question: >-
      Classify the following reaction into an EC number category:
      ATP + protein -> ADP + phosphoprotein
      What is the first-level EC classification?
    expected_output:
      - type: mcq_answer
        value: "2"
    graders:
      - type: code
      - type: model
        rubric: >-
          Expected: EC 2 (Transferases). The reaction transfers a phosphate group
          from ATP to a protein, which is a transferase activity.
          1.0 = correctly identifies EC 2 / Transferases.
          0.5 = identifies the reaction type but wrong EC number.
          0.0 = incorrect classification.
    tags:
      benchmark: ProteinFunction
      category: ECClassification
      subject: Enzymology
      answer_type: short_answer
      difficulty: easy
```

Run it:

```bash
# Validate the suite
bioagenteval validate tasks/protein_function.yaml

# Run against your agent (code grader only, no API calls)
bioagenteval run tasks/protein_function.yaml \
  -a myagent:MyAgent \
  --skip-model-grader \
  -o results/protein_function_report.json

# Compare with a previous run
bioagenteval diff results/baseline.json results/protein_function_report.json
```
