# BioAgentEval

Evaluation harness for biomedical knowledge-graph QA agents. Measures answer quality through deterministic checks, LLM-based rubric scoring, and human review, with multi-trial pass@k metrics, execution metrics, and full trajectory capture.

Built following [Anthropic's "Demystifying Evals for AI Agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Quick Start

```bash
git clone git@github.com:yuanhao96/BioAgentEval.git
cd BioAgentEval
pip install -e ".[dev]"
```

Validate a task suite:

```bash
bioagenteval validate tasks/biomedical_core.yaml
```

Run an evaluation (code grading only, no API keys needed):

```bash
bioagenteval run tasks/biomedical_core.yaml \
  --agent bioagenteval.agents.baseline_qa:BaselineQAAgent \
  --skip-model-grader \
  --output results/report.json
```

Filter by tags (run only simple tasks):

```bash
bioagenteval run tasks/biomedical_core.yaml \
  --agent bioagenteval.agents.baseline_qa:BaselineQAAgent \
  --skip-model-grader \
  --tags complexity=simple
```

Compare two reports:

```bash
bioagenteval diff results/report_v1.json results/report_v2.json
```

Run with all graders (requires `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`):

```bash
bioagenteval run tasks/biomedical_core.yaml \
  --agent bioagenteval.agents.baseline_qa:BaselineQAAgent \
  --output results/report.json
```

## How It Works

```
EvalSuite (named group of tasks)
  └── Task (question + expected_output + tags + graders)
        └── EvalResult (aggregated across trials)
              └── TrialResult (one attempt + metrics)
                    ├── Transcript → TranscriptEvent (tool calls, Cypher queries, ...)
                    └── GradeResult (output from one grader)
```

1. **Define tasks** in YAML with questions, typed expected outputs, tags, and grader configs
2. **Wrap your agent** by implementing `run(question) -> AgentResponse` and `reset()` (no base class needed)
3. **Run the harness** — it executes multiple trials per task, computes metrics, grades each trial, and calculates pass@k
4. **Read the report** — structured JSON with per-task scores, execution metrics, and overall summary

### Task definition

Tasks use typed `expected_output` items:

```yaml
tasks:
  - id: gene_diabetes_association
    question: "What genes are associated with type 1 diabetes?"
    expected_output:
      - type: entities
        value: [INS, HLA-DRB1, PTPN22]
      - type: cypher_patterns
        value: ["MATCH.*Gene"]
    tags:
      complexity: complex
    graders:
      - type: code
      - type: model
        rubric: "Is the answer accurate and complete?"
```

Supported types: `entities`, `cypher_patterns`, `mcq_answer`, `numeric_range`, `json_schema`, `tool_calls`, `turn_limit`, `trajectory_pattern`.

#### JSON Schema validation

Use `json_schema` to validate that an agent's response is valid JSON conforming to a schema:

```yaml
  - id: gene_structured_output
    question: "Return a JSON object describing the TP53 gene."
    expected_output:
      - type: json_schema
        value:
          type: object
          properties:
            symbol:
              type: string
            name:
              type: string
          required:
            - symbol
            - name
    graders:
      - type: code
```

The grader parses the agent's outcome as JSON, validates it against the schema (JSON Schema Draft 7), and returns score 1.0 if valid or 0.0 with specific validation errors in `GradeResult.details`.

#### Trajectory grading

Grade agent behavior (not just outcomes) using transcript-based checks:

```yaml
  - id: gene_query_trajectory
    question: "Query the knowledge graph for TP53 gene details."
    expected_output:
      - type: tool_calls
        value:
          - tool_name: cypher_query
            params: {}              # optional: verify specific parameters
      - type: turn_limit
        value:
          max_turns: 5
      - type: trajectory_pattern
        value:
          - "llm_call"             # regex matched against event_types in order
          - "cypher_query"
          - "llm_.*"
    graders:
      - type: code
```

- **tool_calls**: Verifies expected tools were called (with optional param matching). Returns fraction found.
- **turn_limit**: Enforces max LLM turns. Returns 1.0 if within limit, 0.0 if exceeded.
- **trajectory_pattern**: Ordered regex matching on event_type sequence. Returns fraction of patterns matched.

### Graders

| Type | What it does | API key needed |
|------|-------------|----------------|
| `code` | Deterministic checks dispatched by `expected_output` type: entity presence, Cypher pattern matching, MCQ answer, numeric range, JSON Schema validation, tool call verification, turn limit, trajectory pattern | No |
| `model` | LLM-based rubric scoring (includes expected output and execution metrics in prompt) | `OPENAI_API_KEY` |
| `human` | Stub that flags results for manual review | No |

### Grader Weights

Graders support a `weight` field (default 1.0) for weighted score aggregation:

```yaml
    graders:
      - type: code
        weight: 2.0    # Deterministic check weighted 2x
      - type: model
        weight: 1.0
```

Use `TrialResult.weighted_score()` for weighted average and `weighted_passed(threshold)` for pass/fail.

### Metrics

**Evaluation metrics:**
- **pass@k** — unbiased combinatorial estimator: `1 - C(n-c, k) / C(n, k)`. Probability of at least 1 pass in k trials.
- **pass^k** (`pass_hat_k`) — consistency estimator: `C(c, k) / C(n, k)`. Probability that ALL k trials pass. Critical for customer-facing agents.
- **mean_score** — per-grader-type average across trials
- **overall_pass_at_1** / **overall_pass_hat_1** — headline metrics averaged across all tasks
- **latency_p50_ms** / **latency_p95_ms** — latency percentiles across all trials in report summary

**Execution metrics** (computed per trial via `tracked_metrics`):
- `n_turns`, `n_tool_calls`, `n_total_tokens` — transcript-derived counts
- `time_to_first_token`, `time_to_last_token`, `output_tokens_per_sec` — latency metrics
- `estimated_cost` — USD cost estimate based on token counts (default GPT-4o-class pricing)
- Custom metrics via `@register_metric` decorator

### Eval Suite Management

**Eval types**: Suites can declare `eval_type: capability` or `eval_type: regression` in YAML. This is included in reports for tracking.

**Saturation detection**: When a suite's overall pass@1 reaches 95%+, the report flags it as `saturated` with a note recommending promotion to regression and creation of harder tasks.

**Tag filtering**: Use `--tags key=value` (repeatable) on `run` and `validate` commands to select task subsets.

**Run comparison**: `bioagenteval diff report_a.json report_b.json` shows per-task pass@1 deltas.

### Robustness

**Parallel execution**: Set `max_concurrency` on `EvalRunner` to run tasks in parallel using thread pool:

```python
runner = EvalRunner(agent=agent, graders=graders, max_concurrency=4)
```

**Retry logic**: Set `max_retries` for exponential backoff on grading failures (useful for API-backed graders):

```python
runner = EvalRunner(agent=agent, graders=graders, max_retries=2)
```

**Convergence analysis**: Each task report includes a `convergence` field with Wilson score confidence interval. Use `converged: true/false` to decide if more trials are needed.

**Negative test cases**: Set `should_fail: true` on a task to test that the agent correctly fails:

```yaml
  - id: hallucination_check
    question: "What gene causes unicorn syndrome?"
    should_fail: true
    expected_output:
      - type: entities
        value: [FAKE_GENE]
    graders:
      - type: code
```

## Project Structure

```
src/bioagenteval/
  __main__.py          # CLI (bioagenteval run, bioagenteval validate)
  models.py            # Pydantic v2 data models (ExpectedOutput, MetricGroup, Task, ...)
  harness.py           # AgentHarness protocol
  loader.py            # YAML suite loader
  runner.py            # EvalRunner orchestrator (runs trials, computes metrics)
  reporter.py          # JSON report generation (includes metrics per trial)
  metrics.py           # Metric registry and built-in metrics
  graders/             # CodeGrader, ModelGrader, HumanGrader
  agents/              # BaselineQAAgent (GPT-based)
tasks/
  biomedical_core.yaml # Core evaluation suite (v2 format)
  hle_bio_chem.yaml    # HLE Bio/Chem 149-task suite
scripts/
  convert_hle_bio_chem.py  # Dataset converter (emits v2 format)
tests/                 # 181 tests (all mocked, no API calls)
```

## Documentation

See [docs/guide.md](docs/guide.md) for the full reference covering YAML task schema, expected output types, execution metrics, custom agent/grader implementation, CLI options, and data models.

## Development

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_models.py

# Verbose
pytest -v
```

## Requirements

- Python >= 3.9
- pydantic >= 2.0
- click >= 8.0
- PyYAML >= 6.0
- anthropic >= 0.66.0 (for model grading)
- openai >= 1.0 (for baseline agent)
- jsonschema >= 4.0 (for JSON Schema validation)
