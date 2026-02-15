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

Supported types: `entities`, `cypher_patterns`, `mcq_answer`, `numeric_range`.

### Graders

| Type | What it does | API key needed |
|------|-------------|----------------|
| `code` | Deterministic checks dispatched by `expected_output` type: entity presence, Cypher pattern matching, MCQ answer, numeric range | No |
| `model` | LLM-based rubric scoring (includes expected output and execution metrics in prompt) | `OPENAI_API_KEY` |
| `human` | Stub that flags results for manual review | No |

### Metrics

**Evaluation metrics:**
- **pass@k** — unbiased combinatorial estimator: `1 - C(n-c, k) / C(n, k)`
- **mean_score** — per-grader-type average across trials
- **overall_pass_at_1** — headline metric averaged across all tasks

**Execution metrics** (computed per trial via `tracked_metrics`):
- `n_turns`, `n_tool_calls`, `n_total_tokens` — transcript-derived counts
- `time_to_first_token`, `time_to_last_token`, `output_tokens_per_sec` — latency metrics
- Custom metrics via `@register_metric` decorator

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
tests/                 # 108 tests (all mocked, no API calls)
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
