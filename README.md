# BioAgentEval

Evaluation harness for biomedical knowledge-graph QA agents. Measures answer quality through deterministic checks, LLM-based rubric scoring, and human review, with multi-trial pass@k metrics and full trajectory capture.

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
  └── Task (question + expected entities + graders)
        └── EvalResult (aggregated across trials)
              └── TrialResult (one attempt)
                    ├── Transcript → TranscriptEvent (tool calls, Cypher queries, ...)
                    └── GradeResult (output from one grader)
```

1. **Define tasks** in YAML with questions, expected entities, and grader configs
2. **Wrap your agent** by implementing `run(question) -> AgentResponse` and `reset()` (no base class needed)
3. **Run the harness** — it executes multiple trials per task, grades each, and computes pass@k
4. **Read the report** — structured JSON with per-task and overall metrics

### Graders

| Type | What it does | API key needed |
|------|-------------|----------------|
| `code` | Deterministic entity presence + Cypher pattern matching | No |
| `model` | LLM-based rubric scoring via Claude | `ANTHROPIC_API_KEY` |
| `human` | Stub that flags results for manual review | No |

### Metrics

- **pass@k** — unbiased combinatorial estimator: `1 - C(n-c, k) / C(n, k)`
- **mean_score** — per-grader-type average across trials
- **overall_pass_at_1** — headline metric averaged across all tasks

## Project Structure

```
src/bioagenteval/
  __main__.py          # CLI (bioagenteval run, bioagenteval validate)
  models.py            # Pydantic v2 data models
  harness.py           # AgentHarness protocol
  loader.py            # YAML suite loader
  runner.py            # EvalRunner orchestrator
  reporter.py          # JSON report generation
  graders/             # CodeGrader, ModelGrader, HumanGrader
  agents/              # BaselineQAAgent (GPT-based)
tasks/
  biomedical_core.yaml # Example evaluation suite
tests/                 # 63 tests (all mocked, no API calls)
```

## Documentation

See [docs/guide.md](docs/guide.md) for the full reference covering YAML task schema, custom agent/grader implementation, metrics, CLI options, and data models.

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
