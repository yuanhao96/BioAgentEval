# BioAgentEval — Evaluation Harness for Biomedical KG Agents

**BioAgentEval** evaluates biomedical question-answering agents over a Neo4j knowledge graph. It measures answer quality through deterministic checks, LLM-based rubric scoring, and human review, with multi-trial pass@k metrics and full trajectory capture.

Built following [Anthropic's "Demystifying Evals for AI Agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

```
EvalSuite (named group of tasks)
  └── Task (question + expected entities + graders)
        └── EvalResult (aggregated across trials)
              └── TrialResult (one attempt)
                    ├── Transcript → TranscriptEvent (tool calls, Cypher queries, …)
                    └── GradeResult (output from one grader)
```

---

## Quick Start

### Install

```bash
git clone git@github.com:yuanhao96/BioAgentEval.git
cd BioAgentEval
pip install -e ".[dev]"
```

### Validate a suite

Check that your YAML task file is well-formed:

```bash
bioagenteval validate tasks/biomedical_core.yaml
```

Output:

```
Suite: biomedical_core
Tasks: 3
  gene_diabetes_association: 3 trials, graders=['code', 'model']
  gene_overview_ins: 2 trials, graders=['code']
  pathway_insulin_signaling: 3 trials, graders=['code', 'model']
Validation passed.
```

### Run an evaluation

Run with only deterministic (code) grading — no API keys needed:

```bash
bioagenteval run tasks/biomedical_core.yaml \
  --agent bioagenteval.agents.baseline_qa:BaselineQAAgent \
  --skip-model-grader \
  --output results/report.json
```

Run with all graders (requires `OPENAI_API_KEY` for the agent and `ANTHROPIC_API_KEY` for model grading):

```bash
bioagenteval run tasks/biomedical_core.yaml \
  --agent bioagenteval.agents.baseline_qa:BaselineQAAgent \
  --output results/report.json
```

### Read the report

The output is a JSON file with per-task and overall metrics:

```json
{
  "suite_name": "biomedical_core",
  "run_id": "a1b2c3d4-...",
  "timestamp": "2026-02-13T18:30:00+00:00",
  "results": [
    {
      "task_id": "gene_diabetes_association",
      "pass_at_1": 1.0,
      "mean_scores": {"code": 0.75, "model": 0.9},
      "num_trials": 3,
      "trials": [...]
    }
  ],
  "summary": {
    "total_tasks": 3,
    "overall_pass_at_1": 0.83
  }
}
```

---

## Defining Tasks (YAML)

Tasks are defined in YAML suite files. Each file contains one suite with one or more tasks.

### Full schema

```yaml
# Suite-level fields
name: my_suite                  # Required. Suite identifier.
description: What this tests    # Optional.
default_num_trials: 3           # Optional (default: 1). Inherited by tasks.

tasks:
  - id: task_id                 # Required. Unique within suite.
    question: "..."             # Required. The question sent to the agent.
    expected_entities:          # Optional. Entities the answer should mention.
      - BRCA1
      - TP53
    expected_complexity: simple # Optional. "simple" or "complex".
    expected_cypher_patterns:   # Optional. Regex patterns for Cypher queries.
      - "MATCH.*Gene.*BRCA1"
    num_trials: 5               # Optional. Overrides default_num_trials.
    metadata:                   # Optional. Arbitrary key-value pairs.
      source: literature
    graders:                    # List of graders to apply.
      - type: code
        checks:
          - entity_presence
          - cypher_pattern
      - type: model
        rubric: >
          Is the answer accurate and complete?
        params:
          model: claude-sonnet-4-5-20250929   # Override default model.
      - type: human
```

### Suite-level defaults

`default_num_trials` sets the number of trials for any task that omits `num_trials`. An explicit task-level `num_trials` always wins.

### Grader configuration

Each grader entry has:

| Field    | Type         | Description                                          |
|----------|--------------|------------------------------------------------------|
| `type`   | `str`        | `"code"`, `"model"`, or `"human"` (required)         |
| `checks` | `list[str]`  | Code grader checks: `"entity_presence"`, `"cypher_pattern"` |
| `rubric` | `str`        | Rubric text for model grader                         |
| `weight` | `float`      | Weight (default 1.0, reserved for future use)        |
| `params` | `dict`       | Extra params, e.g. `{"model": "claude-sonnet-4-5-20250929"}` |

### Complete example

```yaml
name: biomedical_core
description: Core biomedical knowledge evaluation tasks
default_num_trials: 3

tasks:
  - id: gene_diabetes_association
    question: "What genes are associated with type 1 diabetes?"
    expected_entities:
      - INS
      - HLA-DRB1
      - HLA-DQB1
      - PTPN22
    expected_complexity: complex
    graders:
      - type: code
        checks:
          - entity_presence
      - type: model
        rubric: >
          Does the answer correctly identify major genes associated with
          type 1 diabetes? Is it comprehensive and accurate?
    num_trials: 3

  - id: gene_overview_ins
    question: "Tell me about the INS gene and its role in disease."
    expected_entities:
      - INS
      - ENSG00000254647
      - insulin
    expected_complexity: simple
    graders:
      - type: code
        checks:
          - entity_presence
    num_trials: 2
```

### Tips for writing tasks

- **Expected entities**: use the canonical form the agent is likely to produce (gene symbols, ENSEMBL IDs). Matching is case-insensitive substring search.
- **Cypher patterns**: use regex. The pattern is matched against the concatenation of all Cypher queries in the transcript (case-insensitive).
- **Rubrics**: write as if instructing a domain expert. State what a passing answer must contain.
- **Complexity**: `simple` = single-hop lookup, `complex` = multi-hop or aggregation. This is metadata for analysis, not used by graders.

---

## Agents

### AgentHarness protocol

Any Python object with `run()` and `reset()` methods satisfies the protocol. No inheritance required — this uses Python's structural subtyping (`typing.Protocol`).

```python
from bioagenteval.models import AgentResponse

class AgentHarness(Protocol):
    def run(self, question: str) -> AgentResponse: ...
    def reset(self) -> None: ...
```

`AgentResponse` contains:
- `outcome: str` — the agent's final answer text
- `transcript: Transcript` — the full trajectory of events

### Implementing a custom agent

```python
from datetime import datetime, timezone
from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent

class MyKGAgent:
    def __init__(self):
        self.graph = connect_to_neo4j()

    def run(self, question: str) -> AgentResponse:
        started_at = datetime.now(timezone.utc)

        # Your agent logic here
        cypher = self._generate_cypher(question)
        result = self.graph.execute(cypher)
        answer = self._format_answer(result)

        finished_at = datetime.now(timezone.utc)

        transcript = Transcript(
            task_id="",  # filled by EvalRunner context
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": cypher},
                    timestamp=started_at,
                ),
                TranscriptEvent(
                    event_type="cypher_result",
                    data={"rows": len(result)},
                    timestamp=finished_at,
                ),
            ],
            cypher_queries=[cypher],
            started_at=started_at,
            finished_at=finished_at,
        )

        return AgentResponse(outcome=answer, transcript=transcript)

    def reset(self) -> None:
        # Clear any cached state between trials
        pass
```

### Recording transcript events

Use `event_type` to categorize steps. The code grader specifically looks for `event_type="cypher_query"` with `data.query` to evaluate Cypher patterns. Common event types:

| event_type       | data keys              | Description                    |
|------------------|------------------------|--------------------------------|
| `llm_call`       | `question`, `model`    | LLM API request               |
| `llm_response`   | `answer`               | LLM API response               |
| `cypher_query`   | `query`                | Cypher query sent to Neo4j     |
| `cypher_result`  | `rows`, `columns`      | Neo4j query result             |
| `tool_call`      | `tool`, `args`         | Generic tool invocation        |

### Built-in: BaselineQAAgent

A simple single-turn agent using OpenAI GPT (`src/bioagenteval/agents/baseline_qa.py`):

```python
from bioagenteval.agents import BaselineQAAgent

agent = BaselineQAAgent(model="gpt-4o")  # default model
response = agent.run("What genes are associated with type 1 diabetes?")
print(response.outcome)
```

Requires `OPENAI_API_KEY` environment variable.

### Loading custom agents via CLI

The `--agent` flag takes a `module:Class` path. The class is dynamically imported and instantiated with no arguments:

```bash
# Built-in agent
bioagenteval run suite.yaml --agent bioagenteval.agents.baseline_qa:BaselineQAAgent

# Your custom agent (must be importable)
bioagenteval run suite.yaml --agent my_project.agents:MyKGAgent
```

---

## Graders

All graders implement `BaseGrader` and its single method:

```python
class BaseGrader(ABC):
    @abstractmethod
    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult: ...
```

Each grader returns a `GradeResult` with:
- `grader_type: str` — `"code"`, `"model"`, or `"human"`
- `score: float` — 0.0 to 1.0
- `passed: bool` — whether this grade counts as passing
- `details: dict` — grader-specific information

### CodeGrader

Deterministic checks with no API calls. Supports two check types configured via `config.checks`:

**`entity_presence`** — case-insensitive substring matching of expected entities in the agent's answer:

```
score = (entities found in outcome) / (total expected entities)
```

Example: if `expected_entities: [INS, HLA-DRB1]` and the answer mentions "INS" but not "HLA-DRB1", score = 0.5.

**`cypher_pattern`** — regex matching against Cypher queries from the transcript. Extracts queries from `TranscriptEvent` entries where `event_type == "cypher_query"`, joins them, and applies each pattern:

```
score = (patterns matched) / (total expected patterns)
```

When multiple checks are configured, the final score is their average. A trial passes if `score >= 0.5`.

If no checks are configured, score defaults to 1.0.

### ModelGrader

LLM-based rubric scoring using the Anthropic API. Sends the task question, expected entities, agent response, and rubric to Claude, expecting a JSON response with `score`, `passed`, and `reasoning`.

```python
from bioagenteval.graders import ModelGrader

grader = ModelGrader(model="claude-sonnet-4-5-20250929")  # default
```

The model can be overridden per-task via `config.params["model"]`.

On API or parsing errors, returns `score=0.0, passed=False` with the error in `details`.

Requires `ANTHROPIC_API_KEY` environment variable.

### HumanGrader

A stub that flags results for manual review. Always returns `score=0.0, passed=False` with `status: "pending_human_review"`. Use this for calibration — compare human judgments against code and model graders to validate your eval suite.

### Writing a custom grader

```python
from bioagenteval.graders.base import BaseGrader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

class SemanticSimilarityGrader(BaseGrader):
    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        # Your grading logic
        reference = config.params.get("reference_answer", "")
        similarity = compute_similarity(outcome, reference)

        return GradeResult(
            grader_type="semantic",
            score=similarity,
            passed=similarity >= 0.7,
            details={"reference": reference},
        )
```

Register it in the graders dict when creating `EvalRunner`:

```python
runner = EvalRunner(
    agent=my_agent,
    graders={
        "code": CodeGrader(),
        "model": ModelGrader(),
        "semantic": SemanticSimilarityGrader(),
    },
)
```

Then reference it in your YAML tasks:

```yaml
graders:
  - type: semantic
    params:
      reference_answer: "The INS gene encodes insulin..."
```

---

## Metrics

### pass@k

The probability that at least one of k randomly selected trials passes. Uses the unbiased combinatorial estimator:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where n = total trials, c = passing trials, and C is the binomial coefficient.

A trial passes when **all** of its graders return `passed=True`.

Edge cases:
- `n == 0` or `k <= 0` → 0.0
- `k > n` → clamped to n
- All trials pass → 1.0
- No trials pass → 0.0

```python
result = EvalResult(task_id="t1", trials=[...])
result.pass_at_k(k=1)   # probability of passing in 1 try
result.pass_at_k(k=3)   # probability of at least 1 pass in 3 tries
```

### mean_score

Average score across all trials for a specific grader type:

```python
result.mean_score("code")    # mean of all code grader scores
result.mean_score("model")   # mean of all model grader scores
```

Returns 0.0 if no grades of that type exist.

### Report summary

The report includes `overall_pass_at_1`: the average of pass@1 values across all tasks. This is the single headline metric for a suite run.

---

## Running Evaluations

### Programmatic API

```python
from bioagenteval.loader import load_suite
from bioagenteval.runner import EvalRunner
from bioagenteval.reporter import EvalReporter
from bioagenteval.graders import CodeGrader, ModelGrader
from bioagenteval.agents import BaselineQAAgent

# Load
suite, tasks = load_suite("tasks/biomedical_core.yaml")

# Configure
agent = BaselineQAAgent()
runner = EvalRunner(
    agent=agent,
    graders={"code": CodeGrader(), "model": ModelGrader()},
)

# Run
results = runner.run_suite(tasks)

# Report
EvalReporter.save_report(suite.name, results, "report.json")

# Or get the dict directly
report = EvalReporter.generate_report(suite.name, results)
print(f"Overall pass@1: {report['summary']['overall_pass_at_1']:.2%}")
```

### CLI reference

```
Usage: bioagenteval [OPTIONS] COMMAND [ARGS]...

  BioAgentEval — evaluation harness for biomedical KG agents.

Options:
  -v, --verbose  Enable verbose logging.

Commands:
  run       Run an evaluation suite against an agent.
  validate  Validate a task suite YAML file.
```

**`bioagenteval run`**

```
Usage: bioagenteval run [OPTIONS] SUITE_PATH

Options:
  -a, --agent TEXT          Agent module path (required), e.g.
                            bioagenteval.agents.baseline_qa:BaselineQAAgent
  -o, --output TEXT         Output JSON path (default: eval_report.json)
  --skip-model-grader       Skip model-based grading
```

**`bioagenteval validate`**

```
Usage: bioagenteval validate [OPTIONS] SUITE_PATH
```

### Report JSON structure

```json
{
  "suite_name": "...",
  "run_id": "uuid",
  "timestamp": "ISO-8601",
  "results": [
    {
      "task_id": "...",
      "pass_at_1": 0.0-1.0,
      "mean_scores": {"code": 0.75, "model": 0.9},
      "num_trials": 3,
      "trials": [
        {
          "trial_num": 0,
          "outcome": "The agent's answer...",
          "grades": [{"grader_type": "code", "score": 0.75, "passed": true, "details": {...}}],
          "transcript": {"task_id": "...", "events": [...], ...},
          "duration_ms": 1234.5,
          "error": null
        }
      ]
    }
  ],
  "summary": {
    "total_tasks": 3,
    "overall_pass_at_1": 0.83
  }
}
```

---

## Data Models Reference

All models are Pydantic v2 `BaseModel` instances defined in `src/bioagenteval/models.py`.

| Model             | Key Fields                                                        | Purpose                          |
|-------------------|-------------------------------------------------------------------|----------------------------------|
| `GraderConfig`    | `type`, `checks`, `rubric`, `weight`, `params`                    | Configures a grader for a task   |
| `Task`            | `id`, `question`, `expected_entities`, `expected_cypher_patterns`, `graders`, `num_trials` | Single evaluation test case      |
| `EvalSuite`       | `name`, `description`, `task_ids`, `default_num_trials`           | Named group of tasks             |
| `TranscriptEvent` | `event_type`, `event_name`, `data`, `timestamp`                   | Single step in agent trajectory  |
| `Transcript`      | `task_id`, `events`, `cypher_queries`, `neo4j_results`, `started_at`, `finished_at` | Full trial trajectory |
| `GradeResult`     | `grader_type`, `score`, `passed`, `details`                       | Output from one grader           |
| `TrialResult`     | `task_id`, `trial_num`, `outcome`, `transcript`, `grades`, `duration_ms`, `error` | One attempt at a task |
| `EvalResult`      | `task_id`, `trials` + methods `pass_at_k()`, `mean_score()`       | Aggregated result for one task   |
| `AgentResponse`   | `outcome`, `transcript`                                           | Structured agent return value    |

---

## Eval Design Best Practices

From [Anthropic's guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):

- **Grade the outcome, not the path.** Don't penalize creative agent strategies — if the answer is correct, the route doesn't matter.
- **Make tasks unambiguous.** Two domain experts should independently agree on pass/fail. If they can't, rewrite the task.
- **Start with real failures.** Seed your suite with 20-50 questions the agent actually got wrong, then expand.
- **Read transcripts regularly.** They reveal grading bugs, agent failure modes, and prompt issues that metrics alone miss.
- **Promote saturated evals.** When pass rate nears 100%, move those tasks to a regression suite and write harder ones.
