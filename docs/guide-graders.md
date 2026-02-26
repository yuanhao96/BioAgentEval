# Customizing Graders

This guide explains how to configure, combine, and extend graders in BioAgentEval. Graders evaluate agent outputs and assign scores. They are the core mechanism that determines whether an agent passes or fails a task.

For defining the tasks and expected outputs that graders check against, see [Customizing Benchmarks](guide-benchmarks.md). For building the agents that produce the outputs graders evaluate, see [Customizing Agent Baselines](guide-agents.md).

---

## Grader Architecture

Every grader extends `BaseGrader` and implements a single method:

```python
from bioagenteval.graders.base import BaseGrader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

class MyGrader(BaseGrader):
    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        # Evaluate the outcome and return a GradeResult
        ...
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `task` | `Task` | The task being graded (includes `expected_output`, `question`, `tags`) |
| `outcome` | `str` | The agent's final text output |
| `transcript` | `Transcript` | Full execution trajectory (events, timing, queries) |
| `config` | `GraderConfig` | Grader configuration (type, rubric, weight, params) |
| `metrics` | `dict` or `None` | Computed metrics from the trial (token counts, latency, etc.) |

**Return value:**

```python
GradeResult(
    grader_type="code",      # Identifies which grader produced this result
    score=0.85,              # Float 0.0–1.0
    passed=True,             # Boolean pass/fail
    details={"info": "..."},  # Arbitrary details for debugging
)
```

---

## Built-in Grader Types

### CodeGrader (Deterministic)

The `CodeGrader` runs deterministic checks against the task's `expected_output` list. Each expected output item dispatches to a `_check_X` function based on `type`. The final score is the average of all check scores, and the trial passes if the score is >= 0.5.

**Configuration:**

```yaml
graders:
  - type: code
```

No rubric or params are needed. The CodeGrader reads all its configuration from `task.expected_output`. See [Customizing Benchmarks](guide-benchmarks.md#expected-output-types) for the full list of 17 supported check types.

**Scoring:** Each check returns a float from 0.0 to 1.0. The final code grader score is the arithmetic mean of all checks. If there are no expected outputs, the score is 1.0.

**Supported check types:**

| Category | Types |
|---|---|
| Text matching | `mcq_answer`, `exact_match`, `keyword_coverage`, `entities` |
| Numeric | `numeric_range`, `numeric_tolerance` |
| Structured output | `json_schema`, `set_similarity`, `precision_at_k` |
| Transcript-based | `tool_calls`, `turn_limit`, `trajectory_pattern`, `cypher_patterns` |
| Code | `code_valid`, `test_results` |
| Research | `groundedness` |
| Agent state | `state_check` |

### ModelGrader (LLM-Based)

The `ModelGrader` sends the task question, expected output, agent outcome, and a rubric to an LLM, which returns a structured JSON score.

**Configuration:**

```yaml
graders:
  - type: model
    rubric: >-
      Evaluate accuracy and completeness.
      1.0 = fully correct. 0.5 = partial. 0.0 = wrong.
    weight: 1.0
    params:
      model: gpt-4o           # Optional — override the default model
```

**Rubric guidelines:**
- Be specific about what constitutes each score level.
- Include the expected answer in the rubric so the LLM knows what's correct.
- Use concrete thresholds (e.g., "0.5 = partially correct" not "0.5 = maybe ok").

**Unknown verdict:** If the LLM cannot determine a score (insufficient information, ambiguous response), it can return `"verdict": "unknown"`. Unknown verdicts are excluded from the weighted score calculation so they don't unfairly penalize the agent.

**Provider support:** The ModelGrader works with both OpenAI and Anthropic via the `LLMClient` abstraction:

```bash
# Use OpenAI (default)
bioagenteval run tasks/my_suite.yaml -a myagent:MyAgent --provider openai

# Use Anthropic
bioagenteval run tasks/my_suite.yaml -a myagent:MyAgent --provider anthropic
```

### PairwiseGrader (Comparison)

The `PairwiseGrader` compares the agent's output against a reference output using an LLM judge. Useful for comparing a new agent against a baseline.

**Configuration:**

```yaml
graders:
  - type: pairwise
    rubric: Which response is more accurate and complete?
    params:
      reference_outcome: >-
        The expected reference answer goes here.
      model: gpt-4o
```

**Scoring:**
- Score close to 1.0: Agent output is clearly better than reference.
- Score around 0.5: Roughly equal (tie).
- Score close to 0.0: Reference is clearly better.

The `reference_outcome` must be provided in `config.params`. Without it, the grader returns score 0.0 with an error.

### ConsensusGrader (Multi-Judge)

The `ConsensusGrader` wraps any other grader and runs it N times, then aggregates via majority vote.

**Configuration (programmatic only):**

```python
from bioagenteval.graders import ConsensusGrader, ModelGrader, create_llm_client

client = create_llm_client("openai")
inner = ModelGrader(client=client)
consensus = ConsensusGrader(inner_grader=inner, num_judges=3)
```

**Aggregation:**
- `passed`: strict majority vote (pass_count > fail_count). On even split, result is `False`.
- `score`: average of all successful judge scores.
- `details`: includes individual judge results and vote counts.

The ConsensusGrader is automatically registered in the CLI when model grading is enabled. It wraps a `ModelGrader` with 3 judges by default.

### HumanGrader (Review Flag)

The `HumanGrader` flags tasks for manual review. It always returns `score=0.0, passed=False` with status `pending_human_review`.

**Configuration:**

```yaml
graders:
  - type: human
```

Use this for tasks that require subjective evaluation or domain expert review. The outcome preview (first 200 characters) is included in the details for quick review.

---

## GraderConfig Fields

Every grader is configured through the `GraderConfig` model:

```python
class GraderConfig(BaseModel):
    type: str                              # "code", "model", "pairwise", "human"
    rubric: str = ""                       # LLM rubric text (model/pairwise only)
    weight: float = 1.0                    # Weight in the weighted score
    params: dict[str, Any] = Field(...)    # Type-specific parameters
```

| Field | Used By | Description |
|---|---|---|
| `type` | All | Selects which grader class to use |
| `rubric` | `model`, `pairwise` | Scoring instructions for the LLM |
| `weight` | All | Contribution to `TrialResult.weighted_score()` |
| `params.model` | `model`, `pairwise` | Override the LLM model |
| `params.reference_outcome` | `pairwise` | Required reference output for comparison |

---

## Weighted Scoring

When a task has multiple graders, each produces a `GradeResult` with a `weight`. The trial's final score is a weighted average:

```
weighted_score = sum(grade.score * grade.weight) / sum(grade.weight)
```

A trial passes when `weighted_score >= 0.5`.

**Example:**

```yaml
graders:
  - type: code          # weight 1.0 (default)
  - type: model         # weight 2.0 — counts double
    rubric: "..."
    weight: 2.0
```

If code scores 0.8 and model scores 0.6:
```
weighted_score = (0.8 * 1.0 + 0.6 * 2.0) / (1.0 + 2.0) = 2.0 / 3.0 = 0.67
```

Grades with `"status": "unknown"` are excluded from the calculation entirely.

---

## Creating a Custom Code Check

To add a new deterministic check type to the `CodeGrader`, follow this pattern:

### Step 1: Write the Check Function

Add a new function in `src/bioagenteval/graders/code_grader.py`:

```python
def _check_my_custom(
    value: dict[str, Any], outcome: str,
) -> tuple[float, dict[str, Any]]:
    """Check description.

    value: {"expected": ..., "threshold": ...}
    """
    expected = value.get("expected", "")
    threshold = value.get("threshold", 0.5)

    # Your logic here
    score = compute_score(expected, outcome, threshold)

    details = {"expected": expected, "computed": score}
    return score, details
```

**Conventions:**
- Function name must be `_check_<type_name>`.
- Return `float` if no extra details, or `tuple[float, dict]` if you have details.
- The `value` parameter comes from `expected_output[].value` in the YAML.
- Always handle edge cases (empty input, missing keys).

### Step 2: Register in the Dispatch Table

Add the dispatch entry in `CodeGrader.grade()`:

```python
elif eo.type == "my_custom":
    mc_score, mc_details = _check_my_custom(eo.value, outcome)
    check_results["my_custom"] = mc_score
    extra_details.update(mc_details)
```

### Step 3: Use in YAML

```yaml
expected_output:
  - type: my_custom
    value:
      expected: "some value"
      threshold: 0.8
```

### Step 4: Write Tests

```python
def test_my_custom_check_passes():
    task = Task(
        id="test_1",
        question="Test question",
        expected_output=[
            ExpectedOutput(type="my_custom", value={"expected": "foo", "threshold": 0.5}),
        ],
    )
    grader = CodeGrader()
    config = GraderConfig(type="code")
    transcript = Transcript(task_id="test_1")
    result = grader.grade(task, "The answer is foo", transcript, config)
    assert result.score >= 0.5
```

---

## Creating a Custom Grader Class

For checks that don't fit the `CodeGrader` dispatch pattern (e.g., calling an external API, running a custom model), create a new grader class:

### Step 1: Implement BaseGrader

```python
# src/bioagenteval/graders/my_grader.py
from bioagenteval.graders.base import BaseGrader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

class MyCustomGrader(BaseGrader):
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        # Read config
        threshold = config.params.get("threshold", 0.8)

        # Your grading logic
        score = self._evaluate(outcome, task.expected_output, threshold)

        return GradeResult(
            grader_type="my_custom",
            score=score,
            passed=score >= 0.5,
            details={"threshold": threshold},
        )

    def _evaluate(self, outcome, expected_output, threshold):
        # Implementation
        return 1.0
```

### Step 2: Register with EvalRunner

```python
from bioagenteval.runner import EvalRunner
from mymodule.graders import MyCustomGrader

graders = {
    "code": CodeGrader(),
    "model": ModelGrader(client=client),
    "my_custom": MyCustomGrader(api_key="..."),
}

runner = EvalRunner(agent=my_agent, graders=graders)
```

### Step 3: Reference in Task YAML

```yaml
graders:
  - type: my_custom
    params:
      threshold: 0.9
```

The `type` string must match the key in the `graders` dict passed to `EvalRunner`.

---

## LLM Client Abstraction

The `LLMClient` Protocol enables model graders to work with different LLM providers without code changes:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        system: str = "",
        model: str = "",
        max_tokens: int = 256,
    ) -> str: ...
```

### Built-in Clients

| Client | Provider | Default Model |
|---|---|---|
| `OpenAILLMClient` | OpenAI | `gpt-4o` |
| `AnthropicLLMClient` | Anthropic | `claude-sonnet-4-20250514` |

### Creating a Custom Client

```python
class OllamaLLMClient:
    """LLM client for a local Ollama instance."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def complete(
        self,
        messages: list[dict],
        *,
        system: str = "",
        model: str = "llama3",
        max_tokens: int = 256,
    ) -> str:
        import requests

        all_messages = list(messages)
        if system:
            all_messages.insert(0, {"role": "system", "content": system})

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": model, "messages": all_messages, "stream": False},
        )
        return response.json()["message"]["content"]
```

Use it with any LLM grader:

```python
client = OllamaLLMClient()
graders = {
    "model": ModelGrader(client=client),
    "pairwise": PairwiseGrader(client=client),
    "consensus": ConsensusGrader(inner_grader=ModelGrader(client=client)),
}
```

---

## Grader Calibration

Validate your grader's accuracy against labeled examples before trusting it in production:

```python
from bioagenteval.calibration import calibrate_grader, CalibrationResult
from bioagenteval.graders.code_grader import CodeGrader
from bioagenteval.models import GraderConfig, Task, Transcript, ExpectedOutput

grader = CodeGrader()

examples = [
    {
        "task": Task(
            id="cal_1",
            question="What gene is mutated in Li-Fraumeni?",
            expected_output=[ExpectedOutput(type="entities", value=["TP53"])],
        ),
        "outcome": "The TP53 gene is mutated.",
        "transcript": Transcript(task_id="cal_1"),
        "config": GraderConfig(type="code"),
        "expected_passed": True,
    },
    {
        "task": Task(
            id="cal_2",
            question="What gene is mutated in Li-Fraumeni?",
            expected_output=[ExpectedOutput(type="entities", value=["TP53"])],
        ),
        "outcome": "I'm not sure.",
        "transcript": Transcript(task_id="cal_2"),
        "config": GraderConfig(type="code"),
        "expected_passed": False,
    },
]

result: CalibrationResult = calibrate_grader(grader, examples)
print(f"Accuracy: {result.accuracy:.0%}")
print(f"Precision: {result.precision:.0%}")
print(f"Recall: {result.recall:.0%}")
print(f"Confusion: {result.confusion_matrix}")
```

**CalibrationResult fields:**

| Field | Description |
|---|---|
| `accuracy` | Fraction of correct predictions (TP + TN) / total |
| `precision` | TP / (TP + FP) — of predicted passes, how many are correct |
| `recall` | TP / (TP + FN) — of actual passes, how many were detected |
| `confusion_matrix` | `{"tp": N, "fp": N, "tn": N, "fn": N}` |
| `details` | Per-example results with task_id, expected, actual, correct |

---

## Custom Metrics

Metrics measure execution characteristics (latency, cost, token usage) rather than output quality. They're computed from the transcript and stored alongside grade results.

### Registering a Custom Metric

```python
from bioagenteval.metrics import register_metric
from bioagenteval.models import Transcript

@register_metric("n_cypher_queries")
def _n_cypher_queries(transcript: Transcript, duration_ms: float) -> int:
    """Count Cypher queries in the transcript."""
    return len(transcript.cypher_queries)

@register_metric("avg_tokens_per_turn")
def _avg_tokens_per_turn(transcript: Transcript, duration_ms: float) -> float | None:
    """Average tokens per LLM call."""
    llm_events = [ev for ev in transcript.events if ev.event_type == "llm_call"]
    if not llm_events:
        return None
    total = sum(
        ev.data.get("prompt_tokens", 0) + ev.data.get("completion_tokens", 0)
        for ev in llm_events
    )
    return total / len(llm_events)
```

**Metric function signature:**

```python
MetricFn = Callable[[Transcript, float], Any]
# Parameters: (transcript, duration_ms)
# Returns: any JSON-serializable value
```

### Using Custom Metrics in YAML

```yaml
default_tracked_metrics:
  - type: transcript
    metrics:
      - n_cypher_queries
      - avg_tokens_per_turn
```

Ensure your custom metric module is imported before the runner executes. The simplest way is to import it in your agent module or in a conftest.py.

### Built-in Metrics Reference

| Metric | Group | Returns | Description |
|---|---|---|---|
| `n_turns` | transcript | `int` | Count of `llm_call` events |
| `n_tool_calls` | transcript | `int` | Count of `tool_call`, `tool_use`, `cypher_query` events |
| `n_total_tokens` | transcript | `int` | Sum of prompt + completion tokens |
| `time_to_first_token` | latency | `float\|None` | ms from start to first response event |
| `output_tokens_per_sec` | latency | `float\|None` | Completion tokens / second |
| `time_to_last_token` | latency | `float` | Total duration in ms |
| `estimated_cost` | cost | `float` | USD estimate (GPT-4o pricing) |

---

## Grader Retry and Error Handling

The `EvalRunner` supports automatic retry for graders that may fail due to transient errors (API timeouts, rate limits):

```python
runner = EvalRunner(
    agent=my_agent,
    graders=graders,
    max_retries=2,  # Retry up to 2 times on failure
)
```

Retry uses exponential backoff (1s, 2s, 4s, ...). After all retries are exhausted, the grader returns `score=0.0, passed=False` with the error in details.

---

## Common Patterns

### Code + Model Grader Combination

The most common pattern is deterministic + LLM grading:

```yaml
graders:
  - type: code           # Fast, deterministic, catches exact answers
  - type: model          # Flexible, catches partial credit and nuance
    rubric: >-
      Expected: TP53 causes Li-Fraumeni syndrome.
      1.0 = mentions TP53 and Li-Fraumeni correctly.
      0.5 = mentions one but not both.
      0.0 = incorrect or irrelevant.
```

The code grader catches clear pass/fail cases. The model grader handles partial credit and subjective quality.

### Heavy Model Weight for Subjective Tasks

For tasks without a single correct answer:

```yaml
graders:
  - type: code
    weight: 0.5
  - type: model
    weight: 2.0
    rubric: >-
      Evaluate the research summary for completeness, accuracy, and depth.
```

### Skip Model Grading for Fast Iteration

During development, run only the code grader:

```bash
bioagenteval run tasks/my_suite.yaml -a myagent:MyAgent --skip-model-grader
```

This avoids LLM API costs and latency while iterating on deterministic checks.

### Negative Test Inversion

When a task has `should_fail: true`, the runner inverts all grades after grading:

```python
# After grading:
for grade in grades:
    grade.passed = not grade.passed
    grade.score = 1.0 - grade.score
```

This means you should design `expected_output` to match a "correct failure" response. For example, if the agent should refuse a question, use `keyword_coverage` to check for refusal language — the code grader will score high when the refusal is detected, and the runner inverts it so the task passes.
