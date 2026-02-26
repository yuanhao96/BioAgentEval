# Customizing Agent Baselines

This guide explains how to build, configure, and run agent baselines against BioAgentEval evaluation suites. An agent is any Python class that satisfies the `AgentHarness` protocol — no inheritance required.

For creating the evaluation tasks that test your agent, see [Customizing Benchmarks](guide-benchmarks.md). For configuring how agent outputs are scored, see [Customizing Graders](guide-graders.md).

---

## AgentHarness Protocol

Every agent must implement two methods:

```python
from bioagenteval.models import AgentResponse

class MyAgent:
    def run(self, question: str) -> AgentResponse:
        """Run the agent on a question, return structured response."""
        ...

    def reset(self) -> None:
        """Reset agent state between trials."""
        ...
```

That's it. No base class, no inheritance, no registration. BioAgentEval uses structural subtyping (duck typing) — if your class has `run` and `reset` with the right signatures, it works.

The `AgentHarness` Protocol is defined in `src/bioagenteval/harness.py` and is `@runtime_checkable`, so you can verify compliance:

```python
from bioagenteval.harness import AgentHarness

agent = MyAgent()
assert isinstance(agent, AgentHarness)  # True if protocol is satisfied
```

---

## AgentResponse

The `run` method must return an `AgentResponse`:

```python
from bioagenteval.models import AgentResponse, Transcript

class AgentResponse(BaseModel):
    outcome: str            # The agent's final text output
    transcript: Transcript  # Full execution trajectory
```

**`outcome`**: The text that graders evaluate. This is what the `CodeGrader` checks against `expected_output` and what the `ModelGrader` shows to the LLM judge.

**`transcript`**: The execution trajectory. Contains events (tool calls, LLM calls, queries), timing, and optional Neo4j artifacts. Used by transcript-based checks (`tool_calls`, `turn_limit`, `trajectory_pattern`, `state_check`, `test_results`) and metrics computation.

---

## Transcript and Events

### Transcript Structure

```python
class Transcript(BaseModel):
    task_id: str
    events: list[TranscriptEvent] = []
    cypher_queries: list[str] = []          # Neo4j-specific
    neo4j_results: list[dict[str, Any]] = [] # Neo4j-specific
    started_at: datetime | None = None
    finished_at: datetime | None = None
```

### TranscriptEvent

```python
class TranscriptEvent(BaseModel):
    event_type: str                         # Category: "llm_call", "tool_call", etc.
    event_name: str = ""                    # Specific name: "search_pubmed", etc.
    data: dict[str, Any] = {}              # Arbitrary payload
    timestamp: datetime = <now>
```

### Standard Event Types

Use these event types for compatibility with built-in grader checks and metrics:

| Event Type | Used By | Required `data` Fields |
|---|---|---|
| `llm_call` | `n_turns`, `turn_limit`, `time_to_first_token` | `prompt_tokens`, `completion_tokens` (for token metrics) |
| `llm_response` | `time_to_first_token` | `answer` (optional) |
| `tool_call` | `n_tool_calls`, `tool_calls` check | None (use `event_name` for tool name) |
| `tool_use` | `n_tool_calls`, `tool_calls` check | None (use `event_name` for tool name) |
| `cypher_query` | `n_tool_calls`, `cypher_patterns` check | `query` |
| `test_result` | `test_results` check | `test_name`, `passed` (bool) |
| `state_snapshot` | `state_check` check | Arbitrary key-value state |

You can add custom event types for your own analysis — they won't interfere with built-in checks.

---

## Building an Agent: Step by Step

### Minimal Agent (Single-Turn QA)

```python
# src/myagent/simple_qa.py
from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent
from datetime import datetime, timezone

class SimpleQAAgent:
    """Single-turn QA agent using OpenAI."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()

    def run(self, question: str) -> AgentResponse:
        started = datetime.now(timezone.utc)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}],
        )

        answer = response.choices[0].message.content
        finished = datetime.now(timezone.utc)

        transcript = Transcript(
            task_id="",  # Filled by the runner if needed
            events=[
                TranscriptEvent(
                    event_type="llm_call",
                    event_name="chat_completion",
                    data={
                        "model": self.model,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                    timestamp=started,
                ),
                TranscriptEvent(
                    event_type="llm_response",
                    event_name="chat_completion_result",
                    data={"answer": answer},
                    timestamp=finished,
                ),
            ],
            started_at=started,
            finished_at=finished,
        )

        return AgentResponse(outcome=answer, transcript=transcript)

    def reset(self) -> None:
        pass  # Stateless — nothing to reset
```

### Multi-Turn Agent with Tool Use

```python
# src/myagent/tool_agent.py
from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent
from datetime import datetime, timezone

class ToolUsingAgent:
    """Agent that uses tools across multiple turns."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()
        self.conversation: list[dict] = []
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_literature",
                    "description": "Search PubMed for papers",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            },
        ]

    def run(self, question: str) -> AgentResponse:
        started = datetime.now(timezone.utc)
        events: list[TranscriptEvent] = []
        self.conversation.append({"role": "user", "content": question})

        max_turns = 10
        for turn in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                tools=self.tools,
            )

            events.append(TranscriptEvent(
                event_type="llm_call",
                event_name=f"turn_{turn}",
                data={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            ))

            message = response.choices[0].message
            self.conversation.append(message.model_dump())

            # Check for tool calls
            if message.tool_calls:
                for tc in message.tool_calls:
                    events.append(TranscriptEvent(
                        event_type="tool_call",
                        event_name=tc.function.name,
                        data={"arguments": tc.function.arguments},
                    ))

                    # Execute tool and add result
                    result = self._execute_tool(tc.function.name, tc.function.arguments)
                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                # No tool calls — we have the final answer
                break

        answer = self.conversation[-1].get("content", "")
        finished = datetime.now(timezone.utc)

        transcript = Transcript(
            task_id="",
            events=events,
            started_at=started,
            finished_at=finished,
        )

        return AgentResponse(outcome=answer, transcript=transcript)

    def reset(self) -> None:
        self.conversation.clear()

    def _execute_tool(self, name: str, arguments: str) -> str:
        # Implement tool execution
        return '{"results": []}'
```

### Knowledge Graph Agent with Cypher

```python
# src/myagent/kg_agent.py
from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent
from datetime import datetime, timezone

class KnowledgeGraphAgent:
    """Agent that queries a Neo4j knowledge graph."""

    def __init__(self, neo4j_uri: str = "bolt://localhost:7687"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(neo4j_uri)

    def run(self, question: str) -> AgentResponse:
        started = datetime.now(timezone.utc)
        events = []
        cypher_queries = []
        neo4j_results = []

        # Step 1: Generate Cypher
        cypher = self._generate_cypher(question)
        cypher_queries.append(cypher)
        events.append(TranscriptEvent(
            event_type="cypher_query",
            event_name="generated_query",
            data={"query": cypher},
        ))

        # Step 2: Execute query
        with self.driver.session() as session:
            result = session.run(cypher)
            records = [dict(r) for r in result]
            neo4j_results.append({"records": records})

        # Step 3: Generate answer from results
        answer = self._synthesize_answer(question, records)
        finished = datetime.now(timezone.utc)

        events.append(TranscriptEvent(
            event_type="llm_response",
            event_name="final_answer",
            data={"answer": answer},
            timestamp=finished,
        ))

        transcript = Transcript(
            task_id="",
            events=events,
            cypher_queries=cypher_queries,
            neo4j_results=neo4j_results,
            started_at=started,
            finished_at=finished,
        )

        return AgentResponse(outcome=answer, transcript=transcript)

    def reset(self) -> None:
        pass  # Stateless queries — nothing to reset

    def _generate_cypher(self, question: str) -> str:
        # Use LLM to generate Cypher
        return "MATCH (g:Gene {name: 'TP53'}) RETURN g"

    def _synthesize_answer(self, question: str, records: list) -> str:
        # Use LLM to synthesize answer from records
        return "TP53 is a tumor suppressor gene."
```

### Agent with State (Computer Use / Conversational)

```python
# src/myagent/stateful_agent.py
from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent

class StatefulAgent:
    """Agent that maintains state across interactions."""

    def __init__(self):
        self.state: dict = {}

    def run(self, question: str) -> AgentResponse:
        events = []

        # Process the question and update state
        self.state["last_query"] = question
        self.state["ui.screen"] = "results"

        # Record state snapshot for state_check grading
        events.append(TranscriptEvent(
            event_type="state_snapshot",
            event_name="after_processing",
            data=dict(self.state),  # Copy current state
        ))

        answer = f"Processed: {question}"
        transcript = Transcript(task_id="", events=events)
        return AgentResponse(outcome=answer, transcript=transcript)

    def reset(self) -> None:
        self.state.clear()  # Clear state between trials
```

---

## Running Your Agent

### From the CLI

```bash
# Basic run
bioagenteval run tasks/my_suite.yaml \
  -a myagent.simple_qa:SimpleQAAgent \
  -o results/report.json

# Skip model grading (faster, no API costs)
bioagenteval run tasks/my_suite.yaml \
  -a myagent.simple_qa:SimpleQAAgent \
  --skip-model-grader \
  -o results/report.json

# Use Anthropic for model grading
bioagenteval run tasks/my_suite.yaml \
  -a myagent.simple_qa:SimpleQAAgent \
  --provider anthropic \
  -o results/report.json

# Filter to specific tasks
bioagenteval run tasks/my_suite.yaml \
  -a myagent.simple_qa:SimpleQAAgent \
  --tags difficulty=easy \
  --tags category=LitQA2
```

The `-a` flag takes a `MODULE:CLASS` string. The module must be importable (on `sys.path` or installed). The class must have a no-argument constructor.

### Programmatic Usage

```python
from bioagenteval.graders import CodeGrader, ModelGrader, create_llm_client
from bioagenteval.loader import load_suite
from bioagenteval.reporter import EvalReporter
from bioagenteval.runner import EvalRunner

# Load suite
suite, tasks = load_suite("tasks/my_suite.yaml")

# Create agent
agent = SimpleQAAgent(model="gpt-4o")

# Set up graders
client = create_llm_client("openai")
graders = {
    "code": CodeGrader(),
    "model": ModelGrader(client=client),
}

# Run evaluation
runner = EvalRunner(
    agent=agent,
    graders=graders,
    max_concurrency=4,   # Parallel task execution
    max_retries=2,       # Retry on grader failure
    default_timeout=120, # 120s timeout per trial
)
results = runner.run_suite(tasks)

# Generate report
report = EvalReporter.generate_report(suite.name, results, eval_type=suite.eval_type)
EvalReporter.save_report(suite.name, results, "results/report.json", eval_type=suite.eval_type)

# Print summary
summary = report["summary"]
print(f"Tasks: {summary['total_tasks']}")
print(f"pass@1: {summary['overall_pass_at_1']:.0%}")
print(f"pass^1: {summary['overall_pass_hat_1']:.0%}")
```

---

## EvalRunner Configuration

The `EvalRunner` orchestrates running tasks and grading:

```python
runner = EvalRunner(
    agent=agent,                  # AgentHarness instance
    graders=graders,              # dict[str, BaseGrader]
    max_concurrency=1,            # Parallel task execution (1 = sequential)
    max_retries=0,                # Retry grading on failure
    hooks=[cleanup_hook],         # TrialHook instances
    default_timeout=None,         # Default timeout in seconds per trial
)
```

| Parameter | Default | Description |
|---|---|---|
| `max_concurrency` | `1` | Number of tasks to run in parallel. Set > 1 for independent tasks. |
| `max_retries` | `0` | Number of retry attempts for failed grading calls (exponential backoff). |
| `hooks` | `[]` | `TrialHook` instances for setup/teardown per trial. |
| `default_timeout` | `None` | Seconds before a trial times out. Per-task `metadata.timeout` overrides this. |

### Trial Execution Flow

For each task and trial:

1. `agent.reset()` — Clear agent state.
2. `hook.setup(task)` — Run setup hooks (database reset, file cleanup).
3. `agent.run(question)` — Execute the agent (with optional timeout).
4. `compute_metrics(...)` — Compute tracked metrics from transcript.
5. `grader.grade(...)` — Run each grader (with optional retry).
6. Invert scores if `task.should_fail` is true.
7. `hook.teardown(task, trial_result)` — Run teardown hooks.

---

## Trial Hooks

Hooks run before and after each trial for environment management:

```python
from bioagenteval.harness import TrialHook
from bioagenteval.models import Task, TrialResult

class DatabaseResetHook:
    """Reset Neo4j database state between trials."""

    def __init__(self, driver):
        self.driver = driver

    def setup(self, task: Task) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n:TestNode) DELETE n")

    def teardown(self, task: Task, trial: TrialResult) -> None:
        # Log trial results, capture state snapshots, etc.
        print(f"Task {task.id} trial {trial.trial_num}: score={trial.weighted_score()}")
```

```python
class FileCleanupHook:
    """Clean up temporary files between trials."""

    def __init__(self, tmp_dir: str):
        self.tmp_dir = tmp_dir

    def setup(self, task: Task) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def teardown(self, task: Task, trial: TrialResult) -> None:
        pass
```

Register hooks with the runner:

```python
hooks = [DatabaseResetHook(driver), FileCleanupHook("/tmp/agent_workspace")]
runner = EvalRunner(agent=agent, graders=graders, hooks=hooks)
```

Hook errors are caught and logged — they never break trial execution.

---

## Timeouts

### Runner-Level Default

```python
runner = EvalRunner(agent=agent, graders=graders, default_timeout=120)
```

### Per-Task Override

```yaml
tasks:
  - id: complex_analysis
    question: "Run a full genomic analysis pipeline..."
    metadata:
      timeout: 300  # 5 minutes for this specific task
```

Per-task `metadata.timeout` overrides the runner's `default_timeout`. If no timeout is set at either level, the trial runs without a time limit.

Timeouts use `ThreadPoolExecutor` — the agent thread continues running after timeout but the result is discarded.

---

## Transcript Analysis

Debug agent behavior using the transcript analysis utilities:

```python
from bioagenteval.transcript_analysis import (
    summarize_transcript,
    extract_tool_sequence,
    detect_retries,
)

# After running a trial:
trial = results[0].trials[0]
transcript = trial.transcript

# Summary: event counts, timing, error detection
summary = summarize_transcript(transcript)
# {"total_events": 5, "event_counts": {"llm_call": 2, "tool_call": 3},
#  "duration_ms": 1234.5, "has_errors": False}

# Tool sequence: ordered list of what the agent did
sequence = extract_tool_sequence(transcript)
# ["llm_call:turn_0", "tool_call:search_pubmed", "llm_call:turn_1"]

# Retry detection: find agent looping
retries = detect_retries(transcript, threshold=3)
# [{"event_type": "tool_call", "event_name": "search_pubmed", "count": 5, "start_index": 2}]
```

---

## Evaluation Reports

Reports are generated as JSON with per-task results and an overall summary:

```python
report = EvalReporter.generate_report(suite.name, results)
```

### Report Structure

```json
{
  "suite_name": "my_suite",
  "run_id": "uuid",
  "timestamp": "2026-02-25T...",
  "eval_type": "capability",
  "results": [
    {
      "task_id": "task_1",
      "pass_at_1": 0.67,
      "pass_hat_1": 0.33,
      "mean_scores": {"code": 0.85, "model": 0.70},
      "num_trials": 3,
      "convergence": {"pass_rate": 0.67, "ci_width": 0.38, "converged": false},
      "trials": [...]
    }
  ],
  "summary": {
    "total_tasks": 10,
    "overall_pass_at_1": 0.75,
    "overall_pass_hat_1": 0.60,
    "latency_p50_ms": 1234,
    "latency_p95_ms": 5678,
    "saturated": false
  }
}
```

### Key Metrics

| Metric | Description |
|---|---|
| `pass_at_1` | Unbiased probability of >= 1 pass in 1 trial |
| `pass_hat_1` | Probability that ALL 1 trial passes (consistency) |
| `convergence` | Wilson score confidence interval — `converged=true` when CI width < 0.3 |
| `saturated` | True when overall pass@1 >= 95% (suite should be promoted to regression) |

### Comparing Runs

```bash
bioagenteval diff results/baseline.json results/new_agent.json
```

Shows per-task and overall pass@1 deltas between two reports.

---

## The Built-in Baseline Agent

BioAgentEval includes a reference baseline at `src/bioagenteval/agents/baseline_qa.py`:

```python
from bioagenteval.agents.baseline_qa import BaselineQAAgent
```

**What it does:**
- Single-turn GPT-4o completion with a biomedical system prompt.
- Records `llm_call` and `llm_response` events with token counts.
- Stateless `reset()`.

**Use it for:**
- Verifying your evaluation suite works end-to-end.
- Establishing a baseline pass rate to compare custom agents against.
- Testing grader configurations before building complex agents.

```bash
bioagenteval run tasks/hle_bio_chem.yaml \
  -a bioagenteval.agents.baseline_qa:BaselineQAAgent \
  -o results/baseline.json
```

---

## Best Practices

### Recording Rich Transcripts

The more detail you capture in the transcript, the more diagnostic power your evaluation has:

```python
# Record token usage for cost metrics
events.append(TranscriptEvent(
    event_type="llm_call",
    data={
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "model": self.model,
    },
))

# Record tool calls for tool_calls checks
events.append(TranscriptEvent(
    event_type="tool_call",
    event_name="search_pubmed",  # Matches tool_calls expected_output
    data={"query": search_query, "results_count": len(results)},
))

# Record state snapshots for state_check
events.append(TranscriptEvent(
    event_type="state_snapshot",
    data={"ui.screen": "results", "patient.id": "P001"},
))

# Record test results for test_results checks
events.append(TranscriptEvent(
    event_type="test_result",
    data={"test_name": "test_loading", "passed": True},
))
```

### Stateful Reset

If your agent maintains state (conversation history, cached results, database connections), clear it properly in `reset()`:

```python
def reset(self) -> None:
    self.conversation.clear()     # Clear chat history
    self.cache = {}               # Clear caches
    self.tool_results.clear()     # Clear tool output history
    # DON'T close API clients — they're reused across trials
```

The runner calls `reset()` before every trial to ensure independence between runs.

### Handling Errors Gracefully

If your agent raises an exception during `run()`, the runner catches it and records:
- `outcome = ""` (empty string — all checks will likely fail)
- `error = str(exception)` in the trial result

To provide partial results even on error, catch exceptions within `run()` and return whatever you have:

```python
def run(self, question: str) -> AgentResponse:
    events = []
    try:
        answer = self._process(question, events)
    except Exception as e:
        answer = f"Error: {e}"
        events.append(TranscriptEvent(
            event_type="error",
            data={"error": str(e)},
        ))

    transcript = Transcript(task_id="", events=events)
    return AgentResponse(outcome=answer, transcript=transcript)
```

### Parallel Execution

For independent tasks, enable parallel execution:

```python
runner = EvalRunner(
    agent=agent,
    graders=graders,
    max_concurrency=4,
)
```

Your agent must be thread-safe if `max_concurrency > 1`. Each trial calls `agent.reset()` then `agent.run()`. If your agent uses shared mutable state, add locks or create per-thread instances.

Hooks must also be thread-safe when using parallel execution.
