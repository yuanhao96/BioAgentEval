# Evaluation Harness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete evaluation harness for biomedical QA agents following [Anthropic's eval guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), with YAML task loading, agent wrapping via protocol, multi-grader evaluation (code/model/human), full trajectory capture, and structured JSON output. Includes a baseline GPT-based biomedical QA agent for testing.

**Architecture:** The harness separates concerns into: (1) data models (Pydantic v2), (2) a YAML task loader, (3) an `AgentHarness` protocol that wraps any agent without code changes, (4) pluggable graders (code, model, human), (5) an `EvalRunner` orchestrator, and (6) a JSON reporter. The baseline agent implements the `AgentHarness` protocol using OpenAI's API. The CLI (Click) ties everything together.

**Tech Stack:** Python 3.9+, Pydantic v2, Click, PyYAML, OpenAI SDK (baseline agent), Anthropic SDK (model grader), pytest + pytest-mock

---

## File Layout

```
src/pankeval/
  __init__.py              # Package init
  __main__.py              # Click CLI group + commands
  models.py                # All Pydantic data models
  loader.py                # YAML -> EvalSuite + list[Task]
  harness.py               # AgentHarness Protocol + AgentResponse model
  runner.py                # EvalRunner: orchestrates suite execution
  reporter.py              # Structured JSON output
  graders/
    __init__.py             # Re-exports CodeGrader, ModelGrader, HumanGrader
    base.py                 # BaseGrader ABC
    code_grader.py          # Deterministic: entity_presence, cypher_pattern
    model_grader.py         # LLM rubric grading via Anthropic Claude
    human_grader.py         # Stub for human calibration
  agents/
    __init__.py             # Re-exports BaselineQAAgent
    baseline_qa.py          # Baseline GPT biomedical QA agent
tasks/
  biomedical_core.yaml      # Example evaluation suite
tests/
  conftest.py               # sys.path setup
  test_models.py            # Unit tests for all data models
  test_loader.py            # Unit tests for YAML loader
  test_harness.py           # Unit tests for harness protocol compliance
  test_code_grader.py       # Unit tests for deterministic grading
  test_model_grader.py      # Unit tests for LLM grading (mocked)
  test_runner.py            # Unit tests for eval orchestration (mocked)
  test_reporter.py          # Unit tests for JSON output
  test_baseline_agent.py    # Unit tests for baseline agent (mocked)
  test_integration.py       # End-to-end integration test (all mocked)
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/pankeval/__init__.py`
- Create: `src/pankeval/__main__.py`
- Create: `tests/conftest.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/pankeval/graders src/pankeval/agents tests tasks docs/plans
```

**Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pankeval"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "pydantic>=2.0",
    "PyYAML>=6.0",
    "click>=8.0",
    "anthropic>=0.66.0",
    "openai>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock>=3.10"]

[project.scripts]
pankeval = "pankeval.__main__:cli"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 3: Write `src/pankeval/__init__.py`**

```python
"""PanKgraph Agent Evaluation Harness."""
```

**Step 4: Write `src/pankeval/__main__.py`**

```python
import click


@click.group()
def cli():
    """PanKgraph evaluation harness."""
    pass


if __name__ == "__main__":
    cli()
```

**Step 5: Write `tests/conftest.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

**Step 6: Write empty `__init__.py` files for subpackages**

```python
# src/pankeval/graders/__init__.py
"""Grader implementations."""

# src/pankeval/agents/__init__.py
"""Agent implementations."""
```

**Step 7: Install and verify**

Run: `pip install -e ".[dev]"`
Run: `pytest --collect-only`
Expected: 0 tests collected, no errors

**Step 8: Commit**

```bash
git add pyproject.toml src/ tests/conftest.py docs/
git commit -m "feat: project scaffolding with pankeval package"
```

---

### Task 2: Core Data Models

**Files:**
- Create: `src/pankeval/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing tests**

Write `tests/test_models.py`:

```python
import pytest
from pankeval.models import (
    Task, GraderConfig, EvalSuite,
    TranscriptEvent, Transcript,
    GradeResult, TrialResult, EvalResult,
    AgentResponse,
)


class TestGraderConfig:
    def test_defaults(self):
        gc = GraderConfig(type="code")
        assert gc.type == "code"
        assert gc.checks == []
        assert gc.rubric == ""
        assert gc.weight == 1.0
        assert gc.params == {}

    def test_full_config(self):
        gc = GraderConfig(
            type="model",
            rubric="Is the answer complete?",
            weight=0.5,
            params={"model": "claude-sonnet-4-5-20250929"},
        )
        assert gc.weight == 0.5


class TestTask:
    def test_minimal_task(self):
        t = Task(
            id="t1",
            question="What genes are associated with type 1 diabetes?",
        )
        assert t.id == "t1"
        assert t.question == "What genes are associated with type 1 diabetes?"
        assert t.expected_entities == []
        assert t.graders == []
        assert t.metadata == {}
        assert t.num_trials == 1

    def test_full_task(self):
        t = Task(
            id="t2",
            question="Tell me about INS gene",
            expected_entities=["INS", "ENSG00000254647"],
            expected_complexity="simple",
            graders=[
                GraderConfig(type="code", checks=["entity_presence"]),
                GraderConfig(type="model", rubric="Is the answer complete?"),
            ],
            metadata={"category": "entity_overview"},
            num_trials=3,
        )
        assert len(t.graders) == 2
        assert t.graders[0].type == "code"
        assert t.num_trials == 3

    def test_task_requires_question(self):
        with pytest.raises(Exception):
            Task(id="t3")


class TestTranscript:
    def test_empty_transcript(self):
        tr = Transcript(task_id="t1")
        assert tr.events == []
        assert tr.task_id == "t1"

    def test_add_events(self):
        tr = Transcript(task_id="t1")
        ev = TranscriptEvent(
            event_type="stream_event",
            event_name="complexity_classified",
            data={"complexity": "simple"},
        )
        tr.events.append(ev)
        assert len(tr.events) == 1
        assert tr.events[0].event_name == "complexity_classified"

    def test_transcript_records_cypher_queries(self):
        tr = Transcript(task_id="t1")
        ev = TranscriptEvent(
            event_type="cypher_query",
            data={"query": "MATCH (g:Gene) RETURN g LIMIT 5"},
        )
        tr.events.append(ev)
        assert tr.events[0].data["query"].startswith("MATCH")


class TestGradeResult:
    def test_grade_result(self):
        g = GradeResult(
            grader_type="code",
            score=0.8,
            passed=True,
            details={"entity_presence": True},
        )
        assert g.score == 0.8
        assert g.passed is True

    def test_score_bounds(self):
        g = GradeResult(grader_type="code", score=0.0, passed=False)
        assert g.score == 0.0
        g2 = GradeResult(grader_type="code", score=1.0, passed=True)
        assert g2.score == 1.0


class TestTrialResult:
    def test_trial_result(self):
        tr = TrialResult(
            task_id="t1",
            trial_num=0,
            outcome="Some response text",
            transcript=Transcript(task_id="t1"),
            grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
            duration_ms=1234.5,
        )
        assert tr.trial_num == 0
        assert tr.duration_ms == 1234.5


class TestEvalResult:
    def test_eval_result_aggregation(self):
        trials = [
            TrialResult(
                task_id="t1",
                trial_num=i,
                outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=s, passed=s >= 0.5)],
                duration_ms=1000.0,
            )
            for i, s in enumerate([1.0, 0.0, 1.0])
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_at_k(k=1) > 0.0
        assert er.pass_at_k(k=3) > 0.0
        assert er.mean_score("code") == pytest.approx(2.0 / 3.0)

    def test_pass_at_k_edge_cases(self):
        er = EvalResult(task_id="t1", trials=[])
        assert er.pass_at_k(k=1) == 0.0
        assert er.pass_at_k(k=0) == 0.0

    def test_pass_at_k_all_pass(self):
        trials = [
            TrialResult(
                task_id="t1",
                trial_num=i,
                outcome="answer",
                transcript=Transcript(task_id="t1"),
                grades=[GradeResult(grader_type="code", score=1.0, passed=True)],
                duration_ms=100.0,
            )
            for i in range(5)
        ]
        er = EvalResult(task_id="t1", trials=trials)
        assert er.pass_at_k(k=1) == 1.0

    def test_mean_score_missing_grader(self):
        er = EvalResult(task_id="t1", trials=[])
        assert er.mean_score("nonexistent") == 0.0


class TestAgentResponse:
    def test_agent_response(self):
        resp = AgentResponse(
            outcome="INS gene is associated with diabetes",
            transcript=Transcript(task_id="t1"),
        )
        assert resp.outcome == "INS gene is associated with diabetes"
        assert resp.transcript.task_id == "t1"


class TestEvalSuite:
    def test_suite(self):
        s = EvalSuite(
            name="core",
            description="Core capability tests",
            task_ids=["t1", "t2"],
        )
        assert len(s.task_ids) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL (ImportError — `models.py` doesn't exist yet)

**Step 3: Write minimal implementation**

Write `src/pankeval/models.py`:

```python
"""Core data models for the evaluation harness."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class GraderConfig(BaseModel):
    """Configuration for a single grader attached to a task."""
    type: str
    checks: list[str] = Field(default_factory=list)
    rubric: str = ""
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A single evaluation task (test case)."""
    id: str
    question: str
    expected_entities: list[str] = Field(default_factory=list)
    expected_complexity: str | None = None
    expected_cypher_patterns: list[str] = Field(default_factory=list)
    graders: list[GraderConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    num_trials: int = 1


class EvalSuite(BaseModel):
    """A named collection of tasks."""
    name: str
    description: str = ""
    task_ids: list[str] = Field(default_factory=list)
    default_graders: list[GraderConfig] = Field(default_factory=list)
    default_num_trials: int = 1


class TranscriptEvent(BaseModel):
    """A single event in the agent's trajectory."""
    event_type: str
    event_name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Transcript(BaseModel):
    """Full trajectory of one trial: all intermediate steps."""
    task_id: str
    events: list[TranscriptEvent] = Field(default_factory=list)
    cypher_queries: list[str] = Field(default_factory=list)
    neo4j_results: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class GradeResult(BaseModel):
    """Output from a single grader."""
    grader_type: str
    score: float
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class TrialResult(BaseModel):
    """Result of one trial (one attempt at a task)."""
    task_id: str
    trial_num: int
    outcome: str
    transcript: Transcript
    grades: list[GradeResult] = Field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None


class EvalResult(BaseModel):
    """Aggregated result across all trials of one task."""
    task_id: str
    trials: list[TrialResult] = Field(default_factory=list)

    def pass_at_k(self, k: int = 1) -> float:
        """Unbiased estimator: probability of >= 1 pass in k trials."""
        n = len(self.trials)
        if n == 0 or k <= 0:
            return 0.0
        c = sum(1 for t in self.trials if all(g.passed for g in t.grades))
        if k > n:
            k = n
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

    def mean_score(self, grader_type: str) -> float:
        """Mean score across trials for a specific grader type."""
        scores = [
            g.score
            for t in self.trials
            for g in t.grades
            if g.grader_type == grader_type
        ]
        return sum(scores) / len(scores) if scores else 0.0


class AgentResponse(BaseModel):
    """Structured response returned by an agent harness."""
    outcome: str
    transcript: Transcript
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All 17 tests PASS

**Step 5: Commit**

```bash
git add src/pankeval/models.py tests/test_models.py
git commit -m "feat: core data models with pass@k and agent response"
```

---

### Task 3: Task Loader (YAML -> Models)

**Files:**
- Create: `src/pankeval/loader.py`
- Create: `tests/test_loader.py`
- Create: `tasks/biomedical_core.yaml` (example suite)

**Step 1: Write the example YAML task suite**

Write `tasks/biomedical_core.yaml`:

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

  - id: pathway_insulin_signaling
    question: "What are the key genes in the insulin signaling pathway?"
    expected_entities:
      - INSR
      - IRS1
      - PIK3CA
      - AKT1
    expected_complexity: complex
    expected_cypher_patterns:
      - "MATCH.*Pathway.*insulin"
    graders:
      - type: code
        checks:
          - entity_presence
          - cypher_pattern
      - type: model
        rubric: >
          Does the answer describe the insulin signaling pathway accurately,
          mentioning key receptor and downstream signaling molecules?
    num_trials: 3
```

**Step 2: Write the failing tests**

Write `tests/test_loader.py`:

```python
import pytest
import tempfile
from pathlib import Path

from pankeval.loader import load_suite
from pankeval.models import EvalSuite, Task


MINIMAL_YAML = """\
name: test_suite
description: A test suite
tasks:
  - id: t1
    question: "What is gene X?"
"""

FULL_YAML = """\
name: full_suite
description: Full featured suite
default_num_trials: 5
tasks:
  - id: t1
    question: "What genes cause diabetes?"
    expected_entities:
      - INS
      - HLA-DRB1
    expected_complexity: complex
    graders:
      - type: code
        checks:
          - entity_presence
      - type: model
        rubric: "Is it correct?"
    num_trials: 3
  - id: t2
    question: "Tell me about INS."
    expected_entities:
      - INS
    num_trials: 2
"""


class TestLoadSuite:
    def test_load_minimal(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(MINIMAL_YAML)
        suite, tasks = load_suite(f)
        assert isinstance(suite, EvalSuite)
        assert suite.name == "test_suite"
        assert len(tasks) == 1
        assert tasks[0].id == "t1"
        assert tasks[0].question == "What is gene X?"

    def test_load_full(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(FULL_YAML)
        suite, tasks = load_suite(f)
        assert suite.name == "full_suite"
        assert suite.default_num_trials == 5
        assert len(tasks) == 2
        assert tasks[0].expected_entities == ["INS", "HLA-DRB1"]
        assert len(tasks[0].graders) == 2
        assert tasks[0].num_trials == 3
        assert tasks[1].num_trials == 2
        assert suite.task_ids == ["t1", "t2"]

    def test_load_applies_default_num_trials(self, tmp_path):
        yaml_content = """\
name: defaults
default_num_trials: 7
tasks:
  - id: t1
    question: "Q?"
"""
        f = tmp_path / "suite.yaml"
        f.write_text(yaml_content)
        suite, tasks = load_suite(f)
        assert tasks[0].num_trials == 7

    def test_task_explicit_trials_override_default(self, tmp_path):
        yaml_content = """\
name: override
default_num_trials: 7
tasks:
  - id: t1
    question: "Q?"
    num_trials: 2
"""
        f = tmp_path / "suite.yaml"
        f.write_text(yaml_content)
        _, tasks = load_suite(f)
        assert tasks[0].num_trials == 2

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_suite(Path("/nonexistent/path.yaml"))

    def test_load_string_path(self, tmp_path):
        f = tmp_path / "suite.yaml"
        f.write_text(MINIMAL_YAML)
        suite, tasks = load_suite(str(f))
        assert suite.name == "test_suite"
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_loader.py -v`
Expected: FAIL (ImportError — `loader.py` doesn't exist)

**Step 4: Write minimal implementation**

Write `src/pankeval/loader.py`:

```python
"""Load evaluation suites and tasks from YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pankeval.models import EvalSuite, GraderConfig, Task


def load_suite(path: str | Path) -> tuple[EvalSuite, list[Task]]:
    """Load an evaluation suite from a YAML file.

    Returns (EvalSuite, list[Task]).  Tasks whose YAML block omits
    ``num_trials`` inherit ``default_num_trials`` from the suite.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    default_trials = raw.get("default_num_trials", 1)

    tasks: list[Task] = []
    task_ids: list[str] = []
    for t_raw in raw.get("tasks", []):
        # Apply default num_trials if not explicitly set
        if "num_trials" not in t_raw:
            t_raw["num_trials"] = default_trials

        # Parse grader configs
        grader_dicts = t_raw.pop("graders", [])
        graders = [GraderConfig(**g) for g in grader_dicts]

        task = Task(**t_raw, graders=graders)
        tasks.append(task)
        task_ids.append(task.id)

    suite = EvalSuite(
        name=raw["name"],
        description=raw.get("description", ""),
        task_ids=task_ids,
        default_num_trials=default_trials,
    )
    return suite, tasks
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_loader.py -v`
Expected: All 6 tests PASS

**Step 6: Commit**

```bash
git add src/pankeval/loader.py tests/test_loader.py tasks/biomedical_core.yaml
git commit -m "feat: YAML task loader with default_num_trials inheritance"
```

---

### Task 4: Agent Harness Protocol

**Files:**
- Create: `src/pankeval/harness.py`
- Create: `tests/test_harness.py`

**Step 1: Write the failing tests**

Write `tests/test_harness.py`:

```python
import pytest
from pankeval.harness import AgentHarness
from pankeval.models import AgentResponse, Task, Transcript


class FakeAgent:
    """A trivial agent that satisfies the AgentHarness protocol."""
    def __init__(self):
        self.call_count = 0

    def run(self, question: str) -> AgentResponse:
        self.call_count += 1
        return AgentResponse(
            outcome=f"Answer to: {question}",
            transcript=Transcript(task_id="fake"),
        )

    def reset(self) -> None:
        self.call_count = 0


class BadAgent:
    """Missing the run method."""
    def reset(self) -> None:
        pass


class TestAgentHarness:
    def test_fake_agent_satisfies_protocol(self):
        agent = FakeAgent()
        assert isinstance(agent, AgentHarness)

    def test_bad_agent_does_not_satisfy(self):
        agent = BadAgent()
        assert not isinstance(agent, AgentHarness)

    def test_fake_agent_run(self):
        agent = FakeAgent()
        resp = agent.run("What is INS?")
        assert resp.outcome == "Answer to: What is INS?"
        assert resp.transcript.task_id == "fake"
        assert agent.call_count == 1

    def test_fake_agent_reset(self):
        agent = FakeAgent()
        agent.run("Q1")
        agent.run("Q2")
        assert agent.call_count == 2
        agent.reset()
        assert agent.call_count == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_harness.py -v`
Expected: FAIL (ImportError — `harness.py` doesn't exist)

**Step 3: Write minimal implementation**

Write `src/pankeval/harness.py`:

```python
"""Agent harness protocol for wrapping agents without code changes."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pankeval.models import AgentResponse


@runtime_checkable
class AgentHarness(Protocol):
    """Protocol that any agent must satisfy for evaluation.

    Implement ``run`` and ``reset`` on your agent class. No inheritance
    required — structural subtyping (duck typing) is used.
    """

    def run(self, question: str) -> AgentResponse:
        """Run the agent on a question, return structured response."""
        ...

    def reset(self) -> None:
        """Reset agent state between trials."""
        ...
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_harness.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/pankeval/harness.py tests/test_harness.py
git commit -m "feat: AgentHarness protocol for wrapping agents"
```

---

### Task 5: Grader Infrastructure + Code Grader

**Files:**
- Create: `src/pankeval/graders/base.py`
- Create: `src/pankeval/graders/code_grader.py`
- Modify: `src/pankeval/graders/__init__.py`
- Create: `tests/test_code_grader.py`

**Step 1: Write the failing tests**

Write `tests/test_code_grader.py`:

```python
import pytest
from pankeval.graders.base import BaseGrader
from pankeval.graders.code_grader import CodeGrader
from pankeval.models import (
    GraderConfig, GradeResult, Task, Transcript, TranscriptEvent,
)


class TestCodeGraderIsBaseGrader:
    def test_inherits_base(self):
        grader = CodeGrader()
        assert isinstance(grader, BaseGrader)


class TestEntityPresence:
    def make_task(self, entities):
        return Task(
            id="t1",
            question="Q?",
            expected_entities=entities,
            graders=[GraderConfig(type="code", checks=["entity_presence"])],
        )

    def test_all_entities_present(self):
        task = self.make_task(["INS", "HLA-DRB1"])
        outcome = "The INS gene and HLA-DRB1 are associated with diabetes."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0
        assert result.grader_type == "code"

    def test_partial_entities(self):
        task = self.make_task(["INS", "HLA-DRB1", "PTPN22"])
        outcome = "The INS gene is important."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == pytest.approx(1.0 / 3.0)

    def test_no_entities_present(self):
        task = self.make_task(["INS", "HLA-DRB1"])
        outcome = "I don't know."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is False
        assert result.score == 0.0

    def test_case_insensitive_match(self):
        task = self.make_task(["ins", "hla-drb1"])
        outcome = "The INS gene and HLA-DRB1 are relevant."
        config = task.graders[0]
        result = CodeGrader().grade(task, outcome, Transcript(task_id="t1"), config)
        assert result.passed is True

    def test_no_expected_entities(self):
        task = self.make_task([])
        config = GraderConfig(type="code", checks=["entity_presence"])
        result = CodeGrader().grade(task, "anything", Transcript(task_id="t1"), config)
        assert result.passed is True
        assert result.score == 1.0


class TestCypherPattern:
    def make_task_with_cypher(self, patterns):
        return Task(
            id="t1",
            question="Q?",
            expected_cypher_patterns=patterns,
            graders=[GraderConfig(type="code", checks=["cypher_pattern"])],
        )

    def test_cypher_pattern_found(self):
        task = self.make_task_with_cypher(["MATCH.*Gene"])
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": "MATCH (g:Gene) RETURN g"},
                )
            ],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "answer", transcript, config)
        assert result.passed is True
        assert result.score == 1.0

    def test_cypher_pattern_not_found(self):
        task = self.make_task_with_cypher(["MATCH.*Pathway"])
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": "MATCH (g:Gene) RETURN g"},
                )
            ],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "answer", transcript, config)
        assert result.passed is False

    def test_no_cypher_events(self):
        task = self.make_task_with_cypher(["MATCH.*Gene"])
        transcript = Transcript(task_id="t1", events=[])
        config = task.graders[0]
        result = CodeGrader().grade(task, "answer", transcript, config)
        assert result.passed is False
        assert result.score == 0.0


class TestMultipleChecks:
    def test_combined_checks(self):
        task = Task(
            id="t1",
            question="Q?",
            expected_entities=["INS"],
            expected_cypher_patterns=["MATCH.*Gene"],
            graders=[
                GraderConfig(
                    type="code",
                    checks=["entity_presence", "cypher_pattern"],
                )
            ],
        )
        transcript = Transcript(
            task_id="t1",
            events=[
                TranscriptEvent(
                    event_type="cypher_query",
                    data={"query": "MATCH (g:Gene {name:'INS'}) RETURN g"},
                )
            ],
        )
        config = task.graders[0]
        result = CodeGrader().grade(task, "INS is a gene", transcript, config)
        assert result.passed is True
        assert result.score == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_code_grader.py -v`
Expected: FAIL (ImportError)

**Step 3: Write `src/pankeval/graders/base.py`**

```python
"""Base grader abstract class."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pankeval.models import GradeResult, GraderConfig, Task, Transcript


class BaseGrader(ABC):
    """Abstract base for all grader implementations."""

    @abstractmethod
    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        """Grade an agent's output for one trial.

        Args:
            task: The evaluation task with expected answers.
            outcome: The agent's final answer text.
            transcript: Full trajectory of the trial.
            config: The GraderConfig for this grader invocation.

        Returns:
            GradeResult with score, passed, and details.
        """
        ...
```

**Step 4: Write `src/pankeval/graders/code_grader.py`**

```python
"""Deterministic code-based grader: entity presence, Cypher pattern matching."""
from __future__ import annotations

import re

from pankeval.graders.base import BaseGrader
from pankeval.models import GradeResult, GraderConfig, Task, Transcript


class CodeGrader(BaseGrader):
    """Deterministic grader using string matching and pattern checks."""

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        check_results: dict[str, float] = {}

        for check in config.checks:
            if check == "entity_presence":
                check_results[check] = self._check_entity_presence(task, outcome)
            elif check == "cypher_pattern":
                check_results[check] = self._check_cypher_pattern(task, transcript)

        if not check_results:
            score = 1.0
        else:
            score = sum(check_results.values()) / len(check_results)

        return GradeResult(
            grader_type="code",
            score=score,
            passed=score >= 0.5,
            details=check_results,
        )

    def _check_entity_presence(self, task: Task, outcome: str) -> float:
        if not task.expected_entities:
            return 1.0
        outcome_lower = outcome.lower()
        found = sum(
            1 for e in task.expected_entities if e.lower() in outcome_lower
        )
        return found / len(task.expected_entities)

    def _check_cypher_pattern(self, task: Task, transcript: Transcript) -> float:
        if not task.expected_cypher_patterns:
            return 1.0
        cypher_queries = [
            ev.data.get("query", "")
            for ev in transcript.events
            if ev.event_type == "cypher_query"
        ]
        all_cypher = " ".join(cypher_queries)
        matched = sum(
            1
            for pat in task.expected_cypher_patterns
            if re.search(pat, all_cypher, re.IGNORECASE)
        )
        return matched / len(task.expected_cypher_patterns)
```

**Step 5: Update `src/pankeval/graders/__init__.py`**

```python
"""Grader implementations."""
from pankeval.graders.code_grader import CodeGrader

__all__ = ["CodeGrader"]
```

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_code_grader.py -v`
Expected: All 11 tests PASS

**Step 7: Commit**

```bash
git add src/pankeval/graders/ tests/test_code_grader.py
git commit -m "feat: BaseGrader ABC + CodeGrader with entity/cypher checks"
```

---

### Task 6: Model Grader (LLM-Based Rubric Scoring)

**Files:**
- Create: `src/pankeval/graders/model_grader.py`
- Modify: `src/pankeval/graders/__init__.py`
- Create: `tests/test_model_grader.py`

**Step 1: Write the failing tests**

Write `tests/test_model_grader.py`:

```python
import json
import pytest
from unittest.mock import MagicMock, patch

from pankeval.graders.base import BaseGrader
from pankeval.graders.model_grader import ModelGrader
from pankeval.models import GradeResult, GraderConfig, Task, Transcript


def _mock_anthropic_response(score: float, passed: bool, reasoning: str):
    """Build a mock Anthropic messages.create response."""
    content_text = json.dumps({
        "score": score,
        "passed": passed,
        "reasoning": reasoning,
    })
    mock_block = MagicMock()
    mock_block.text = content_text
    mock_resp = MagicMock()
    mock_resp.content = [mock_block]
    return mock_resp


class TestModelGraderIsBaseGrader:
    def test_inherits_base(self):
        with patch("pankeval.graders.model_grader.anthropic"):
            grader = ModelGrader()
        assert isinstance(grader, BaseGrader)


class TestModelGraderGrade:
    def setup_method(self):
        self.task = Task(
            id="t1",
            question="What genes are associated with diabetes?",
            expected_entities=["INS", "HLA-DRB1"],
            graders=[
                GraderConfig(
                    type="model",
                    rubric="Is the answer complete and accurate?",
                )
            ],
        )
        self.transcript = Transcript(task_id="t1")
        self.config = self.task.graders[0]

    @patch("pankeval.graders.model_grader.anthropic")
    def test_passing_grade(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response(
            0.9, True, "The answer is comprehensive."
        )

        grader = ModelGrader()
        result = grader.grade(
            self.task, "INS and HLA-DRB1 are key genes.", self.transcript, self.config
        )
        assert result.passed is True
        assert result.score == 0.9
        assert result.grader_type == "model"
        assert "reasoning" in result.details

    @patch("pankeval.graders.model_grader.anthropic")
    def test_failing_grade(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response(
            0.2, False, "The answer is incomplete."
        )

        grader = ModelGrader()
        result = grader.grade(
            self.task, "I don't know.", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.2

    @patch("pankeval.graders.model_grader.anthropic")
    def test_api_error_returns_zero(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        grader = ModelGrader()
        result = grader.grade(
            self.task, "answer", self.transcript, self.config
        )
        assert result.passed is False
        assert result.score == 0.0
        assert "error" in result.details

    @patch("pankeval.graders.model_grader.anthropic")
    def test_custom_model(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response(
            0.8, True, "Good."
        )

        grader = ModelGrader(model="claude-opus-4-6")
        result = grader.grade(
            self.task, "answer", self.transcript, self.config
        )
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_grader.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Write `src/pankeval/graders/model_grader.py`:

```python
"""LLM-based rubric grader using Anthropic Claude."""
from __future__ import annotations

import json
import logging

import anthropic

from pankeval.graders.base import BaseGrader
from pankeval.models import GradeResult, GraderConfig, Task, Transcript

logger = logging.getLogger(__name__)

GRADING_PROMPT = """\
You are evaluating an AI agent's response to a biomedical question.

## Task
Question: {question}
Expected entities: {expected_entities}

## Agent's Response
{outcome}

## Rubric
{rubric}

## Instructions
Score the response from 0.0 to 1.0 based on the rubric above.
Respond ONLY with valid JSON (no markdown fences):
{{"score": <float 0.0-1.0>, "passed": <bool>, "reasoning": "<brief explanation>"}}
"""


class ModelGrader(BaseGrader):
    """LLM-based grader that scores responses against a rubric."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.model = model
        self.client = anthropic.Anthropic()

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        prompt = GRADING_PROMPT.format(
            question=task.question,
            expected_entities=", ".join(task.expected_entities) or "N/A",
            outcome=outcome,
            rubric=config.rubric or "Is the response accurate and complete?",
        )

        try:
            response = self.client.messages.create(
                model=config.params.get("model", self.model),
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            parsed = json.loads(raw)
            return GradeResult(
                grader_type="model",
                score=float(parsed["score"]),
                passed=bool(parsed["passed"]),
                details={"reasoning": parsed.get("reasoning", "")},
            )
        except Exception as e:
            logger.warning("ModelGrader failed: %s", e)
            return GradeResult(
                grader_type="model",
                score=0.0,
                passed=False,
                details={"error": str(e)},
            )
```

**Step 4: Update `src/pankeval/graders/__init__.py`**

```python
"""Grader implementations."""
from pankeval.graders.code_grader import CodeGrader
from pankeval.graders.model_grader import ModelGrader

__all__ = ["CodeGrader", "ModelGrader"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_model_grader.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/pankeval/graders/ tests/test_model_grader.py
git commit -m "feat: ModelGrader with Anthropic Claude rubric scoring"
```

---

### Task 7: Human Grader Stub

**Files:**
- Create: `src/pankeval/graders/human_grader.py`
- Modify: `src/pankeval/graders/__init__.py`
- Create: `tests/test_human_grader.py`

**Step 1: Write the failing tests**

Write `tests/test_human_grader.py`:

```python
import pytest
from pankeval.graders.base import BaseGrader
from pankeval.graders.human_grader import HumanGrader
from pankeval.models import GraderConfig, Task, Transcript


class TestHumanGrader:
    def test_inherits_base(self):
        grader = HumanGrader()
        assert isinstance(grader, BaseGrader)

    def test_returns_pending_result(self):
        task = Task(id="t1", question="Q?")
        config = GraderConfig(type="human")
        result = HumanGrader().grade(
            task, "some answer", Transcript(task_id="t1"), config
        )
        assert result.grader_type == "human"
        assert result.passed is False
        assert result.score == 0.0
        assert "pending" in result.details.get("status", "")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_human_grader.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Write `src/pankeval/graders/human_grader.py`:

```python
"""Human grader stub — returns pending results for manual review."""
from __future__ import annotations

import logging

from pankeval.graders.base import BaseGrader
from pankeval.models import GradeResult, GraderConfig, Task, Transcript

logger = logging.getLogger(__name__)


class HumanGrader(BaseGrader):
    """Stub grader that flags results for human review."""

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
    ) -> GradeResult:
        logger.info("Task %s flagged for human review", task.id)
        return GradeResult(
            grader_type="human",
            score=0.0,
            passed=False,
            details={
                "status": "pending_human_review",
                "task_id": task.id,
                "outcome_preview": outcome[:200],
            },
        )
```

**Step 4: Update `src/pankeval/graders/__init__.py`**

```python
"""Grader implementations."""
from pankeval.graders.code_grader import CodeGrader
from pankeval.graders.human_grader import HumanGrader
from pankeval.graders.model_grader import ModelGrader

__all__ = ["CodeGrader", "HumanGrader", "ModelGrader"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_human_grader.py -v`
Expected: All 2 tests PASS

**Step 6: Commit**

```bash
git add src/pankeval/graders/ tests/test_human_grader.py
git commit -m "feat: HumanGrader stub for manual review calibration"
```

---

### Task 8: Eval Runner

**Files:**
- Create: `src/pankeval/runner.py`
- Create: `tests/test_runner.py`

**Step 1: Write the failing tests**

Write `tests/test_runner.py`:

```python
import pytest
from unittest.mock import MagicMock

from pankeval.models import (
    AgentResponse, EvalResult, GradeResult, GraderConfig,
    Task, Transcript, TranscriptEvent, TrialResult,
)
from pankeval.runner import EvalRunner


class FakeAgent:
    """Deterministic fake agent for testing the runner."""
    def __init__(self, answer: str = "Test answer"):
        self.answer = answer
        self.run_count = 0

    def run(self, question: str) -> AgentResponse:
        self.run_count += 1
        return AgentResponse(
            outcome=self.answer,
            transcript=Transcript(
                task_id="fake",
                events=[
                    TranscriptEvent(
                        event_type="llm_call",
                        data={"question": question},
                    )
                ],
            ),
        )

    def reset(self) -> None:
        pass


class FakeGrader:
    """Always-passing grader for testing."""
    def grade(self, task, outcome, transcript, config):
        return GradeResult(grader_type="code", score=1.0, passed=True)


class FailingGrader:
    """Always-failing grader for testing."""
    def grade(self, task, outcome, transcript, config):
        return GradeResult(grader_type="code", score=0.0, passed=False)


class TestEvalRunner:
    def test_run_single_task_single_trial(self):
        task = Task(
            id="t1",
            question="What is INS?",
            graders=[GraderConfig(type="code", checks=["entity_presence"])],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(),
            graders={"code": FakeGrader()},
        )
        result = runner.run_task(task)
        assert isinstance(result, EvalResult)
        assert result.task_id == "t1"
        assert len(result.trials) == 1
        assert result.trials[0].outcome == "Test answer"
        assert result.trials[0].grades[0].passed is True

    def test_run_multiple_trials(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=3,
        )
        agent = FakeAgent()
        runner = EvalRunner(agent=agent, graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert len(result.trials) == 3
        for i, trial in enumerate(result.trials):
            assert trial.trial_num == i

    def test_run_task_calls_reset_between_trials(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=3,
        )
        agent = MagicMock()
        agent.run.return_value = AgentResponse(
            outcome="answer",
            transcript=Transcript(task_id="t1"),
        )
        runner = EvalRunner(agent=agent, graders={"code": FakeGrader()})
        runner.run_task(task)
        assert agent.reset.call_count == 3

    def test_run_suite(self):
        tasks = [
            Task(id="t1", question="Q1?", graders=[GraderConfig(type="code")], num_trials=2),
            Task(id="t2", question="Q2?", graders=[GraderConfig(type="code")], num_trials=1),
        ]
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        results = runner.run_suite(tasks)
        assert len(results) == 2
        assert results[0].task_id == "t1"
        assert len(results[0].trials) == 2
        assert results[1].task_id == "t2"
        assert len(results[1].trials) == 1

    def test_duration_is_recorded(self):
        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(agent=FakeAgent(), graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert result.trials[0].duration_ms >= 0.0

    def test_agent_error_captured(self):
        agent = MagicMock()
        agent.run.side_effect = RuntimeError("Agent crashed")
        agent.reset.return_value = None
        task = Task(
            id="t1", question="Q?",
            graders=[GraderConfig(type="code")],
            num_trials=1,
        )
        runner = EvalRunner(agent=agent, graders={"code": FakeGrader()})
        result = runner.run_task(task)
        assert len(result.trials) == 1
        assert result.trials[0].error == "Agent crashed"
        assert result.trials[0].outcome == ""

    def test_multiple_graders_per_task(self):
        task = Task(
            id="t1",
            question="Q?",
            graders=[
                GraderConfig(type="code", checks=["entity_presence"]),
                GraderConfig(type="model", rubric="Is it good?"),
            ],
            num_trials=1,
        )
        runner = EvalRunner(
            agent=FakeAgent(),
            graders={
                "code": FakeGrader(),
                "model": FailingGrader(),
            },
        )
        result = runner.run_task(task)
        assert len(result.trials[0].grades) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Write `src/pankeval/runner.py`:

```python
"""Evaluation runner: orchestrates running tasks against an agent."""
from __future__ import annotations

import logging
import time
from typing import Any

from pankeval.harness import AgentHarness
from pankeval.models import EvalResult, Task, Transcript, TrialResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates running an evaluation suite against an agent.

    Args:
        agent: Any object satisfying the AgentHarness protocol.
        graders: Mapping of grader type name to grader instance.
                 Each grader must have a ``grade(task, outcome, transcript, config)`` method.
    """

    def __init__(self, agent: AgentHarness, graders: dict[str, Any]):
        self.agent = agent
        self.graders = graders

    def run_suite(self, tasks: list[Task]) -> list[EvalResult]:
        """Run all tasks and return aggregated results."""
        results: list[EvalResult] = []
        for task in tasks:
            logger.info("Running task %s (%d trials)", task.id, task.num_trials)
            result = self.run_task(task)
            results.append(result)
        return results

    def run_task(self, task: Task) -> EvalResult:
        """Run all trials for a single task."""
        trials: list[TrialResult] = []
        for trial_num in range(task.num_trials):
            trial = self._run_trial(task, trial_num)
            trials.append(trial)
        return EvalResult(task_id=task.id, trials=trials)

    def _run_trial(self, task: Task, trial_num: int) -> TrialResult:
        """Run a single trial: reset -> run agent -> grade."""
        self.agent.reset()

        start = time.perf_counter()
        try:
            response = self.agent.run(task.question)
            duration_ms = (time.perf_counter() - start) * 1000
            outcome = response.outcome
            transcript = response.transcript
            error = None
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning("Trial %d of task %s failed: %s", trial_num, task.id, e)
            outcome = ""
            transcript = Transcript(task_id=task.id)
            error = str(e)

        grades = []
        for grader_config in task.graders:
            grader = self.graders.get(grader_config.type)
            if grader is None:
                logger.warning("No grader registered for type: %s", grader_config.type)
                continue
            grade = grader.grade(task, outcome, transcript, grader_config)
            grades.append(grade)

        return TrialResult(
            task_id=task.id,
            trial_num=trial_num,
            outcome=outcome,
            transcript=transcript,
            grades=grades,
            duration_ms=duration_ms,
            error=error,
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/pankeval/runner.py tests/test_runner.py
git commit -m "feat: EvalRunner orchestrator with trial isolation and error capture"
```

---

### Task 9: Reporter (Structured JSON Output)

**Files:**
- Create: `src/pankeval/reporter.py`
- Create: `tests/test_reporter.py`

**Step 1: Write the failing tests**

Write `tests/test_reporter.py`:

```python
import json
import pytest
from pathlib import Path

from pankeval.models import (
    EvalResult, GradeResult, Transcript, TrialResult,
)
from pankeval.reporter import EvalReporter


def _make_result(task_id: str, scores: list[float]) -> EvalResult:
    trials = [
        TrialResult(
            task_id=task_id,
            trial_num=i,
            outcome=f"answer_{i}",
            transcript=Transcript(task_id=task_id),
            grades=[GradeResult(grader_type="code", score=s, passed=s >= 0.5)],
            duration_ms=100.0 * (i + 1),
        )
        for i, s in enumerate(scores)
    ]
    return EvalResult(task_id=task_id, trials=trials)


class TestEvalReporter:
    def test_generate_report_structure(self):
        results = [
            _make_result("t1", [1.0, 0.0, 1.0]),
            _make_result("t2", [0.0]),
        ]
        report = EvalReporter.generate_report("test_suite", results)
        assert report["suite_name"] == "test_suite"
        assert "run_id" in report
        assert "timestamp" in report
        assert len(report["results"]) == 2
        assert "summary" in report

    def test_per_task_metrics(self):
        results = [_make_result("t1", [1.0, 0.0, 1.0])]
        report = EvalReporter.generate_report("s", results)
        task_report = report["results"][0]
        assert task_report["task_id"] == "t1"
        assert "pass_at_1" in task_report
        assert "mean_scores" in task_report
        assert task_report["num_trials"] == 3
        assert len(task_report["trials"]) == 3

    def test_summary_aggregation(self):
        results = [
            _make_result("t1", [1.0, 1.0]),
            _make_result("t2", [0.0, 0.0]),
        ]
        report = EvalReporter.generate_report("s", results)
        summary = report["summary"]
        assert summary["total_tasks"] == 2
        assert 0.0 <= summary["overall_pass_at_1"] <= 1.0

    def test_save_report(self, tmp_path):
        results = [_make_result("t1", [1.0])]
        out_file = tmp_path / "report.json"
        EvalReporter.save_report("s", results, out_file)
        assert out_file.exists()
        loaded = json.loads(out_file.read_text())
        assert loaded["suite_name"] == "s"

    def test_trials_include_outcome_and_grades(self):
        results = [_make_result("t1", [0.8])]
        report = EvalReporter.generate_report("s", results)
        trial = report["results"][0]["trials"][0]
        assert trial["outcome"] == "answer_0"
        assert len(trial["grades"]) == 1
        assert trial["grades"][0]["score"] == 0.8
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Write `src/pankeval/reporter.py`:

```python
"""Structured JSON report generation for evaluation results."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pankeval.models import EvalResult


class EvalReporter:
    """Generates structured JSON reports from evaluation results."""

    @staticmethod
    def generate_report(
        suite_name: str,
        results: list[EvalResult],
    ) -> dict[str, Any]:
        """Generate a full evaluation report as a dict."""
        task_reports = []
        pass_at_1_values = []

        for result in results:
            p1 = result.pass_at_k(k=1)
            pass_at_1_values.append(p1)

            grader_types = {
                g.grader_type for t in result.trials for g in t.grades
            }
            mean_scores = {
                gt: result.mean_score(gt) for gt in sorted(grader_types)
            }

            trial_dicts = []
            for trial in result.trials:
                trial_dicts.append({
                    "trial_num": trial.trial_num,
                    "outcome": trial.outcome,
                    "grades": [g.model_dump() for g in trial.grades],
                    "transcript": trial.transcript.model_dump(mode="json"),
                    "duration_ms": trial.duration_ms,
                    "error": trial.error,
                })

            task_reports.append({
                "task_id": result.task_id,
                "pass_at_1": p1,
                "mean_scores": mean_scores,
                "num_trials": len(result.trials),
                "trials": trial_dicts,
            })

        overall_pass_at_1 = (
            sum(pass_at_1_values) / len(pass_at_1_values)
            if pass_at_1_values
            else 0.0
        )

        return {
            "suite_name": suite_name,
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": task_reports,
            "summary": {
                "total_tasks": len(results),
                "overall_pass_at_1": overall_pass_at_1,
            },
        }

    @staticmethod
    def save_report(
        suite_name: str,
        results: list[EvalResult],
        path: str | Path,
    ) -> None:
        """Generate and save a report to a JSON file."""
        report = EvalReporter.generate_report(suite_name, results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reporter.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/pankeval/reporter.py tests/test_reporter.py
git commit -m "feat: EvalReporter with JSON output and summary aggregation"
```

---

### Task 10: CLI Integration

**Files:**
- Modify: `src/pankeval/__main__.py`

**Step 1: Write the updated CLI**

Modify `src/pankeval/__main__.py`:

```python
import json
import logging
from pathlib import Path

import click

from pankeval.graders import CodeGrader, HumanGrader, ModelGrader
from pankeval.loader import load_suite
from pankeval.reporter import EvalReporter
from pankeval.runner import EvalRunner


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose):
    """PanKgraph evaluation harness."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@cli.command()
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--agent", "-a", required=True, help="Agent module path, e.g. pankeval.agents.baseline_qa:BaselineQAAgent")
@click.option("--output", "-o", default="eval_report.json", help="Output JSON path.")
@click.option("--skip-model-grader", is_flag=True, help="Skip model-based grading.")
def run(suite_path, agent, output, skip_model_grader):
    """Run an evaluation suite against an agent."""
    # Load suite
    suite, tasks = load_suite(suite_path)
    click.echo(f"Loaded suite '{suite.name}' with {len(tasks)} tasks")

    # Import agent class
    module_path, class_name = agent.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    agent_cls = getattr(mod, class_name)
    agent_instance = agent_cls()

    # Set up graders
    graders = {"code": CodeGrader()}
    if not skip_model_grader:
        graders["model"] = ModelGrader()
    graders["human"] = HumanGrader()

    # Run
    runner = EvalRunner(agent=agent_instance, graders=graders)
    results = runner.run_suite(tasks)

    # Report
    EvalReporter.save_report(suite.name, results, output)
    click.echo(f"Report saved to {output}")

    # Print summary
    report = EvalReporter.generate_report(suite.name, results)
    summary = report["summary"]
    click.echo(f"Tasks: {summary['total_tasks']}, Overall pass@1: {summary['overall_pass_at_1']:.2%}")


@cli.command()
@click.argument("suite_path", type=click.Path(exists=True))
def validate(suite_path):
    """Validate a task suite YAML file."""
    suite, tasks = load_suite(suite_path)
    click.echo(f"Suite: {suite.name}")
    click.echo(f"Tasks: {len(tasks)}")
    for t in tasks:
        grader_types = [g.type for g in t.graders]
        click.echo(f"  {t.id}: {t.num_trials} trials, graders={grader_types}")
    click.echo("Validation passed.")


if __name__ == "__main__":
    cli()
```

**Step 2: Verify CLI help works**

Run: `pankeval --help`
Expected: Shows help with `run` and `validate` commands

Run: `pankeval validate tasks/biomedical_core.yaml`
Expected: Shows suite info and "Validation passed."

**Step 3: Commit**

```bash
git add src/pankeval/__main__.py
git commit -m "feat: CLI with run and validate commands"
```

---

### Task 11: Baseline Biomedical QA Agent

**Files:**
- Create: `src/pankeval/agents/baseline_qa.py`
- Modify: `src/pankeval/agents/__init__.py`
- Create: `tests/test_baseline_agent.py`

**Step 1: Write the failing tests**

Write `tests/test_baseline_agent.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

from pankeval.agents.baseline_qa import BaselineQAAgent
from pankeval.harness import AgentHarness
from pankeval.models import AgentResponse, Transcript


def _mock_openai_response(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 50
    return mock_resp


class TestBaselineQAAgent:
    @patch("pankeval.agents.baseline_qa.openai")
    def test_satisfies_harness_protocol(self, mock_openai_mod):
        agent = BaselineQAAgent()
        assert isinstance(agent, AgentHarness)

    @patch("pankeval.agents.baseline_qa.openai")
    def test_run_returns_agent_response(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "INS gene encodes insulin."
        )

        agent = BaselineQAAgent()
        resp = agent.run("What is the INS gene?")
        assert isinstance(resp, AgentResponse)
        assert resp.outcome == "INS gene encodes insulin."
        assert resp.transcript.task_id == "baseline"

    @patch("pankeval.agents.baseline_qa.openai")
    def test_transcript_captures_events(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "Answer here."
        )

        agent = BaselineQAAgent()
        resp = agent.run("Q?")
        events = resp.transcript.events
        assert len(events) >= 1
        assert events[0].event_type == "llm_call"
        assert "question" in events[0].data

    @patch("pankeval.agents.baseline_qa.openai")
    def test_reset_clears_state(self, mock_openai_mod):
        agent = BaselineQAAgent()
        agent.reset()  # Should not raise

    @patch("pankeval.agents.baseline_qa.openai")
    def test_custom_model(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response("A")

        agent = BaselineQAAgent(model="gpt-4o")
        agent.run("Q?")
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_baseline_agent.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

Write `src/pankeval/agents/baseline_qa.py`:

```python
"""Baseline biomedical QA agent using OpenAI GPT."""
from __future__ import annotations

from datetime import datetime, timezone

import openai

from pankeval.models import AgentResponse, Transcript, TranscriptEvent

SYSTEM_PROMPT = """\
You are a biomedical knowledge assistant. Answer questions about genes,
diseases, pathways, and their relationships accurately and concisely.
When relevant, mention specific gene names, identifiers (e.g., ENSEMBL IDs),
and known associations. If you are uncertain, say so.
"""


class BaselineQAAgent:
    """Simple single-turn biomedical QA agent using OpenAI GPT.

    Implements the AgentHarness protocol (run + reset) so it can be
    plugged directly into EvalRunner.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()

    def run(self, question: str) -> AgentResponse:
        """Send question to GPT and return structured response."""
        started_at = datetime.now(timezone.utc)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )

        answer = response.choices[0].message.content
        finished_at = datetime.now(timezone.utc)

        transcript = Transcript(
            task_id="baseline",
            events=[
                TranscriptEvent(
                    event_type="llm_call",
                    event_name="chat_completion",
                    data={
                        "question": question,
                        "model": self.model,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                    timestamp=started_at,
                ),
                TranscriptEvent(
                    event_type="llm_response",
                    event_name="chat_completion_result",
                    data={"answer": answer},
                    timestamp=finished_at,
                ),
            ],
            started_at=started_at,
            finished_at=finished_at,
        )

        return AgentResponse(outcome=answer, transcript=transcript)

    def reset(self) -> None:
        """No persistent state to clear in this simple agent."""
        pass
```

**Step 4: Update `src/pankeval/agents/__init__.py`**

```python
"""Agent implementations."""
from pankeval.agents.baseline_qa import BaselineQAAgent

__all__ = ["BaselineQAAgent"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_baseline_agent.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/pankeval/agents/ tests/test_baseline_agent.py
git commit -m "feat: BaselineQAAgent with GPT and AgentHarness protocol"
```

---

### Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

Write `tests/test_integration.py`:

```python
"""End-to-end integration test: load suite -> run agent -> grade -> report."""
import json
import pytest

from pankeval.graders import CodeGrader
from pankeval.loader import load_suite
from pankeval.models import AgentResponse, Transcript, TranscriptEvent
from pankeval.reporter import EvalReporter
from pankeval.runner import EvalRunner


SUITE_YAML = """\
name: integration_test
description: Minimal integration test suite
default_num_trials: 2

tasks:
  - id: gene_lookup
    question: "What genes are associated with type 1 diabetes?"
    expected_entities:
      - INS
      - HLA-DRB1
    expected_complexity: complex
    graders:
      - type: code
        checks:
          - entity_presence
    num_trials: 2
  - id: pathway_check
    question: "Describe the insulin signaling pathway."
    expected_entities:
      - INSR
      - IRS1
    expected_cypher_patterns:
      - "MATCH.*insulin"
    graders:
      - type: code
        checks:
          - entity_presence
          - cypher_pattern
    num_trials: 1
"""


class StubAgent:
    """Agent that returns a canned response with Cypher queries."""
    def run(self, question: str) -> AgentResponse:
        return AgentResponse(
            outcome="INS and HLA-DRB1 are key genes. INSR and IRS1 in pathway.",
            transcript=Transcript(
                task_id="stub",
                events=[
                    TranscriptEvent(
                        event_type="cypher_query",
                        data={"query": "MATCH (p:Pathway {name:'insulin signaling'}) RETURN p"},
                    ),
                    TranscriptEvent(
                        event_type="llm_response",
                        data={"answer": "INS and HLA-DRB1..."},
                    ),
                ],
            ),
        )

    def reset(self) -> None:
        pass


class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        # 1. Write and load suite
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(SUITE_YAML)
        suite, tasks = load_suite(suite_file)
        assert len(tasks) == 2

        # 2. Run evaluation
        runner = EvalRunner(
            agent=StubAgent(),
            graders={"code": CodeGrader()},
        )
        results = runner.run_suite(tasks)
        assert len(results) == 2

        # 3. Check results
        # Task 1: INS and HLA-DRB1 should both be found
        assert results[0].pass_at_k(k=1) == 1.0
        # Task 2: entities found, cypher pattern matched
        assert results[1].trials[0].grades[0].passed is True

        # 4. Generate and save report
        report_path = tmp_path / "report.json"
        EvalReporter.save_report(suite.name, results, report_path)

        report = json.loads(report_path.read_text())
        assert report["suite_name"] == "integration_test"
        assert report["summary"]["total_tasks"] == 2
        assert report["summary"]["overall_pass_at_1"] == 1.0
        assert len(report["results"][0]["trials"]) == 2
        assert len(report["results"][1]["trials"]) == 1

    def test_partial_match_pipeline(self, tmp_path):
        """Test that partial entity matches produce fractional scores."""
        yaml = """\
name: partial
tasks:
  - id: t1
    question: "Genes for diabetes?"
    expected_entities:
      - INS
      - MISSING_GENE
      - ANOTHER_MISSING
    graders:
      - type: code
        checks:
          - entity_presence
"""
        f = tmp_path / "partial.yaml"
        f.write_text(yaml)
        _, tasks = load_suite(f)

        runner = EvalRunner(
            agent=StubAgent(),
            graders={"code": CodeGrader()},
        )
        results = runner.run_suite(tasks)
        # Only 1/3 entities found -> score ~ 0.33, passed = False
        trial = results[0].trials[0]
        assert trial.grades[0].score == pytest.approx(1.0 / 3.0)
        assert trial.grades[0].passed is False
```

**Step 2: Run the integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All 2 tests PASS

**Step 3: Run the full test suite**

Run: `pytest -v`
Expected: All tests across all files PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: end-to-end integration tests for full eval pipeline"
```

---

## Summary of Components

| Component | File | Purpose |
|-----------|------|---------|
| Data models | `src/pankeval/models.py` | Pydantic v2 models: Task, Transcript, GradeResult, EvalResult, AgentResponse |
| Loader | `src/pankeval/loader.py` | YAML task definitions -> EvalSuite + list[Task] |
| Harness | `src/pankeval/harness.py` | `AgentHarness` Protocol — wrap any agent via duck typing |
| Code grader | `src/pankeval/graders/code_grader.py` | Deterministic: entity_presence, cypher_pattern checks |
| Model grader | `src/pankeval/graders/model_grader.py` | LLM rubric scoring via Anthropic Claude |
| Human grader | `src/pankeval/graders/human_grader.py` | Stub for manual review calibration |
| Runner | `src/pankeval/runner.py` | Orchestrates: reset -> run agent -> grade, multi-trial |
| Reporter | `src/pankeval/reporter.py` | JSON structured output with pass@k, scores, transcripts |
| CLI | `src/pankeval/__main__.py` | `pankeval run` and `pankeval validate` commands |
| Baseline agent | `src/pankeval/agents/baseline_qa.py` | GPT-based biomedical QA (implements AgentHarness) |
| Example suite | `tasks/biomedical_core.yaml` | 3 biomedical evaluation tasks |

## Usage After Implementation

```bash
# Validate a task suite
pankeval validate tasks/biomedical_core.yaml

# Run evaluation with baseline agent
pankeval run tasks/biomedical_core.yaml \
  --agent pankeval.agents.baseline_qa:BaselineQAAgent \
  --output results/eval_report.json

# Run without model grader (no Anthropic API key needed)
pankeval run tasks/biomedical_core.yaml \
  --agent pankeval.agents.baseline_qa:BaselineQAAgent \
  --skip-model-grader \
  --output results/eval_report.json

# Run all tests
pytest -v
```
