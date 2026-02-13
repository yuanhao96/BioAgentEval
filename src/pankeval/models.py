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
