"""Core data models for the evaluation harness."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class GraderConfig(BaseModel):
    """Configuration for a single grader attached to a task."""
    type: str
    rubric: str = ""
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class ExpectedOutput(BaseModel):
    """A single ground-truth item, evaluated by domain-specific code graders."""
    type: str          # "entities", "cypher_patterns", "mcq_answer", "numeric_range"
    value: Any         # Ground truth, interpreted by code grader based on type
    params: dict[str, Any] = Field(default_factory=dict)


class MetricGroup(BaseModel):
    """A group of execution metrics to compute from trial data."""
    type: str          # "transcript", "latency", "cost"
    metrics: list[str] = Field(default_factory=list)


class Task(BaseModel):
    """A single evaluation task (test case)."""
    id: str
    question: str
    expected_output: list[ExpectedOutput] = Field(default_factory=list)
    tags: dict[str, Any] = Field(default_factory=dict)
    tracked_metrics: list[MetricGroup] = Field(default_factory=list)
    graders: list[GraderConfig] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    num_trials: int = 1
    should_fail: bool = False


class EvalSuite(BaseModel):
    """A named collection of tasks."""
    name: str
    description: str = ""
    eval_type: Literal["", "capability", "regression"] = ""
    task_ids: list[str] = Field(default_factory=list)
    default_graders: list[GraderConfig] = Field(default_factory=list)
    default_num_trials: int = 1
    default_tracked_metrics: list[MetricGroup] = Field(default_factory=list)


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
    weight: float = 1.0
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
    metrics: dict[str, Any] = Field(default_factory=dict)

    def _known_grades(self) -> list[GradeResult]:
        """Return grades excluding those with 'unknown' status."""
        return [
            g for g in self.grades
            if g.details.get("status") != "unknown"
        ]

    def has_unknown_grades(self) -> bool:
        """Whether any grades have 'unknown' status."""
        return any(g.details.get("status") == "unknown" for g in self.grades)

    def weighted_score(self) -> float:
        """Weighted average of grade scores using each grade's weight.

        Grades with status 'unknown' are excluded from the calculation.
        """
        known = self._known_grades()
        if not known:
            return 0.0
        total_weight = sum(g.weight for g in known)
        if total_weight == 0:
            return 0.0
        return sum(g.score * g.weight for g in known) / total_weight

    def weighted_passed(self, threshold: float = 0.5) -> bool:
        """Whether the weighted score meets the pass threshold.

        Grades with status 'unknown' are excluded from the calculation.
        """
        return self.weighted_score() >= threshold


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

    def pass_hat_k(self, k: int = 1) -> float:
        """Unbiased estimator: probability that ALL k trials pass.

        Measures consistency — critical for customer-facing agents.
        Formula: C(c, k) / C(n, k) where c = passing trials, n = total.
        """
        n = len(self.trials)
        if n == 0 or k <= 0:
            return 0.0
        c = sum(1 for t in self.trials if all(g.passed for g in t.grades))
        if k > n:
            k = n
        if c < k:
            return 0.0
        return math.comb(c, k) / math.comb(n, k)

    def convergence_check(
        self, confidence: float = 0.95, width_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Check if the pass rate has converged using Wilson score interval.

        Returns a dict with pass_rate, ci_lower, ci_upper, ci_width,
        and converged (True if ci_width < width_threshold).
        """
        n = len(self.trials)
        if n == 0:
            return {
                "pass_rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "ci_width": 1.0, "converged": False, "n_trials": 0,
            }
        c = sum(1 for t in self.trials if all(g.passed for g in t.grades))
        p = c / n

        # Wilson score interval (normal approximation)
        # z for 95% confidence = 1.96
        z = 1.96 if confidence == 0.95 else 1.645
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
        ci_lower = max(0.0, center - spread)
        ci_upper = min(1.0, center + spread)
        ci_width = ci_upper - ci_lower

        return {
            "pass_rate": p,
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "ci_width": round(ci_width, 4),
            "converged": ci_width < width_threshold,
            "n_trials": n,
        }

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
