"""LLM-based rubric grader using OpenAI GPT-4o."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from bioagenteval.graders.base import BaseGrader
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

# Load .env from project root (parent of src/)
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

logger = logging.getLogger(__name__)

GRADING_PROMPT = """\
You are evaluating an AI agent's response to a biomedical question.

## Task
Question: {question}

## Expected Output
{expected_output_section}

## Agent's Response
{outcome}

## Rubric
{rubric}
{metrics_section}
## Instructions
Score the response from 0.0 to 1.0 based on the rubric above.
Respond ONLY with valid JSON (no markdown fences):
{{"score": <float 0.0-1.0>, "passed": <bool>, "reasoning": "<brief explanation>"}}
"""


def _format_expected_output(task: Task) -> str:
    """Format expected_output items for the grading prompt."""
    if not task.expected_output:
        return "N/A"
    parts = []
    for eo in task.expected_output:
        parts.append(f"- Type: {eo.type}, Value: {eo.value}")
    return "\n".join(parts)


def _format_metrics(metrics: dict[str, Any] | None) -> str:
    """Format metrics section for the grading prompt."""
    if not metrics:
        return ""
    lines = ["\n## Execution Metrics"]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


class ModelGrader(BaseGrader):
    """LLM-based grader that scores responses against a rubric."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI()

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        prompt = GRADING_PROMPT.format(
            question=task.question,
            expected_output_section=_format_expected_output(task),
            outcome=outcome,
            rubric=config.rubric or "Is the response accurate and complete?",
            metrics_section=_format_metrics(metrics),
        )

        try:
            response = self.client.chat.completions.create(
                model=config.params.get("model", self.model),
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content
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
