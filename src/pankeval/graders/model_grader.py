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
