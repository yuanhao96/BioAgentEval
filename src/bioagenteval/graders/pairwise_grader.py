"""Pairwise comparison grader: compares two agent outputs using an LLM judge."""
from __future__ import annotations

import json
import logging
from typing import Any

from bioagenteval.graders.base import BaseGrader, format_expected_output
from bioagenteval.graders.llm_client import LLMClient, create_llm_client
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT = """\
You are comparing two AI agent responses to a biomedical question.

## Task
Question: {question}

## Expected Output
{expected_output_section}

## Response A (Agent Under Test)
{outcome_a}

## Response B (Reference)
{outcome_b}

## Rubric
{rubric}

## Instructions
Compare Response A against Response B. Determine which response is better \
based on the rubric above.

Respond ONLY with valid JSON (no markdown fences):
{{"preferred": "A"|"B"|"tie", "score": <float 0.0-1.0>, "passed": <bool>, "reasoning": "<brief explanation>"}}

Scoring guide:
- If A is clearly better: score close to 1.0, passed=true
- If roughly equal (tie): score around 0.5
- If B is clearly better: score close to 0.0, passed=false
"""


class PairwiseGrader(BaseGrader):
    """Compares an agent's output against a reference output using an LLM judge.

    The reference outcome must be provided in ``config.params["reference_outcome"]``.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        provider: str = "openai",
        model: str = "",
    ):
        if client is not None:
            self.client = client
        else:
            self.client = create_llm_client(provider)
        self.model = model

    def grade(
        self,
        task: Task,
        outcome: str,
        transcript: Transcript,
        config: GraderConfig,
        metrics: dict[str, Any] | None = None,
    ) -> GradeResult:
        reference_outcome = config.params.get("reference_outcome")
        if reference_outcome is None:
            logger.warning(
                "PairwiseGrader: no reference_outcome in config.params for task %s",
                task.id,
            )
            return GradeResult(
                grader_type="pairwise",
                score=0.0,
                passed=False,
                details={"error": "missing reference_outcome in config.params"},
            )

        prompt = PAIRWISE_PROMPT.format(
            question=task.question,
            expected_output_section=format_expected_output(task),
            outcome_a=outcome,
            outcome_b=reference_outcome,
            rubric=config.rubric or "Which response is more accurate and complete?",
        )

        model = config.params.get("model", self.model)

        try:
            raw = self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=256,
            )
            parsed = json.loads(raw)
            return GradeResult(
                grader_type="pairwise",
                score=float(parsed["score"]),
                passed=bool(parsed["passed"]),
                details={
                    "preferred": parsed.get("preferred", ""),
                    "reasoning": parsed.get("reasoning", ""),
                },
            )
        except Exception as e:
            logger.warning("PairwiseGrader failed: %s", e)
            return GradeResult(
                grader_type="pairwise",
                score=0.0,
                passed=False,
                details={"error": str(e)},
            )
