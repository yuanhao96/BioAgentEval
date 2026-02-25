"""LLM-based rubric grader supporting OpenAI and Anthropic providers."""
from __future__ import annotations

import json
import logging
from typing import Any

from bioagenteval.graders.base import BaseGrader, format_expected_output
from bioagenteval.graders.llm_client import LLMClient, create_llm_client
from bioagenteval.models import GradeResult, GraderConfig, Task, Transcript

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

If you cannot determine a score because the information is insufficient or \
the response is too ambiguous to evaluate fairly, you may return an "unknown" \
verdict instead of guessing.

Respond ONLY with valid JSON (no markdown fences):
{{"score": <float 0.0-1.0>, "passed": <bool>, "verdict": "pass"|"fail"|"unknown", "reasoning": "<brief explanation>"}}
"""


def _format_metrics(metrics: dict[str, Any] | None) -> str:
    """Format metrics section for the grading prompt."""
    if not metrics:
        return ""
    lines = ["\n## Execution Metrics"]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


class ModelGrader(BaseGrader):
    """LLM-based grader that scores responses against a rubric.

    Supports both OpenAI and Anthropic providers via the LLMClient abstraction.
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
        prompt = GRADING_PROMPT.format(
            question=task.question,
            expected_output_section=format_expected_output(task),
            outcome=outcome,
            rubric=config.rubric or "Is the response accurate and complete?",
            metrics_section=_format_metrics(metrics),
        )

        model = config.params.get("model", self.model)

        try:
            raw = self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=256,
            )
            parsed = json.loads(raw)

            verdict = parsed.get("verdict", "")
            if verdict == "unknown":
                return GradeResult(
                    grader_type="model",
                    score=0.0,
                    passed=False,
                    details={
                        "status": "unknown",
                        "reasoning": parsed.get("reasoning", ""),
                    },
                )

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
