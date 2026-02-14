"""Baseline biomedical QA agent using OpenAI GPT."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import openai

from bioagenteval.models import AgentResponse, Transcript, TranscriptEvent

# Load .env from project root (parent of src/)
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

SYSTEM_PROMPT = """\
You are a biomedical knowledge assistant. Answer questions about genes,
diseases, pathways, and their relationships accurately and concisely.
When relevant, mention specific gene names, identifiers (e.g., ENSEMBL IDs),
and known associations. If you are uncertain, say so.
"""


class BaselineQAAgent:
    """Simple single-turn biomedical QA agent using OpenAI GPT."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()

    def run(self, question: str) -> AgentResponse:
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
        pass
