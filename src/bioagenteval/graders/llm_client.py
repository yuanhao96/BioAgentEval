"""LLM client abstraction for multi-provider model grading."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM API clients used by model graders."""

    def complete(
        self,
        messages: list[dict],
        *,
        system: str = "",
        model: str = "",
        max_tokens: int = 256,
    ) -> str:
        """Send a completion request and return the response text."""
        ...


class OpenAILLMClient:
    """LLM client wrapping the OpenAI API."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, **kwargs):
        from openai import OpenAI
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        messages: list[dict],
        *,
        system: str = "",
        model: str = "",
        max_tokens: int = 256,
    ) -> str:
        all_messages = list(messages)
        if system:
            all_messages.insert(0, {"role": "system", "content": system})
        response = self._client.chat.completions.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=all_messages,
        )
        return response.choices[0].message.content


class AnthropicLLMClient:
    """LLM client wrapping the Anthropic API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, **kwargs):
        from anthropic import Anthropic
        self._client = Anthropic(**kwargs)

    def complete(
        self,
        messages: list[dict],
        *,
        system: str = "",
        model: str = "",
        max_tokens: int = 256,
    ) -> str:
        kwargs = {
            "model": model or self.DEFAULT_MODEL,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text


def create_llm_client(provider: str = "openai", **kwargs) -> LLMClient:
    """Factory to create an LLM client for the given provider.

    Args:
        provider: "openai" or "anthropic"
        **kwargs: Passed to the underlying client constructor.

    Returns:
        An LLMClient instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    if provider == "openai":
        return OpenAILLMClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicLLMClient(**kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Supported providers: 'openai', 'anthropic'"
        )
