import pytest
from unittest.mock import MagicMock, patch

from bioagenteval.graders.llm_client import (
    AnthropicLLMClient,
    LLMClient,
    OpenAILLMClient,
    create_llm_client,
)


class TestOpenAILLMClient:
    @patch("bioagenteval.graders.llm_client.OpenAILLMClient.__init__", return_value=None)
    def test_implements_protocol(self, _):
        client = OpenAILLMClient.__new__(OpenAILLMClient)
        assert isinstance(client, LLMClient)

    @patch("openai.OpenAI")
    def test_complete_basic(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hello world"))]
        )
        client = OpenAILLMClient()
        result = client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        )
        assert result == "hello world"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"

    @patch("openai.OpenAI")
    def test_complete_with_system(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))]
        )
        client = OpenAILLMClient()
        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
            model="gpt-4o",
        )
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    @patch("openai.OpenAI")
    def test_complete_default_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        client = OpenAILLMClient()
        client.complete(messages=[{"role": "user", "content": "hi"}])
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"


class TestAnthropicLLMClient:
    @patch("anthropic.Anthropic")
    def test_complete_basic(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_block = MagicMock()
        mock_block.text = "hello from claude"
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        client = AnthropicLLMClient()
        result = client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-20250514",
        )
        assert result == "hello from claude"

    @patch("anthropic.Anthropic")
    def test_complete_with_system(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_block = MagicMock()
        mock_block.text = "response"
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        client = AnthropicLLMClient()
        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
            model="claude-sonnet-4-20250514",
        )
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["system"] == "You are helpful."
        # System should NOT be in messages
        messages = call_args.kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)

    @patch("anthropic.Anthropic")
    def test_complete_without_system(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_block = MagicMock()
        mock_block.text = "response"
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        client = AnthropicLLMClient()
        client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-20250514",
        )
        call_args = mock_client.messages.create.call_args
        assert "system" not in call_args.kwargs

    @patch("anthropic.Anthropic")
    def test_default_model(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_block = MagicMock()
        mock_block.text = "ok"
        mock_client.messages.create.return_value = MagicMock(content=[mock_block])

        client = AnthropicLLMClient()
        client.complete(messages=[{"role": "user", "content": "hi"}])
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-20250514"


class TestCreateLLMClient:
    @patch("openai.OpenAI")
    def test_create_openai(self, _):
        client = create_llm_client("openai")
        assert isinstance(client, OpenAILLMClient)

    @patch("anthropic.Anthropic")
    def test_create_anthropic(self, _):
        client = create_llm_client("anthropic")
        assert isinstance(client, AnthropicLLMClient)

    def test_create_invalid_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_llm_client("invalid")

    @patch("openai.OpenAI")
    def test_default_is_openai(self, _):
        client = create_llm_client()
        assert isinstance(client, OpenAILLMClient)
