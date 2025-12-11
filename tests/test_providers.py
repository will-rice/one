"""Tests for LLM providers."""

from unittest.mock import MagicMock, Mock, patch

from pydantic import BaseModel

from one.providers.anthropic import AnthropicProvider
from one.providers.openai import OpenAIProvider


class Person(BaseModel):
    """Test model for structured outputs."""

    name: str
    age: int


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @patch("one.providers.openai.OpenAI")
    def test_init_with_api_key(self, mock_openai: Mock) -> None:
        """Test initialization with explicit API key."""
        OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    @patch("one.providers.openai.OpenAI")
    def test_init_with_env_key(self, mock_openai: Mock) -> None:
        """Test initialization with environment variable."""
        OpenAIProvider(model="gpt-4o-mini")
        mock_openai.assert_called_once_with(api_key="env-key")

    @patch("one.providers.openai.OpenAI")
    def test_generate(self, mock_openai: Mock) -> None:
        """Test text generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Paris"
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        result = provider.generate("What is the capital of France?")

        # Verify
        assert result == "Paris"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["messages"][0]["content"] == "What is the capital of France?"

    @patch("one.providers.openai.OpenAI")
    def test_generate_with_custom_params(self, mock_openai: Mock) -> None:
        """Test text generation with custom parameters."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(model="gpt-4", api_key="test-key")
        result = provider.generate(
            "Test prompt",
            temperature=0.5,
            max_tokens=100,
        )

        # Verify
        assert result == "Response"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @patch("one.providers.openai.OpenAI")
    def test_generate_structured(self, mock_openai: Mock) -> None:
        """Test structured output generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_parsed = Person(name="John", age=30)
        mock_response = MagicMock()
        mock_response.choices[0].message.parsed = mock_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_response

        # Test
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        result = provider.generate(
            "Extract: John is 30",
            response_format=Person,
        )

        # Verify
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        mock_client.beta.chat.completions.parse.assert_called_once()


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @patch("one.providers.anthropic.Anthropic")
    def test_init_with_api_key(self, mock_anthropic: Mock) -> None:
        """Test initialization with explicit API key."""
        AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="test-key")
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"})
    @patch("one.providers.anthropic.Anthropic")
    def test_init_with_env_key(self, mock_anthropic: Mock) -> None:
        """Test initialization with environment variable."""
        AnthropicProvider(model="claude-3-5-sonnet-20241022")
        mock_anthropic.assert_called_once_with(api_key="env-key")

    @patch("one.providers.anthropic.Anthropic")
    def test_generate(self, mock_anthropic: Mock) -> None:
        """Test text generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content[0].text = "Paris"
        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(
            model="claude-3-5-sonnet-20241022", api_key="test-key"
        )
        result = provider.generate("What is the capital of France?")

        # Verify
        assert result == "Paris"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1024

    @patch("one.providers.anthropic.Anthropic")
    def test_generate_with_custom_params(self, mock_anthropic: Mock) -> None:
        """Test text generation with custom parameters."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content[0].text = "Response"
        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(
            model="claude-3-opus-20240229", api_key="test-key"
        )
        result = provider.generate(
            "Test prompt",
            temperature=0.5,
            max_tokens=200,
        )

        # Verify
        assert result == "Response"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-opus-20240229"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 200

    @patch("one.providers.anthropic.Anthropic")
    def test_generate_structured(self, mock_anthropic: Mock) -> None:
        """Test structured output generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content[0].text = '{"name": "John", "age": 30}'
        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(
            model="claude-3-5-sonnet-20241022", api_key="test-key"
        )
        result = provider.generate(
            "Extract: John is 30",
            response_format=Person,
        )

        # Verify
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "JSON" in call_kwargs["system"]
