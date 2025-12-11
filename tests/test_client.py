"""Tests for the unified Model client."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from one.client import Model, _detect_provider


class Person(BaseModel):
    """Test model for structured outputs."""

    name: str
    age: int


class TestDetectProvider:
    """Tests for provider detection."""

    def test_detect_openai_gpt(self) -> None:
        """Test detection of OpenAI GPT models."""
        assert _detect_provider("gpt-4o-mini") == "openai"
        assert _detect_provider("gpt-4") == "openai"
        assert _detect_provider("GPT-3.5-turbo") == "openai"

    def test_detect_openai_o1(self) -> None:
        """Test detection of OpenAI O1 models."""
        assert _detect_provider("o1-preview") == "openai"
        assert _detect_provider("o1-mini") == "openai"

    def test_detect_openai_text(self) -> None:
        """Test detection of OpenAI text models."""
        assert _detect_provider("text-davinci-003") == "openai"

    def test_detect_anthropic(self) -> None:
        """Test detection of Anthropic Claude models."""
        assert _detect_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert _detect_provider("claude-3-opus-20240229") == "anthropic"
        assert _detect_provider("CLAUDE-2.1") == "anthropic"

    def test_detect_invalid(self) -> None:
        """Test detection of invalid model names."""
        with pytest.raises(ValueError, match="Cannot detect provider"):
            _detect_provider("unknown-model")


class TestModel:
    """Tests for the unified Model client."""

    def test_init_openai_model(self) -> None:
        """Test initialization with OpenAI model."""
        with patch("one.client.OpenAIProvider") as mock_openai:
            model = Model(model="gpt-4o-mini", api_key="test-key")
            mock_openai.assert_called_once_with(model="gpt-4o-mini", api_key="test-key")
            assert model.provider_name == "openai"
            assert model.model == "gpt-4o-mini"

    def test_init_anthropic_model(self) -> None:
        """Test initialization with Anthropic model."""
        with patch("one.client.AnthropicProvider") as mock_anthropic:
            model = Model(model="claude-3-5-sonnet-20241022", api_key="test-key")
            mock_anthropic.assert_called_once_with(
                model="claude-3-5-sonnet-20241022", api_key="test-key"
            )
            assert model.provider_name == "anthropic"
            assert model.model == "claude-3-5-sonnet-20241022"

    def test_init_invalid_model(self) -> None:
        """Test initialization with invalid model name."""
        with pytest.raises(ValueError, match="Cannot detect provider"):
            Model(model="invalid-model")

    @patch("one.client.OpenAIProvider")
    def test_generate_openai(self, mock_provider_class: Mock) -> None:
        """Test text generation with OpenAI."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Paris"
        mock_provider_class.return_value = mock_provider

        model = Model(model="gpt-4o-mini")
        result = model.generate("What is the capital of France?")

        assert result == "Paris"
        mock_provider.generate.assert_called_once_with(
            prompt="What is the capital of France?",
            temperature=0.7,
            max_tokens=None,
        )

    @patch("one.client.AnthropicProvider")
    def test_generate_anthropic(self, mock_provider_class: Mock) -> None:
        """Test text generation with Anthropic."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Paris"
        mock_provider_class.return_value = mock_provider

        model = Model(model="claude-3-5-sonnet-20241022")
        result = model.generate("What is the capital of France?")

        assert result == "Paris"
        mock_provider.generate.assert_called_once_with(
            prompt="What is the capital of France?",
            temperature=0.7,
            max_tokens=None,
        )

    @patch("one.client.OpenAIProvider")
    def test_generate_with_custom_params(self, mock_provider_class: Mock) -> None:
        """Test text generation with custom parameters."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Response"
        mock_provider_class.return_value = mock_provider

        model = Model(model="gpt-4")
        result = model.generate(
            "Test prompt",
            temperature=0.5,
            max_tokens=100,
        )

        assert result == "Response"
        mock_provider.generate.assert_called_once_with(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100,
        )

    @patch("one.client.OpenAIProvider")
    def test_generate_structured_openai(self, mock_provider_class: Mock) -> None:
        """Test structured generation with OpenAI."""
        mock_provider = MagicMock()
        mock_person = Person(name="John", age=30)
        mock_provider.generate_structured.return_value = mock_person
        mock_provider_class.return_value = mock_provider

        model = Model(model="gpt-4o-mini")
        result = model.generate_structured(
            "Extract: John is 30",
            response_format=Person,
        )

        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        mock_provider.generate_structured.assert_called_once_with(
            prompt="Extract: John is 30",
            response_format=Person,
            temperature=0.7,
            max_tokens=None,
        )

    @patch("one.client.AnthropicProvider")
    def test_generate_structured_anthropic(self, mock_provider_class: Mock) -> None:
        """Test structured generation with Anthropic."""
        mock_provider = MagicMock()
        mock_person = Person(name="Jane", age=25)
        mock_provider.generate_structured.return_value = mock_person
        mock_provider_class.return_value = mock_provider

        model = Model(model="claude-3-5-sonnet-20241022")
        result = model.generate_structured(
            "Extract: Jane is 25",
            response_format=Person,
        )

        assert isinstance(result, Person)
        assert result.name == "Jane"
        assert result.age == 25
        mock_provider.generate_structured.assert_called_once_with(
            prompt="Extract: Jane is 25",
            response_format=Person,
            temperature=0.7,
            max_tokens=None,
        )

    @patch("one.client.OpenAIProvider")
    def test_generate_structured_with_custom_params(
        self, mock_provider_class: Mock
    ) -> None:
        """Test structured generation with custom parameters."""
        mock_provider = MagicMock()
        mock_person = Person(name="Bob", age=40)
        mock_provider.generate_structured.return_value = mock_person
        mock_provider_class.return_value = mock_provider

        model = Model(model="gpt-4")
        result = model.generate_structured(
            "Extract person",
            response_format=Person,
            temperature=0.3,
            max_tokens=500,
        )

        assert isinstance(result, Person)
        mock_provider.generate_structured.assert_called_once_with(
            prompt="Extract person",
            response_format=Person,
            temperature=0.3,
            max_tokens=500,
        )
