"""Tests for the unified Model client."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from one.client import Model


class Person(BaseModel):
    """Test model for structured outputs."""

    name: str
    age: int


class TestModel:
    """Tests for the unified Model client."""

    def test_init_openai_provider(self) -> None:
        """Test initialization with OpenAI provider."""
        with patch("one.client.OpenAIProvider") as mock_openai:
            model = Model(provider="openai", api_key="test-key")
            mock_openai.assert_called_once_with(api_key="test-key")
            assert model.provider_name == "openai"

    def test_init_anthropic_provider(self) -> None:
        """Test initialization with Anthropic provider."""
        with patch("one.client.AnthropicProvider") as mock_anthropic:
            model = Model(provider="anthropic", api_key="test-key")
            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert model.provider_name == "anthropic"

    def test_init_invalid_provider(self) -> None:
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Model(provider="invalid")  # type: ignore[arg-type]

    @patch("one.client.OpenAIProvider")
    def test_generate_openai_with_default_model(
        self, mock_provider_class: Mock
    ) -> None:
        """Test text generation with OpenAI using default model."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Paris"
        mock_provider_class.return_value = mock_provider

        model = Model(provider="openai")
        result = model.generate("What is the capital of France?")

        assert result == "Paris"
        mock_provider.generate.assert_called_once_with(
            prompt="What is the capital of France?",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=None,
        )

    @patch("one.client.AnthropicProvider")
    def test_generate_anthropic_with_default_model(
        self, mock_provider_class: Mock
    ) -> None:
        """Test text generation with Anthropic using default model."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Paris"
        mock_provider_class.return_value = mock_provider

        model = Model(provider="anthropic")
        result = model.generate("What is the capital of France?")

        assert result == "Paris"
        mock_provider.generate.assert_called_once_with(
            prompt="What is the capital of France?",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=None,
        )

    @patch("one.client.OpenAIProvider")
    def test_generate_with_custom_model(self, mock_provider_class: Mock) -> None:
        """Test text generation with custom model."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Response"
        mock_provider_class.return_value = mock_provider

        model = Model(provider="openai")
        result = model.generate(
            "Test prompt",
            model="gpt-4",
            temperature=0.5,
            max_tokens=100,
        )

        assert result == "Response"
        mock_provider.generate.assert_called_once_with(
            prompt="Test prompt",
            model="gpt-4",
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

        model = Model(provider="openai")
        result = model.generate_structured(
            "Extract: John is 30",
            response_format=Person,
        )

        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30
        mock_provider.generate_structured.assert_called_once_with(
            prompt="Extract: John is 30",
            model="gpt-4o-mini",
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

        model = Model(provider="anthropic")
        result = model.generate_structured(
            "Extract: Jane is 25",
            response_format=Person,
        )

        assert isinstance(result, Person)
        assert result.name == "Jane"
        assert result.age == 25
        mock_provider.generate_structured.assert_called_once_with(
            prompt="Extract: Jane is 25",
            model="claude-3-5-sonnet-20241022",
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

        model = Model(provider="openai")
        result = model.generate_structured(
            "Extract person",
            response_format=Person,
            model="gpt-4",
            temperature=0.3,
            max_tokens=500,
        )

        assert isinstance(result, Person)
        mock_provider.generate_structured.assert_called_once_with(
            prompt="Extract person",
            model="gpt-4",
            response_format=Person,
            temperature=0.3,
            max_tokens=500,
        )
