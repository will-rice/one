"""Unified client for multiple LLM providers."""

from typing import Any, Literal, Type, Union

from pydantic import BaseModel

from one.providers.anthropic import AnthropicProvider
from one.providers.openai import OpenAIProvider

ProviderType = Literal["openai", "anthropic"]


class Model:
    """Unified model client that supports multiple LLM providers.

    This class provides a simple interface to generate text and structured outputs
    from different LLM providers (OpenAI, Anthropic) using a consistent API.

    Example:
        Basic text generation:
        ```python
        model = Model(provider="openai")
        response = model.generate("What is the capital of France?")
        ```

        Structured output generation:
        ```python
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        model = Model(provider="openai")
        person = model.generate_structured(
            "Extract person info: John is 30 years old",
            response_format=Person
        )
        ```
    """

    def __init__(
        self,
        provider: ProviderType = "openai",
        api_key: str | None = None,
    ) -> None:
        """Initialize the model with a specific provider.

        Args:
            provider: The LLM provider to use ("openai" or "anthropic")
            api_key: Optional API key for the provider. If not provided,
                will use environment variables.
        """
        self.provider_name = provider
        self._provider: Union[OpenAIProvider, AnthropicProvider]

        if provider == "openai":
            self._provider = OpenAIProvider(api_key=api_key)
        elif provider == "anthropic":
            self._provider = AnthropicProvider(api_key=api_key)
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Supported providers: openai, anthropic"
            )

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The input prompt
            model: Model identifier. If not provided, uses provider default.
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        if model is None:
            # Use provider defaults
            if self.provider_name == "openai":
                model = "gpt-4o-mini"
            elif self.provider_name == "anthropic":
                model = "claude-3-5-sonnet-20241022"

        return self._provider.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def generate_structured(
        self,
        prompt: str,
        response_format: Type[BaseModel],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate a structured output using a Pydantic model.

        Args:
            prompt: The input prompt
            response_format: Pydantic model class for structured output
            model: Model identifier. If not provided, uses provider default.
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Instance of the response_format model with parsed data
        """
        if model is None:
            # Use provider defaults
            if self.provider_name == "openai":
                model = "gpt-4o-mini"
            elif self.provider_name == "anthropic":
                model = "claude-3-5-sonnet-20241022"

        return self._provider.generate_structured(
            prompt=prompt,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
