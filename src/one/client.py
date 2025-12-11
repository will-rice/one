"""Unified client for multiple LLM providers."""

from typing import Any, Type, Union

from pydantic import BaseModel

from one.providers.anthropic import AnthropicProvider
from one.providers.openai import OpenAIProvider


def _detect_provider(model: str) -> str:
    """Detect provider from model name.

    Args:
        model: Model identifier

    Returns:
        Provider name ("openai" or "anthropic")

    Raises:
        ValueError: If provider cannot be detected from model name
    """
    model_lower = model.lower()
    if model_lower.startswith(("gpt-", "o1-", "text-")):
        return "openai"
    elif model_lower.startswith("claude-"):
        return "anthropic"
    else:
        raise ValueError(
            f"Cannot detect provider from model name: {model}. "
            f"Model names should start with 'gpt-', 'o1-', 'text-' (OpenAI) "
            f"or 'claude-' (Anthropic)"
        )


class Model:
    """Unified model client that supports multiple LLM providers.

    This class provides a simple interface to generate text and structured outputs
    from different LLM providers (OpenAI, Anthropic) using a consistent API.
    The provider is automatically detected from the model name.

    Example:
        Basic text generation:
        ```python
        model = Model(model="gpt-4o-mini")
        response = model.generate("What is the capital of France?")
        ```

        Structured output generation:
        ```python
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        model = Model(model="gpt-4o-mini")
        person = model.generate_structured(
            "Extract person info: John is 30 years old",
            response_format=Person
        )
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        """Initialize the model with a specific model name.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022").
                Provider is automatically detected from the model name.
            api_key: Optional API key for the provider. If not provided,
                will use environment variables.
        """
        self.model = model
        self.provider_name = _detect_provider(model)
        self._provider: Union[OpenAIProvider, AnthropicProvider]

        if self.provider_name == "openai":
            self._provider = OpenAIProvider(api_key=api_key)
        elif self.provider_name == "anthropic":
            self._provider = AnthropicProvider(api_key=api_key)
        else:
            raise ValueError(
                f"Unknown provider: {self.provider_name}. "
                f"Supported providers: openai, anthropic"
            )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        return self._provider.generate(
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def generate_structured(
        self,
        prompt: str,
        response_format: Type[BaseModel],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate a structured output using a Pydantic model.

        Args:
            prompt: The input prompt
            response_format: Pydantic model class for structured output
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Instance of the response_format model with parsed data
        """
        return self._provider.generate_structured(
            prompt=prompt,
            model=self.model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
