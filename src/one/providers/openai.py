"""OpenAI provider implementation."""

import os
from typing import Any, Type

from openai import OpenAI
from pydantic import BaseModel

from one.providers.base import Provider


class OpenAIProvider(Provider):
    """OpenAI provider for text generation and structured outputs.

    This provider uses the OpenAI API to generate completions.
    It supports both regular text generation and structured outputs
    using Pydantic models.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY
                environment variable.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The input prompt
            model: Model identifier (default: gpt-4o-mini)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        model: str,
        response_format: Type[BaseModel],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate a structured output using a Pydantic model.

        Args:
            prompt: The input prompt
            model: Model identifier
            response_format: Pydantic model class for structured output
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Instance of the response_format model with parsed data
        """
        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return completion.choices[0].message.parsed  # type: ignore[return-value]
