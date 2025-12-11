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

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize the OpenAI provider.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "gpt-4")
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY
                environment variable.
        """
        super().__init__(model, api_key)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        prompt: str,
        response_format: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str | BaseModel:
        """Generate a completion with optional structured output.

        Args:
            prompt: The input prompt
            response_format: Optional Pydantic model class for structured output.
                If None, returns plain text. If provided, returns structured model.
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated text (str) if response_format is None,
            or instance of the response_format model if provided
        """
        if response_format is None:
            # Plain text generation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        else:
            # Structured output generation
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return completion.choices[0].message.parsed  # type: ignore[return-value]
