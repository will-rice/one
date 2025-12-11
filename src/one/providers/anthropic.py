"""Anthropic provider implementation."""

import json
import os
from typing import Any, Type

from anthropic import Anthropic
from pydantic import BaseModel

from one.providers.base import Provider


class AnthropicProvider(Provider):
    """Anthropic provider for text generation and structured outputs.

    This provider uses the Anthropic Claude API to generate completions.
    It supports both regular text generation and structured outputs
    using Pydantic models.
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize the Anthropic provider.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY
                environment variable.
        """
        super().__init__(model, api_key)
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

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
            max_tokens: Maximum tokens to generate (default: 1024)
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            Generated text (str) if response_format is None,
            or instance of the response_format model if provided
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS

        if response_format is None:
            # Plain text generation
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return response.content[0].text
        else:
            # Structured output generation
            # Get JSON schema from Pydantic model
            schema = response_format.model_json_schema()

            # Create a prompt that instructs the model to return JSON
            system_prompt = (
                f"You must respond with valid JSON that matches this schema:\n"
                f"{json.dumps(schema, indent=2)}\n\n"
                f"Only return the JSON object, no other text."
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Parse the response as JSON and validate with Pydantic
            response_text = response.content[0].text
            return response_format.model_validate_json(response_text)
