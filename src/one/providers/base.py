"""Base provider abstract class."""

from abc import ABC, abstractmethod
from typing import Any, Type, Union

from pydantic import BaseModel


class Provider(ABC):
    """Abstract base class defining the interface for LLM providers.

    This ABC ensures all providers implement the required methods
    for generating completions with structured outputs.
    """

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Initialize the provider with a model identifier.

        Args:
            model: Model identifier for this provider
            api_key: Optional API key for the provider
        """
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        response_format: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Union[str, BaseModel]:
        """Generate a completion with optional structured output.

        Args:
            prompt: The input prompt
            response_format: Optional Pydantic model class for structured output.
                If None, returns plain text. If provided, returns structured model.
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text (str) if response_format is None,
            or instance of the response_format model if provided
        """
        ...
