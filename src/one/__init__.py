"""One: A clean library for multiple LLM providers with structured outputs."""

from dotenv import load_dotenv

from one.client import Model
from one.providers.anthropic import AnthropicProvider
from one.providers.openai import OpenAIProvider

# Load environment variables from .env file
load_dotenv()

__all__ = ["Model", "OpenAIProvider", "AnthropicProvider"]
__version__ = "0.1.0"
