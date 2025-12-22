# One

A clean, minimalist library for using multiple LLM providers with structured outputs.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/one-llm-lib/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Features

- **Multiple Providers**: Support for OpenAI and Anthropic with a unified interface
- **Structured Outputs**: First-class support for structured outputs using Pydantic models
- **Type Safe**: Full type hints and runtime validation with Pydantic
- **Simple API**: Intuitive interface with sensible defaults
- **Extensible**: Easy to add new providers
- **Well Tested**: Comprehensive test suite with >95% coverage

## Installation

```bash
pip install one-llm-lib
```

Or with development dependencies:

```bash
pip install one-llm-lib[dev]
```

## Quick Start

### Basic Text Generation

```python
from one import Model

# Using OpenAI (provider auto-detected from model name)
model = Model(model="gpt-4o-mini")
response = model.generate("What is the capital of France?")
print(response)  # "Paris"

# Using Anthropic (provider auto-detected from model name)
model = Model(model="claude-3-5-sonnet-20241022")
response = model.generate("What is the capital of France?")
print(response)  # "Paris"
```

### Structured Outputs

```python
from one import Model
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

model = Model(model="gpt-4o-mini")
person = model.generate(
    prompt="Extract information: John is 30 years old and works as a software engineer",
    response_format=Person
)

print(person.name)  # "John"
print(person.age)   # 30
print(person.occupation)  # "software engineer"
```

## Setup

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Or create a `.env` file (automatically loaded by the library):

```bash
cp .env.example .env
# Edit .env and add your API keys
```

The library uses `python-dotenv` to automatically load environment variables from a `.env` file in your project root.

### Advanced Usage

#### Custom Parameters

```python
# Use a specific OpenAI model with custom parameters
model = Model(model="gpt-4")
response = model.generate(
    prompt="Explain quantum computing",
    temperature=0.5,
    max_tokens=500
)

# Use a specific Anthropic model with custom parameters
model = Model(model="claude-3-opus-20240229")
response = model.generate(
    prompt="Explain quantum computing",
    temperature=0.5,
    max_tokens=500
)
```

#### Complex Structured Outputs

```python
from pydantic import BaseModel
from typing import List

class Task(BaseModel):
    title: str
    priority: str
    estimated_hours: float

class Project(BaseModel):
    name: str
    description: str
    tasks: List[Task]

model = Model(model="gpt-4o-mini")
project = model.generate(
    prompt="""
    Create a project plan for building a web application:
    - User authentication
    - Database design
    - API development
    - Frontend implementation
    """,
    response_format=Project
)

for task in project.tasks:
    print(f"{task.title}: {task.estimated_hours}h ({task.priority})")
```

## Architecture

### Adding New Providers

The library is designed to be easily extensible. To add a new provider:

1. Create a new file in `src/one/providers/` (e.g., `src/one/providers/cohere.py`)
2. Inherit from the `Provider` abstract base class:

```python
from one.providers.base import Provider
from pydantic import BaseModel
from typing import Type, Any, Union

class CohereProvider(Provider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model, api_key)
        # Initialize your provider client (self.model is available)
        pass

    def generate(
        self,
        prompt: str,
        response_format: Type[BaseModel] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str | BaseModel:
        # If response_format is None, return plain text
        # If response_format is provided, return structured model
        # Implementation using self.model
        pass
```

3. Update the `_detect_provider` function in `src/one/client.py` to recognize your provider's model names
4. Add the provider initialization logic to `Model.__init__`

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Type Checking

```bash
mypy src/
```

### Linting and Formatting

```bash
ruff check src/
ruff format src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run -a
```

## Dependencies

Core dependencies:

- **openai**: OpenAI API client
- **anthropic**: Anthropic API client
- **pydantic**: Data validation and type safety
- **python-dotenv**: Load environment variables from .env files

Development tools:

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checker
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

## Why "One"?

One library to rule them all. One interface for multiple LLM providers.

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and code quality checks succeed
6. Submit a pull request

## Supported Providers

| Provider  | Status       | Structured Outputs | Default Model              |
| --------- | ------------ | ------------------ | -------------------------- |
| OpenAI    | ✅ Supported | ✅ Native          | gpt-4o-mini                |
| Anthropic | ✅ Supported | ✅ JSON Schema     | claude-3-5-sonnet-20241022 |

## Roadmap

- [ ] Add support for streaming responses
- [ ] Add support for async/await
- [ ] Add more providers (Google, Cohere, etc.)
- [ ] Add token counting utilities
- [ ] Add retry logic and rate limiting
- [ ] Add cost estimation
