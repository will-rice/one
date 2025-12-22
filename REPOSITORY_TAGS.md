# Repository Tags

This document describes the recommended tags for this repository.

## Git Version Tags

This repository uses semantic versioning. The following git tag has been created:

- **v0.1.0** - Initial release with support for OpenAI and Anthropic providers, structured outputs via Pydantic

### How to Push Tags

To push the tag to GitHub (requires appropriate permissions):

```bash
git push origin v0.1.0
```

Or to push all tags:

```bash
git push origin --tags
```

## GitHub Repository Topics

The following topics are recommended for this GitHub repository to improve discoverability:

### Core Topics
- `python` - Primary programming language
- `llm` - Large Language Model library
- `ai` - Artificial Intelligence
- `openai` - OpenAI provider support
- `anthropic` - Anthropic/Claude provider support
- `pydantic` - Structured outputs using Pydantic
- `type-safe` - Type-safe implementation with type hints

### Feature Topics
- `structured-outputs` - First-class structured output support
- `multiple-providers` - Unified interface for multiple LLM providers
- `api-wrapper` - API wrapper/client library
- `python-library` - Python library/package
- `llm-client` - LLM client library

### Technical Topics
- `mypy` - Type checking with mypy
- `pytest` - Testing with pytest
- `ruff` - Linting and formatting with ruff

### How to Add Topics to GitHub

Repository topics can be added through the GitHub web interface:

1. Go to the repository page on GitHub
2. Click the gear icon (⚙️) next to "About" on the right sidebar
3. Add topics in the "Topics" field
4. Click "Save changes"

Alternatively, topics can be managed using the GitHub API or CLI:

```bash
# Using GitHub CLI
gh repo edit will-rice/one --add-topic python,llm,ai,openai,anthropic,pydantic,type-safe,structured-outputs,multiple-providers
```

## Rationale

These tags help with:
- **Discoverability**: Users searching for LLM libraries or Python AI tools can find this project
- **Categorization**: Clear indication of the project's purpose and features
- **SEO**: Better search engine optimization for relevant queries
- **Community**: Helps connect with users interested in similar technologies
