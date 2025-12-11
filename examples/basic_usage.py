"""Basic usage examples for the One library."""

from pydantic import BaseModel

from one import Model


class Person(BaseModel):
    """Example model for structured outputs."""

    name: str
    age: int
    occupation: str


def main() -> None:
    """Run basic examples."""
    print("One Library - Basic Usage Examples")
    print("=" * 50)

    # Example 1: Simple text generation with OpenAI
    print("\n1. Simple text generation with OpenAI:")
    print("-" * 50)
    model = Model(model="gpt-4o-mini")
    try:
        response = model.generate("What is the capital of France? Answer in one word.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error (expected if no API key): {e}")

    # Example 2: Structured output with OpenAI
    print("\n2. Structured output with OpenAI:")
    print("-" * 50)
    try:
        person = model.generate_structured(
            prompt="Extract person info: John is 30 years old and works as a software engineer",
            response_format=Person,
        )
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
    except Exception as e:
        print(f"Error (expected if no API key): {e}")

    # Example 3: Using Anthropic
    print("\n3. Simple text generation with Anthropic:")
    print("-" * 50)
    model_anthropic = Model(model="claude-3-5-sonnet-20241022")
    try:
        response = model_anthropic.generate(
            "What is the capital of Japan? Answer in one word."
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error (expected if no API key): {e}")

    # Example 4: Structured output with Anthropic
    print("\n4. Structured output with Anthropic:")
    print("-" * 50)
    try:
        person = model_anthropic.generate_structured(
            prompt="Extract person info: Jane is 25 years old and works as a data scientist",
            response_format=Person,
        )
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
    except Exception as e:
        print(f"Error (expected if no API key): {e}")

    print("\n" + "=" * 50)
    print("Examples complete!")
    print(
        "\nNote: To run these examples successfully, set OPENAI_API_KEY "
        "and/or ANTHROPIC_API_KEY environment variables."
    )


if __name__ == "__main__":
    main()
