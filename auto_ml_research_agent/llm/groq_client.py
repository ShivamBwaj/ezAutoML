"""
Groq API client with JSON enforcement and retry logic.
"""
import json
import time
from typing import Type, TypeVar, Any, Dict
from pydantic import BaseModel, ValidationError
from groq import Groq
from groq.types.chat.chat_completion import ChatCompletion

from auto_ml_research_agent.exceptions import LLMError

T = TypeVar('T', bound=BaseModel)


class GroqClient:
    """Wrapper for Groq API with structured JSON output and retries"""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_retries: int = 3
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key
            model: Model identifier (default: llama-3.3-70b-versatile)
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Number of retry attempts on failure
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def generate_json(self, prompt: str, response_model: Type[T]) -> T:
        """
        Generate JSON response validated against Pydantic model.

        Args:
            prompt: User prompt (system message added automatically)
            response_model: Pydantic model for response validation

        Returns:
            Parsed and validated response model instance

        Raises:
            LLMError: If max retries exceeded or invalid JSON
        """
        system_message = (
            "You are an AI assistant that always responds with valid JSON. "
            "Do not include any text outside the JSON structure. "
            "Ensure the JSON matches the requested schema exactly."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}  # Enforce JSON mode
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from LLM")

                # Parse JSON
                data = json.loads(content)

                # Validate with Pydantic model
                return response_model(**data)

            except json.JSONDecodeError as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Failed to parse LLM response as JSON after {self.max_retries} attempts. "
                        f"Last error: {str(e)}"
                    ) from e
                # Exponential backoff
                time.sleep(2 ** attempt)

            except ValidationError as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"LLM response failed schema validation after {self.max_retries} attempts. "
                        f"Validation errors: {str(e)}"
                    ) from e
                time.sleep(2 ** attempt)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(f"Groq API failed: {str(e)}") from e
                time.sleep(2 ** attempt)

        raise LLMError("Max retries exceeded")

    def generate_text(self, prompt: str) -> str:
        """
        Generate plain text response (non-JSON).

        Args:
            prompt: User prompt

        Returns:
            Raw text response
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMError(f"Groq API failed: {str(e)}") from e
                time.sleep(2 ** attempt)

        raise LLMError("Max retries exceeded")
