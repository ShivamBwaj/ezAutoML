"""
Groq API client with JSON enforcement and retry logic.
"""
import json
import time
from typing import Type, TypeVar, Any, Dict, List, Optional
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
        api_keys: Optional[List[str]] = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
        max_retries: int = 3
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key
            model: Model identifier (default: llama-3.1-8b-instant)
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Number of retry attempts on failure
        """
        self.api_keys = api_keys[:] if api_keys else [api_key]
        self._active_key_index = 0
        self.client = Groq(api_key=self.api_keys[self._active_key_index])
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.tokens_per_minute_limit = 12000
        self._window_start_by_key: Dict[int, float] = {}
        self._used_tokens_by_key: Dict[int, int] = {}

    def _estimate_tokens(self, text: str) -> int:
        # Conservative heuristic: ~4 chars/token
        return max(1, len(text) // 4)

    def _ensure_rate_budget(self, prompt: str, expected_response_tokens: int = 800) -> None:
        """
        Enforce per-key token/minute budget and rotate key if needed.
        """
        needed = self._estimate_tokens(prompt) + expected_response_tokens
        now = time.time()
        key_idx = self._active_key_index
        start = self._window_start_by_key.get(key_idx, now)
        used = self._used_tokens_by_key.get(key_idx, 0)

        if now - start >= 60:
            start = now
            used = 0

        if used + needed > self.tokens_per_minute_limit:
            # Try rotating to next key first
            if self._rotate_key():
                self._ensure_rate_budget(prompt, expected_response_tokens)
                return
            # No key left: wait until current key budget window resets
            wait_s = max(0.0, 60 - (now - start))
            if wait_s > 0:
                time.sleep(wait_s)
            start = time.time()
            used = 0

        self._window_start_by_key[self._active_key_index] = start
        self._used_tokens_by_key[self._active_key_index] = used + needed

    def _is_rate_limited_error(self, error: Exception) -> bool:
        """Detect retryable rate-limit/quota style errors."""
        msg = str(error).lower()
        return any(token in msg for token in ["rate limit", "429", "quota", "too many requests"])

    def _rotate_key(self) -> bool:
        """Rotate to next API key if available."""
        if len(self.api_keys) <= 1:
            return False
        next_index = self._active_key_index + 1
        if next_index >= len(self.api_keys):
            return False
        self._active_key_index = next_index
        self.client = Groq(api_key=self.api_keys[self._active_key_index])
        return True

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
                self._ensure_rate_budget(prompt)
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
                if self._is_rate_limited_error(e) and self._rotate_key():
                    # Immediate retry on next key to minimize pipeline interruption
                    continue
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
                self._ensure_rate_budget(prompt, expected_response_tokens=1000)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if self._is_rate_limited_error(e) and self._rotate_key():
                    continue
                if attempt == self.max_retries - 1:
                    raise LLMError(f"Groq API failed: {str(e)}") from e
                time.sleep(2 ** attempt)

        raise LLMError("Max retries exceeded")
