import os
import random
from typing import Optional
import openai
from nltk.stem.porter import PorterStemmer


class LLM:
    """LLM client wrapper providing moderation + chat completion with retries."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        self._client = openai.OpenAI(api_key=key)
        self._max_retries = 3
        self._backoff_base = 1

    def _sleep(self, attempt: int) -> None:
        """Sleep with exponential backoff and jitter."""
        import time
        sleep_time = self._backoff_base * attempt
        sleep_time *= (0.9 + random.random() * 0.2)
        time.sleep(sleep_time)

    def _handle_errors(self, attempt: int, e: Exception) -> bool:
        """Determine if we should retry based on error type and attempt count.

        Args:
            attempt (int): Current attempt number.
            e (Exception): The exception encountered during the attempt.
            
        Raises:
            RuntimeError: Authentication failed (401). Check API key.
            RuntimeError: Permission denied (403). Check account/region permissions.
            RuntimeError: Bad request to OpenAI API.
        Returns:
            bool: True if the operation should be retried, False otherwise.
        """        
        retryable = (
            openai.RateLimitError,  # type: ignore[attr-defined]
            openai.APIConnectionError,  # type: ignore[attr-defined]
            openai.APITimeoutError,  # type: ignore[attr-defined]
            openai.InternalServerError,  # type: ignore[attr-defined]
        )
        
        # Non-retryable errors
        if isinstance(e, openai.AuthenticationError):  # type: ignore[attr-defined]
            raise RuntimeError("Authentication failed (401). Check API key.") from e
        if isinstance(e, openai.PermissionDeniedError):  # type: ignore[attr-defined]
            raise RuntimeError("Permission denied (403). Check account/region permissions.") from e
        if isinstance(e, openai.BadRequestError):  # type: ignore[attr-defined]
            raise RuntimeError(f"Bad request to OpenAI API: {e}") from e
        if isinstance(e, retryable):
            return attempt < self._max_retries
        
        # Retry if it's a known retryable error and we have attempts left
        return attempt < self._max_retries

    def _stem_words(self, text: str) -> str:
        """Stem words in the input text using Porter Stemmer to reduce token count."""
        stemmer = PorterStemmer()
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def _moderate(self, text: str) -> None:
        """Run moderation check; raise if flagged or on error.

        Args:
            text (str): Text to be moderated.

        Raises:
            ValueError: If the text is flagged by moderation.
            ValueError: If the moderation response is empty or malformed.
            last_error: The last exception encountered during retries.
            RuntimeError: If an unknown error occurs during moderation.
        """
        last_error: Exception | None = None
        
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.moderations.create(
                    model="omni-moderation-latest",
                    input=text,
                )
                
                results = getattr(resp, "results", [])
                
                if not results:
                    raise ValueError("Empty moderation response")
                
                flagged = any(getattr(r, "flagged", False) for r in results)
                if flagged:
                    raise ValueError("Input rejected by moderation")
                return
            
            except Exception as e:
                last_error = e
                if not self._handle_errors(attempt, e):
                    raise
                self._sleep(attempt)
        
        # If we exhausted retries, raise the last error encountered
        if last_error:
            raise last_error
        
        raise RuntimeError("Unknown moderation error")

    def moderate(self, text: str):
        """Public moderation method"""
        return self._moderate(text)

    def generate(self, prompt: str, *, max_tokens: int = 3000, temperature: float = 0.7, stem: bool = False) -> str:
        """Generate text completion.

        Args:
            prompt (str): Text prompt to generate completion for.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 3000.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.

        Raises:
            ValueError: If the model response is empty or malformed.
            ValueError: If no content is found in the model response.
            last_error: The last exception encountered during retries.
            RuntimeError: If an unknown error occurs during generation.

        Returns:
            str: Generated text completion.
        """
        last_error: Exception | None = None
        
        if stem:
            prompt = self._stem_words(prompt)
        
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model="gpt-3.5-turbo",  # DO NOT CHANGE THE MODEL
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                if not resp or not getattr(resp, "choices", None):
                    raise ValueError("Empty or malformed response from model")
                
                choice = resp.choices[0]
                content = getattr(getattr(choice, "message", None), "content", None)
                
                if not content:
                    raise ValueError("No content field in model response")
                
                return content  # type: ignore
            
            except Exception as e:
                last_error = e
                if not self._handle_errors(attempt, e):
                    raise
                self._sleep(attempt)
 
        # If we exhausted retries, raise the last error encountered
        if last_error:
            raise last_error
        
        raise RuntimeError("Unknown generation error")


__all__ = ["LLM"]
