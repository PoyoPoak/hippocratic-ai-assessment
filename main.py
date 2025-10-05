import os
import random
import openai
import dotenv
from typing import Optional
import config
import scrubadub


"""
Before submitting the assignment, describe here what you would build next with ~2 more hours. (Documented for reviewers.)
"""

dotenv.load_dotenv()


class InputHandler:
    """Prepares and validates raw user input before story generation."""

    def __init__(self, raw_input: str):
        self.raw_input = raw_input
        
        # Populate after processing
        self.cleaned_input: str | None = None
        self.processed_input: str | None = None
        
        self.errors: list[str] = []
        
        self.min_length_chars = config.MIN_INPUT_CHARS
        self.max_length_chars = config.MAX_INPUT_CHARS

        # Init processing
        if not self.raw_input or self.raw_input.strip() == "":
            self.errors.append("Input cannot be empty")
            return
        
        self.cleaned_input = self._clean(self.raw_input)
    
        if not self._validate(self.cleaned_input):
            return 
        
        try:
            self.processed_input = self._improve_prompt(self.cleaned_input)
        except Exception as e: 
            self.errors.append(f"Prompt improvement failed: {e}")
            self.processed_input = self.cleaned_input

    def _clean(self, text: str) -> str:
        """Normalize whitespace and strip surrounding spaces."""
        text = text.replace("\r", "")
        text = text.replace("\t", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def _validate(self, prompt: str) -> bool:
        """Validate prompt length, semantic suitability, and injection safety."""
        if len(prompt) < self.min_length_chars:
            self.errors.append("Input too short to infer story context.")
        if len(prompt) > self.max_length_chars:
            self.errors.append(f"Input exceeds maximum length of {self.max_length_chars} characters (got {len(prompt)}).")
        
        # If there are local errors, skip LLM validation to save resources
        if self.errors:
            print(f"Validation errors: {self.errors}")
            return False

        # LLM semantic validation (gate) - used to handle nuanced cases with LLM as a loose reasoning filter
        gate_prompt = build_prompt("gate", user_request=prompt)
        try:
            response = openai_client.generate(
                gate_prompt,
                max_tokens=config.MAX_TOKENS["gate"],
                temperature=config.TEMPERATURES["gate"],
            )
        except Exception as e:  
            self.errors.append(f"Validation service error: {e}")
            return False
        
        # In case of unexpected formatting or periods added to the end
        cleaned_resp = response.strip().upper()
        
        # Strict LLM gate response enforcement
        if cleaned_resp not in {"VALID", "INVALID"}:
            self.errors.append(f"Gate returned ambiguous response: {cleaned_resp!r}")
            return False
        elif cleaned_resp == "INVALID":
            self.errors.append("Input is not suitable for a children's story.")
            return False

        # Always-on injection pattern scan, not conclusive as these can come in many forms. Keeping simple for demonstrative purposes.
        lowered = prompt.lower()
        for pat in config.INJECTION_PATTERNS:
            if pat in lowered:
                self.errors.append("Potential prompt injection pattern detected; refusing input.")
                return False
        
        # Validation succeeded and no errors recorded
        return self.errors == []

    def _improve_prompt(self, prompt: str) -> str:
        """Prompt LLM with prompt to improve clarity of request while retaining original intent."""
        tmpl = build_prompt("improve", candidate=prompt)
        return openai_client.generate(
            tmpl,
            max_tokens=config.MAX_TOKENS["improve"],
            temperature=config.TEMPERATURES["improve"],
        )
    
    def get_prompt(self) -> str | None:
        """Return refined prompt or None if validation failed."""
        return self.processed_input


class StoryGenerator:
    """Orchestrates multi-stage story generation from user input."""
    
    def _run_stage(self, stage: str, **kwargs) -> str:
        """Generic executor for outline/draft/critique/revise stages.

        Args:
            stage: One of 'outline', 'draft', 'critique', 'revise'.
            **kwargs: Template variables required by that stage's prompt.

        Returns:
            Model completion text for the stage.
        """
        if stage not in ("outline", "draft", "critique", "revise"):
            raise ValueError(f"Unsupported stage: {stage}")

        # Build the prompt for the given stage and call the model
        prompt = build_prompt(stage, **kwargs)
        
        return openai_client.generate(
            prompt,
            max_tokens=config.MAX_TOKENS[stage],
            temperature=config.TEMPERATURES[stage],
        )

    def create_story(self, user_input: str) -> str:
        """Run end-to-end generation from raw user input."""
        handler = InputHandler(user_input)
        if not handler.processed_input:
            return "Error processing input: " + "; ".join(handler.errors)

        processed_input = handler.processed_input
        
        # Multi-stage generation with error handling
        try:
            outline = self._run_stage("outline", idea=processed_input)
            draft = self._run_stage("draft", outline=outline)
            critique = self._run_stage("critique", draft=draft)
            revised = self._run_stage("revise", draft=draft, critique=critique)
            # safe_story = safety_pass(revised) # TODO convert final safety pass into a generic safety function
            return revised
        except Exception as e:
            return f"Story generation failed: {e}"


# def safety_pass(text: str) -> str:
#     """Performs a safety check on a given piece of text"""
#     lowered = text.lower()
    
#     # Length check
#     # TODO Ensure input is at least config.MIN_INPUT_CHARS and at most config.MAX_INPUT_CHARS
    
#     # Scrub any personal identifiable information 
#     # TODO Integrate scrubadub or similar library to remove PII

#     # Moderation API check
#     # TODO Implement call to OpenAI moderation endpoint and block if flagged
    
#     # Blocklist scan
#     block_hit = any(word in lowered for word in config.FINAL_OUTPUT_BLOCKLIST)

#     # LLM audit
#     if not block_hit:
#         try:
#             audit_prompt = build_prompt("safety_audit", story=text)
#             audit_raw = openai_client.generate(
#                 audit_prompt,
#                 max_tokens=config.MAX_TOKENS["safety_audit"],
#                 temperature=config.TEMPERATURES["safety_audit"],
#             ).strip().upper()
#             if audit_raw.startswith("UNSAFE"):
#                 block_hit = True
#             elif audit_raw.startswith("SAFE"):
#                 pass 
#             else:
#                 # Ambiguous audit; allow (fail open) since no blocklist hit
#                 return text
#         except Exception:
#             return text if not block_hit else "UNSAFE STORY BLOCKED"

#     if block_hit:
#         return "UNSAFE STORY BLOCKED"
#     return text


def build_prompt(stage: str, **kwargs) -> str:
    """Render a prompt template for the given stage with provided variables.

    Args:
        stage: Template key (must exist in PROMPTS).
        **kwargs: Named variables referenced by the template's placeholders.

    Returns:
        Interpolated prompt string ready for model submission.

    Raises:
        ValueError: If the stage is unknown or a required placeholder variable is missing.
    """
    template = config.PROMPTS.get(stage)
    if not template:
        raise ValueError(f"Unknown prompt stage: {stage}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(f"Missing template variable '{missing}' for stage '{stage}'") from e


class OpenAI:
    """Client class wrapper handling API usage with unified retry logic."""

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
        """
        Handles errors from OpenAI API calls and determines if a retry should be attempted.

        Args:
            attempt (int): Current attempt number (1-based).
            e (Exception): The exception encountered during the API call.

        Raises:
            RuntimeError: Authentication failed (401).
            RuntimeError: Permission denied (403).
            RuntimeError: Bad request to OpenAI API.

        Returns:
            bool: True if the operation should be submitted again, False otherwise.
        """
        # Returns True if should retry, otherwise raises.
        retryable = (
            openai.RateLimitError,  # type: ignore[attr-defined]
            openai.APIConnectionError,  # type: ignore[attr-defined]
            openai.APITimeoutError,  # type: ignore[attr-defined]
            openai.InternalServerError,  # type: ignore[attr-defined]
        )
        
        # Handle known non-retryable errors
        if isinstance(e, openai.AuthenticationError):  # type: ignore[attr-defined]
            raise RuntimeError("Authentication failed (401). Check API key.") from e
        if isinstance(e, openai.PermissionDeniedError):  # type: ignore[attr-defined]
            raise RuntimeError("Permission denied (403). Check account/region permissions.") from e
        if isinstance(e, openai.BadRequestError):  # type: ignore[attr-defined]
            raise RuntimeError(f"Bad request to OpenAI API: {e}") from e
        if isinstance(e, retryable):
            return attempt < self._max_retries
        
        # Unknown error: retry only if not final attempt
        return attempt < self._max_retries

    def _moderate(self, text: str) -> bool:
        """Moderate input text using OpenAI's moderation endpoint."""
        # Track last error for final raise
        last_error: Exception | None = None
        
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.moderations.create(
                    model="omni-moderation-latest",
                    input=text,
                )
                
                # Expect resp.results list with flagged bools 
                results = getattr(resp, "results", [])
                if not results:
                    raise ValueError("Empty moderation response")

                # Check if any of the results are flagged, if so reject input
                flagged = any(getattr(r, "flagged", False) for r in results)
                if flagged:
                    raise ValueError("Input rejected by moderation")
                return
            
            # Handle and possibly retry on errors
            except Exception as e:  
                last_error = e
                should_retry = self._handle_errors(attempt, e)
                if not should_retry:
                    raise
                self._sleep(attempt)
                
        if last_error:
            raise last_error
        
        raise RuntimeError("Unknown moderation error")

    def generate(self, prompt: str, *, max_tokens: int = 3000, temperature: float = 0.7) -> str:
        """Generate a response from the model."""
        
        # Track last error for final raise
        last_error: Exception | None = None 
        
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
            
            # Handle and possibly retry on errors
            except Exception as e:
                last_error = e
                should_retry = self._handle_errors(attempt, e)
                if not should_retry:
                    raise
                self._sleep(attempt)

        # Raise last error if all attempts failed
        if last_error:
            raise last_error
        
        raise RuntimeError("Unknown generation error")


# Singleton client instance used by helper functions/classes above
openai_client = OpenAI()


def main():
    user_input = input("What kind of story do you want to hear?\n")
    story = StoryGenerator().create_story(user_input)
    print(story)


if __name__ == "__main__":
    main()