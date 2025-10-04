import os
import random
import openai
import dotenv
from typing import Optional
import config


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
            response = call_model(
                gate_prompt,
                    temperature=config.TEMPERATURES["gate"],
                    max_tokens=config.MAX_TOKENS["gate"],
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
        return call_model(
            tmpl,
            temperature=config.TEMPERATURES["improve"],
            max_tokens=config.MAX_TOKENS["improve"],
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
        
        return call_model(
            prompt,
            temperature=config.TEMPERATURES[stage],
            max_tokens=config.MAX_TOKENS[stage],
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
            safe_story = final_safety_pass(revised) # TODO convert final safety pass into a generic safety function
            return safe_story
        except Exception as e:
            return f"Story generation failed: {e}"


def final_safety_pass(story: str) -> str:
    """Perform a final safety audit of the story.

    Process:
        1. Blocklist substring scan (case-insensitive).
        2. LLM safety audit (`safety_audit`) returning one token: SAFE or UNSAFE.
        3. If any unsafe signal: return a generic unsafe message (no rewrite, no reasons).

    Args:
        story: Revised story text.

    Returns:
        Original story if safe; otherwise a short unsafe notice.
    """
    lowered = story.lower()
    
    # Blocklist scan
    block_hit = any(word in lowered for word in config.FINAL_OUTPUT_BLOCKLIST)

    # LLM audit
    if not block_hit:
        try:
            audit_prompt = build_prompt("safety_audit", story=story)
            audit_raw = call_model(
                audit_prompt,
                temperature=config.TEMPERATURES["safety_audit"],
                max_tokens=config.MAX_TOKENS["safety_audit"],
            ).strip().upper()
            if audit_raw.startswith("UNSAFE"):
                block_hit = True
            elif audit_raw.startswith("SAFE"):
                pass 
            else:
                # Ambiguous audit; allow (fail open) since no blocklist hit
                return story
        except Exception:
            return story if not block_hit else "UNSAFE STORY BLOCKED"

    if block_hit:
        return "UNSAFE STORY BLOCKED"
    return story


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


def call_model(prompt: str, max_tokens: Optional[int] = 3000, temperature: Optional[float] = 0.7) -> str:
    """Invoke the OpenAI chat completion API with retries and validation."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    
    llm = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    max_retries = 3
    backoff_base = 1

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm.chat.completions.create(
                model="gpt-3.5-turbo",  # DO NOT CHANGE THE MODEL
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Defensive checks
            if not resp or not getattr(resp, "choices", None):
                raise ValueError("Empty or malformed response from model")
            choice = resp.choices[0]
            content = getattr(getattr(choice, "message", None), "content", None)
            if not content:
                raise ValueError("No content field in model response")
            return content  # type: ignore

        except openai.RateLimitError as e:  # type: ignore[attr-defined]
            last_error = e
            if attempt == max_retries:
                raise
        except openai.APIConnectionError as e:  # type: ignore[attr-defined]
            last_error = e
            if attempt == max_retries:
                raise
        except openai.APITimeoutError as e:  # type: ignore[attr-defined]
            last_error = e
            if attempt == max_retries:
                raise
        except openai.AuthenticationError as e:  # type: ignore[attr-defined]
            raise RuntimeError(
                "Authentication failed (401). Check API key, organization membership, or regenerate key."
            ) from e
        except openai.PermissionDeniedError as e:  # type: ignore[attr-defined]
            raise RuntimeError(
                "Permission denied (403). Possible unsupported region or insufficient privileges."
            ) from e
        except openai.BadRequestError as e:  # type: ignore[attr-defined]
            raise RuntimeError(
                f"Bad request was made to OpenAI API. Verify prompt size/format. Details: {e}"
            ) from e
        except openai.InternalServerError as e:  # type: ignore[attr-defined]
            last_error = e
            if attempt == max_retries:
                raise
        except Exception as e:  # Catch-all for unexpected issues
            last_error = e
            if attempt == max_retries:
                raise

        # Retry for transient errors that may clear up soon, jittered backoff
        if attempt < max_retries:
            import time
            sleep_time = backoff_base * attempt
            sleep_time *= (0.9 + random.random() * 0.2) 
            time.sleep(sleep_time)

    # Should not reach here; loop either returns or raises.
    if last_error:
        raise last_error
    raise RuntimeError("Unknown error calling model")


def main():
    user_input = input("What kind of story do you want to hear?\n")
    story = StoryGenerator().create_story(user_input)
    print(story)


if __name__ == "__main__":
    main()