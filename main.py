import dotenv
import config
import scrubadub
from llm import LLM


"""
Before submitting the assignment, describe here what you would build next with ~2 more hours. (Documented for reviewers.)
"""

dotenv.load_dotenv()

# Global LLM client instance
llm = LLM()


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
            return False

        # LLM semantic validation (gate) - used to handle nuanced cases with LLM as a loose reasoning filter
        gate_prompt = config.build_prompt("gate", user_request=prompt)
        try:
            response = llm.generate(
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
        tmpl = config.build_prompt("improve", candidate=prompt)
        return llm.generate(
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
        prompt = config.build_prompt(stage, **kwargs)
        return llm.generate(
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
            return safety_pass(revised)
        except Exception as e:
            return f"Story generation failed: {e}"

def safety_pass(text: str, run_audit: bool = True) -> str:
    """Comprehensive safety filter for arbitrary text.""" 
    # PII Scrub
    text = scrubadub.clean(text)
    text = text.lower()
    
    # Moderation check (may raise)
    try:
        llm.moderate(text)
    except ValueError as e:
        if "rejected" in str(e).lower():
            return "UNSAFE CONTENT BLOCKED"
    except Exception: # Fail open and move to other checks
        pass

    # Blocklist substring scan
    if any(word in text for word in config.FINAL_OUTPUT_BLOCKLIST):
        return "UNSAFE CONTENT BLOCKED"

    # Optional LLM audit
    if run_audit:
        try:
            audit_prompt = config.build_prompt("safety_audit", story=text)
            audit_raw = llm.generate(
                audit_prompt,
                max_tokens=config.MAX_TOKENS["safety_audit"],
                temperature=config.TEMPERATURES["safety_audit"],
            ).strip().upper()
            if audit_raw.startswith("UNSAFE"):
                return "UNSAFE CONTENT BLOCKED"
        except Exception: # Fail open and move to other checks
            pass

    return text




def main():
    user_input = input("What kind of story do you want to hear?\n")
    story = StoryGenerator().create_story(user_input)
    print(story)


if __name__ == "__main__":
    main()