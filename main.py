import dotenv
import config
import scrubadub
from llm import LLM
from better_profanity import profanity
import unicodedata
import re
from dataclasses import dataclass

"""
Before submitting the assignment, describe here what you would build next with ~2 more hours. (Documented for reviewers.)
"""

dotenv.load_dotenv()
llm = LLM()


class InputHandler:
    """Prepares and validates raw user input before story generation."""

    def __init__(self, raw_input: str):
        self.raw_input = raw_input
        self.cleaned_input: str | None = None
        self.processed_input: str | None = None
        self.errors: list[str] = []
        self.min_length_chars = config.MIN_INPUT_CHARS
        self.max_length_chars = config.MAX_INPUT_CHARS

        if not self.raw_input or self.raw_input.strip() == "":
            self.errors.append("Input cannot be empty")
            return

        self.cleaned_input = self._clean(self.raw_input)

        # Validate input (populate self.errors if any)
        if not self._validate(self.cleaned_input):
            return

        # Prompt improvement
        try:
            self.processed_input = self._improve_prompt(self.cleaned_input)
        except Exception as e:
            self.processed_input = self.cleaned_input
            self.errors.append(f"Prompt improvement failed: {e}")

    def _clean(self, text: str) -> str:
        """Robust normalization: Unicode fold, remove controls, collapse whitespace & punct, trim."""
        s = unicodedata.normalize("NFKC", text)
        
        # Remove control chars except common whitespace
        s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
        
        # Replace tabs/newlines/multiple spaces with single space
        s = re.sub(r"\s+", " ", s)
        
        # Remove zero-width / directionality marks
        s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", s)
        
        # Collapse excessive punctuation runs (!!!???) -> ! / ? / .
        s = re.sub(r"([!?.]){2,}", r"\1", s)
        
        # Trim leading/trailing whitespace
        s = s.strip()
        
        return s

    def _validate(self, prompt: str) -> bool:
        """Validate prompt length, semantics (gate), moderation, and injection patterns."""
        # Basic length checks
        if len(prompt) < self.min_length_chars:
            self.errors.append("Input too short to infer story context.")
        if len(prompt) > self.max_length_chars:
            self.errors.append(f"Input exceeds maximum length of {self.max_length_chars} characters (got {len(prompt)}).")
        
        # Early exit on length errors, avoid unnecessary LLM calls            
        if self.errors:
            return False

        # Safety pass checking (PII scrub, blocklist, moderation, profanity, audit)
        safety_result = safety_pass(prompt, run_audit=True)
        if safety_result == "UNSAFE CONTENT BLOCKED":
            self.errors.append("Input failed safety checks.")
            return False
        
        # Optionally adopt scrubbed form produced by safety_pass
        prompt = safety_result
        self.cleaned_input = prompt

        # Injection pattern scan
        lowered = prompt.lower()
        for pat in config.INJECTION_PATTERNS:
            if pat in lowered:
                self.errors.append("Potential prompt injection pattern detected; refusing input.")
                return False
            
        return True

    def _improve_prompt(self, prompt: str) -> str:
        """Prompt LLM with prompt to improve clarity of request while retaining original intent."""
        improve_prompt = config.build_prompt("improve", candidate=prompt)
        return llm.generate(
            improve_prompt,
            max_tokens=config.MAX_TOKENS["improve"],
            temperature=config.TEMPERATURES["improve"],
            stem=True
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

    def create_story(self, user_input: str):
        """Run end-to-end generation from raw user input.

        Returns StoryResult-like dict to avoid direct printing side-effects.
        """
        handler = InputHandler(user_input)
        if not handler.processed_input:
            return {"success": False, "story": None, "errors": handler.errors, "unsafe": False}

        processed_input = handler.processed_input
        try:
            outline = self._run_stage("outline", idea=processed_input)
            draft = self._run_stage("draft", outline=outline)
            critique = self._run_stage("critique", draft=draft)
            revised = self._run_stage("revise", draft=draft, critique=critique)
            final_text = safety_pass(revised)
            if final_text == "UNSAFE CONTENT BLOCKED":
                return {"success": False, "story": None, "errors": ["Content blocked by safety filter"], "unsafe": True}
            return {"success": True, "story": final_text, "errors": [], "unsafe": False}
        except Exception as e:
            return {"success": False, "story": None, "errors": [f"Story generation failed"], "unsafe": False}


def safety_pass(text: str, run_audit: bool = True) -> str:
    """Comprehensive safety filter; preserves original casing except for PII replacements."""
    # Scrub PII but keep casing of remaining content
    scrubbed = scrubadub.clean(text)
    check = scrubbed.lower()

    # Moderation (case-insensitive outcome)
    try:
        llm.moderate(check)
    except ValueError as e:
        if "rejected" in str(e).lower():
            return "UNSAFE CONTENT BLOCKED"
    except Exception: # Fail open and move to other checks
        pass

    # Custom Blocklist (substring, case-insensitive)
    if any(word in check for word in config.BLOCKLIST):
        return "UNSAFE CONTENT BLOCKED"

    # Profanity check
    if profanity.contains_profanity(check):
        return "UNSAFE CONTENT BLOCKED"

    # Optional LLM audit (send scrubbed but not lowercased to preserve semantics)
    if run_audit:
        try:
            audit_prompt = config.build_prompt("safety_audit", story=scrubbed)
            audit_raw = llm.generate(
                audit_prompt,
                max_tokens=config.MAX_TOKENS["safety_audit"],
                temperature=config.TEMPERATURES["safety_audit"],
                # stem=True # Could break semantics
            ).strip().upper()
            if audit_raw.startswith("UNSAFE"):
                return "UNSAFE CONTENT BLOCKED"
        except Exception: # Fail open and move to other checks
            pass

    return scrubbed


def main():
    user_input = input("What kind of story do you want to hear?\n")
    result = StoryGenerator().create_story(user_input)
    
    if not result["success"]:
        print("Sorry, something went wrong and we couldn't create your story.")
    else:
        print(result["story"])


if __name__ == "__main__":
    main()