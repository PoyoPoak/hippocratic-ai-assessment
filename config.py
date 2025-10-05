from __future__ import annotations


TEMPERATURES = {
    "improve": 0.5,
    "outline": 0.7,
    "draft": 0.9,
    "critique": 0.3,
    "revise": 0.5,
    "safety_audit": 0.0,     # Deterministic SAFE/UNSAFE classification
}

MAX_TOKENS = {
    "improve": 250,          # Short refined prompt
    "outline": 500,          # 5 bullets, multi-sentence
    "draft": 3000,           # Full story first draft
    "critique": 500,         # 3-5 critique bullets
    "revise": 3000,          # Revised story
    "safety_audit": 32,      # Single short line (SAFE or UNSAFE)
}

MIN_INPUT_CHARS = 5
MAX_INPUT_CHARS = 500

REVISION_ROUNDS = 2

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard previous",
    "you are now",
    "system prompt",
    "forget the rules"
]

BLOCKLIST = [
    "blood",  
    "knife",  
    "gun",    
    "kill",   
    "die",    
    "suicide",
    "drugs",
    "alcohol",
    "sexy",
]

PROMPTS: dict[str, str] = {
    "improve": (
        "You are a prompt engineer and search engine specialist. "
        "Refine users' prompts for generating children's stories (ages 5-10). "
        "Improve clarity, grammar, positivity, and age-appropriateness while preserving intent. "
        "Output ONLY the improved prompt text.\n"
        "USER_INPUT:<USER_INPUT>{candidate}</USER_INPUT>"
    ),
    "outline": (
        "You are a children's story planning assistant. "
        "Create a structured outline for a story for ages 5-10 in 5 bullet points: "
        "1) Setup 2) Rising Action 3) Problem 4) Resolution 5) Moral. "
        "Each bullet should be 2-3 concise sentences (NOT 5) to conserve tokens; vivid, gentle, imaginative. "
        "No violence, no scary horror elements, age-appropriate vocabulary. "
        "The story should have a positive message and lessons suitable for children in the plot. "
        "Provide only the 5 bullets.\n"
        "REFINED_IDEA:<REFINED_IDEA>{idea}</REFINED_IDEA>"
    ),
    "draft": (
        "You write children's stories for ages 5-10. "
        "Write a wholesome, engaging story from a provided outline. "
        "The story should be 500-1000 words, uses simple vivid language, short paragraphs (2-4 sentences each), gentle pacing, suited to children ages 5-10. "
        "Tone should be warm, curious, optimistic. "
        "Conclude naturally with an implicit moral (no 'The moral is' yet).\n"
        "OUTLINE:<OUTLINE>{outline}</OUTLINE>"
    ),
    "critique": (
        "You are an editorial assistant for children's stories ages 5-10. "
        "Provide 3-5 concise bullet points for areas of improvement for the story. "
        "Do not recite or rewrite the draft. Only provide critiques. "
        "Focus on clarity, engagement, pacing, emotional resonance, moral subtlety, age-appropriateness. "
        "Each bullet starts with a hyphen and a verb (e.g., '- Improve ...'). No extra text.\n"
        "DRAFT:<DRAFT>{draft}</DRAFT>"
    ),
    "revise": (
        "You are a professional writer revising a children's story. "
        "Produce a refined version addressing the critique. "
        "ONLY OUTPUT THE REVISED STORY. Do not include any other text or explanations. "
        "Preserve core plot & characters; keep it around 500-1000 words. "
        "Exclude any explicit 'Critique applied' notes, no analysis, just the revised story.\n"
        "ORIGINAL_DRAFT:<ORIGINAL_DRAFT>{draft}</ORIGINAL_DRAFT>\nCRITIQUE:<CRITIQUE>{critique}</CRITIQUE>"
    ),
    "safety_audit": (
        "You audit I/O text for an AI system designed to interact with children (ages 5-10). "
        "For any given story or user input, examine the text and respond ONLY with one token: SAFE or UNSAFE. "
        "UNSAFE IF: explicit violence, self-harm, strong fear, adult themes, sexual content, drugs, alcohol, weapons, hate, discrimination, graphic content, or similar.\n"
        "STORY:<STORY>{story}</STORY>"
    ),
}


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
    template = PROMPTS.get(stage)
    if not template:
        raise ValueError(f"Unknown prompt stage: {stage}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing = e.args[0]
        raise ValueError(f"Missing template variable '{missing}' for stage '{stage}'") from e