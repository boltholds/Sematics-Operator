"""Shared bare-digit protocol for the causal experiments and free generation."""

BARE_CANDIDATES = ("0", "1")
ANSWER_INSTRUCTION = (
    "Reply with exactly one digit: 0 or 1. Do not add words, punctuation, or an explanation."
)


def protocol_metadata(lm):
    token_ids = [lm.tokenizer.encode(c, add_special_tokens=False) for c in BARE_CANDIDATES]
    if not all(token_ids) or token_ids[0] == token_ids[1]:
        raise ValueError("Bare digit candidates must be nonempty and distinguishable")
    return {
        "name": "bare_digit_v1",
        "candidates": list(BARE_CANDIDATES),
        "candidate_token_ids": token_ids,
        "instruction": ANSWER_INSTRUCTION,
        "boundary": "last_prompt_token",
        "forced_prefix": [],
        "generation_policy": "Reapply at the original last prompt position on every uncached forward; never move to generated tokens.",
    }


def parse_generation(generated):
    text = generated["text"].strip()
    incomplete = generated["stop_reason"] == "max_new_tokens"
    binary = text in BARE_CANDIDATES and not incomplete
    return {
        **generated,
        "prediction": int(text) if binary else None,
        "generation_status": "incomplete" if incomplete else "binary" if binary else "format_error",
    }
