"""Resolve intervention positions against the actual chat-template tokenization."""

from enum import StrEnum
from weakref import WeakKeyDictionary

STATE_MARKER = "\nState recorded."
_POSITION_CACHES = WeakKeyDictionary()


class ReftPosition(StrEnum):
    ANSWER = "answer"
    STATE = "state"


def _position(lm, prompt, position):
    # A global strong model key would keep its GPU tensors resident after unload.
    cache = _POSITION_CACHES.setdefault(lm, {})
    key = (prompt, position)
    if key not in cache:
        if len(cache) >= 4096:
            cache.clear()
        cache[key] = _resolve_position(lm, prompt, position)
    return cache[key]


def _resolve_position(lm, prompt, position):
    ids = lm._prompt_ids(prompt)
    if position == ReftPosition.ANSWER:
        return len(ids) - 1
    if prompt.count(STATE_MARKER) != 1:
        raise ValueError("State intervention needs exactly one state boundary marker")
    chat = bool(getattr(lm.tokenizer, "chat_template", None))
    rendered = (
        lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        if chat
        else prompt
    )
    if rendered.count(prompt) != 1:
        raise ValueError("Chat template must preserve the user prompt exactly once")
    boundary = rendered.index(prompt) + prompt.index(STATE_MARKER) + len(STATE_MARKER)
    if not getattr(lm.tokenizer, "is_fast", False):
        raise ValueError("State intervention requires a fast tokenizer with character offsets")
    encoded = lm.tokenizer(rendered, add_special_tokens=not chat, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("State offsets do not match the actual chat tokenization")
    # A token crossing the marker/question boundary must never receive the edit.
    safe = [
        i
        for i, (start, end) in enumerate(encoded["offset_mapping"])
        if 0 <= start < end <= boundary
    ]
    if not safe or safe[-1] >= len(ids) - 1:
        raise ValueError("No safe state token before the question")
    return safe[-1]


def intervention_positions(lm, prompts, position=ReftPosition.ANSWER):
    position = ReftPosition(position)
    return [_position(lm, prompt, position) for prompt in prompts]


def validate_state_prefixes(lm, samples):
    """Fail if any role leaks into the token prefix of a shared world/style."""
    groups, rows = {}, []
    for q in samples:
        prompt = q.prompt()
        pos = _position(lm, prompt, ReftPosition.STATE)
        prefix = lm._prompt_ids(prompt)[: pos + 1]
        previous = groups.setdefault(q.state_key, prefix)
        if previous != prefix:
            raise ValueError(f"State prefix depends on the question: {q.key}")
        rows.append(
            {"key": q.key, "token_position": pos, "prompt_tokens": len(lm._prompt_ids(prompt))}
        )
    return {
        "states": len(groups),
        "questions": len(samples),
        "identical_prefix_per_state": True,
        "positions": rows,
    }
