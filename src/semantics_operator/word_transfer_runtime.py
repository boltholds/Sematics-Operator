"""Raw-token inference with one fixed, question-independent block intervention."""

from contextlib import contextmanager

import torch

from .localization import hidden


@contextmanager
def edit_at(lm, site, position, vector):
    """Reapply at the original prefix position on every cache-free forward."""
    if position < 0:
        raise ValueError("Intervention position must be nonnegative")
    if vector is None:
        yield
        return
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("Intervention requires a finite one-dimensional vector")

    def hook(module, args, output):
        tensor = hidden(output)
        if position >= tensor.shape[1] or vector.numel() != tensor.shape[-1]:
            raise ValueError("Intervention position or hidden width does not match the block")
        result = tensor.clone()
        result[:, position] += vector.to(result)
        if isinstance(output, tuple):
            return (result, *output[1:])
        if isinstance(output, list):
            return [result, *output[1:]]
        return result

    handle = lm.model.get_submodule(site).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _logits(lm, sequence):
    ids, mask = lm._batch([sequence])
    logits = lm.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite word-transfer logits")
    return logits


@torch.inference_mode()
def measure(
    lm,
    prefix,
    site,
    position,
    vector,
    *,
    candidates=(),
    max_new_tokens=16,
    baseline_logp=None,
):
    """Unforced greedy decoding plus summed, separately tokenized candidate likelihoods.

    No chat wrapper, forced space or moving intervention position. Candidate
    likelihoods exclude EOS; free generation must reach EOS to count as complete.
    """
    if not prefix or not 0 <= position < len(prefix):
        raise ValueError("Anchor must be inside the original prefix")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    suffixes = [lm.tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    if candidates and (
        any(not s for s in suffixes) or len({tuple(s) for s in suffixes}) != len(suffixes)
    ):
        raise ValueError("Candidate token sequences must be nonempty and distinguishable")
    budget = max([max_new_tokens, *map(len, suffixes)])
    if len(prefix) + budget > lm.max_length:
        raise ValueError("Prompt plus continuation exceeds max_length; increase it explicitly")
    with edit_at(lm, site, position, vector):
        first = _logits(lm, prefix)[-1].log_softmax(-1)
        scores = []
        for suffix in suffixes:
            score = float(first[suffix[0]])
            if len(suffix) > 1:
                logits = _logits(lm, prefix + suffix[:-1]).log_softmax(-1)
                score += sum(
                    float(logits[len(prefix) + j - 1, t]) for j, t in enumerate(suffix[1:], 1)
                )
            scores.append(score)
        generated, reason, logits = [], "max_new_tokens", first
        stop_ids = set(lm.eos_token_ids())
        for step in range(max_new_tokens):
            token = int(logits.argmax())
            generated.append(token)
            if token in stop_ids:
                reason = "eos"
                break
            if step + 1 < max_new_tokens:
                logits = _logits(lm, prefix + generated)[-1]
        text = lm.tokenizer.decode(generated, skip_special_tokens=True)
    canonical = text.strip().strip(" .!\n\r\t\"'").casefold()
    matches = [i for i, c in enumerate(candidates) if canonical == c.strip().casefold()]
    logp = first.detach().cpu()
    kl = None
    if baseline_logp is not None:
        baseline = baseline_logp.double()
        kl = max(0.0, float((baseline.exp() * (baseline - logp.double())).sum()))
    prediction = int(torch.tensor(scores).argmax()) if scores and len(set(scores)) > 1 else None
    return {
        "candidate_logp": scores,
        "candidate_ids": suffixes,
        "candidate_probabilities": torch.tensor(scores).softmax(0).tolist() if scores else [],
        "prediction": prediction,
        "first_token_kl": kl,
        "generation": {
            "text": text,
            "token_ids": generated,
            "stop_reason": reason,
            "complete": reason == "eos",
            "answer": matches[0] if len(matches) == 1 else None,
        },
    }, logp
