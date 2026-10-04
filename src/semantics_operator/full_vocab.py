"""Teacher-forced digit + EOS learning, with vocabulary-wide locality."""

from contextlib import nullcontext
from enum import StrEnum

import torch
from torch.nn import functional as F

from .positions import ReftPosition, intervention_positions
from .reft import activation_intervention


class LossMode(StrEnum):
    FULL_VOCAB = "full_vocab"
    BINARY = "binary"


def answer_sequences(lm, values):
    eos = lm.eos_token_ids()
    if not eos:
        raise ValueError("Full-vocabulary training requires an EOS token")
    result = []
    for value in values:
        if value not in (0, 1):
            raise ValueError("Expected a binary answer")
        digit = lm.tokenizer.encode(str(value), add_special_tokens=False)
        if not digit or any(token in eos for token in digit):
            raise ValueError("Answer digit must be nonempty and must not contain EOS")
        result.append(digit + [eos[0]])
    return result


def teacher_forced_logits(
    lm, prompts, targets, *, site=None, transform=None, position=ReftPosition.ANSWER
):
    if not prompts or len(prompts) != len(targets) or any(not t for t in targets):
        raise ValueError("Provide one nonempty completion for every prompt")
    prefixes = [lm._prompt_ids(p) for p in prompts]
    sequences = [p + t[:-1] for p, t in zip(prefixes, targets, strict=True)]
    ids, mask = lm._batch(sequences)
    context = (
        activation_intervention(lm, site, intervention_positions(lm, prompts, position), transform)
        if site is not None and transform is not None
        else nullcontext()
    )
    with context:
        logits = lm.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return [
        logits[i, len(p) - 1 : len(p) - 1 + len(t)].float()
        for i, (p, t) in enumerate(zip(prefixes, targets, strict=True))
    ]


@torch.no_grad()
def reference_distributions(lm, samples):
    """CPU cache: full log probabilities under each natural answer's teacher-forced history."""
    result = []
    for start in range(0, len(samples), 2):
        batch = samples[start : start + 2]
        logits = teacher_forced_logits(
            lm, [q.prompt() for q in batch], answer_sequences(lm, [q.answer() for q in batch])
        )
        result.extend(x.log_softmax(-1).cpu() for x in logits)
    return result


def full_vocab_loss(
    logits, reference, targets, affected, locality_weight, *, normalization_counts=None
):
    """Average tokens within each question, then mean task/protected questions separately.

    Protected references use the same natural-answer histories as edited logits.
    Affected questions do not use the reference: their target history can differ.
    """
    task_count, local_count = normalization_counts or (sum(affected), len(affected) - sum(affected))
    zero = logits[0].sum() * 0
    task, locality, rows = zero, zero, []
    for i, (scores, ids, is_affected) in enumerate(zip(logits, targets, affected, strict=True)):
        logp = scores.float().log_softmax(-1)
        ce = F.nll_loss(logp, torch.tensor(ids, device=scores.device), reduction="mean")
        kl = zero
        row = {"target_ce": float(ce.detach()), "locality_kl": None}
        if is_affected:
            task = task + ce / task_count
        else:
            original = reference[i].detach().to(logp)
            if original.shape != logp.shape:
                raise ValueError("Protected distributions need identical teacher-forced histories")
            kl = F.kl_div(logp, original, log_target=True, reduction="none").sum(-1).mean()
            locality = locality + kl / local_count
            row["locality_kl"] = float(kl.detach())
            row["first_token_matches_base"] = bool(scores[0].argmax() == original[0].argmax())
        rows.append(row)
    return (
        task + locality_weight * locality,
        {"task_ce": float(task.detach()), "locality_kl": float(locality.detach())},
        rows,
    )
