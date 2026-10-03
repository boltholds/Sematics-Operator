"""Train-only balanced scheduling with early witnesses distinguishing primitives."""

import random

from .world import OPERATORS, Intervention, Question


def training_batches(
    samples: tuple[Question, ...], intervention: Intervention, *, steps: int, seed: int
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pools = [
        [i for i, q in enumerate(samples) if (q.answer((intervention,)) != q.answer()) == changed]
        for changed in (True, False)
    ]
    if any(not pool for pool in pools):
        raise ValueError("Training needs both changed and unchanged examples")
    rivals = [op for op in OPERATORS if op != intervention]

    def cycle(pool):
        remaining = list(pool)
        rng.shuffle(remaining)
        prefix = []
        # Reserve witnesses before ordinary coverage, without dropping any example.
        for rival in rivals:
            witness = next(
                (
                    i
                    for i in remaining
                    if samples[i].answer((intervention,)) != samples[i].answer((rival,))
                ),
                None,
            )
            if witness is not None:
                prefix.append(witness)
                remaining.remove(witness)
        return prefix + remaining

    streams = []
    for pool in pools:
        stream = []
        while len(stream) < steps:
            stream.extend(cycle(pool))
        streams.append(stream[:steps])
    return list(zip(*streams, strict=True))


def coverage(samples, intervention, batches) -> dict:
    indices = [i for batch in batches for i in batch]
    changed = sum(samples[i].answer((intervention,)) != samples[i].answer() for i in indices)
    return {
        "presentations": len(indices),
        "unique_questions": len(set(indices)),
        "available_questions": len(samples),
        "changed_presentations": changed,
        "unchanged_presentations": len(indices) - changed,
        "distinguishing_presentations": {
            rival.key: sum(
                samples[i].answer((intervention,)) != samples[i].answer((rival,)) for i in indices
            )
            for rival in OPERATORS
            if rival != intervention
        },
        "batch_question_keys": [[samples[i].key for i in batch] for batch in batches],
    }
