"""Strict paired success and aligned cross-representation transfer denominators."""

from .transfer_tasks import PAIR_SCHEMES


def rate(flags):
    flags = list(flags)
    return {
        "count": len(flags),
        "correct": sum(flags),
        "accuracy": sum(flags) / len(flags) if flags else None,
    }


def paired_metrics(predictions, samples, sequence, baseline):
    groups = {}
    for prediction, q, original in zip(predictions, samples, baseline, strict=True):
        world = groups.setdefault(q.pair_id, {}).setdefault(q.world.scheme, {})
        if q.node in world:
            raise ValueError("Paired metrics require unique questions; duplicate node")
        world[q.node] = (q, prediction, original)
    pairs = []
    for key, worlds in groups.items():
        if set(worlds) not in [set(s) for s in PAIR_SCHEMES.values()]:
            raise ValueError("Paired metrics require a complete COPY/NOT pair")
        rows = []
        for nodes in worlds.values():
            first = next(iter(nodes.values()))[0]
            if set(nodes) != set(first.world.values()):
                raise ValueError("Paired metrics require complete states")
            rows.extend(nodes.values())
        first, second = [next(iter(nodes.values()))[0] for nodes in worlds.values()]
        if (first.world.source, first.world.switch, first.world.flag, first.aliases) != (
            second.world.source,
            second.world.switch,
            second.world.flag,
            second.aliases,
        ):
            raise ValueError("Pair must share inputs and aliases")
        affected = set().union(*(first.world.affected(op) for op in sequence))
        # With no intervention, the pair measures complete natural understanding.
        checked = affected or set(first.world.values())
        all_correct = all(p == q.answer(sequence) for q, p, _ in rows)
        both_base = all(p == q.answer() for q, _, p in rows)
        opposite = any(
            first.world.values(sequence)[n] != second.world.values(sequence)[n] for n in checked
        )
        pairs.append(
            {
                "pair_id": key,
                "all_nodes_correct": all_correct,
                "affected_nodes_correct": all(
                    p == q.answer(sequence) for q, p, _ in rows if q.node in checked
                ),
                "baseline_correct": both_base,
                "opposite_consequences": opposite,
            }
        )
    if not pairs:
        raise ValueError("Provide at least one complete pair")
    return {
        "all_nodes_correct": rate(p["all_nodes_correct"] for p in pairs),
        "affected_nodes_correct": rate(p["affected_nodes_correct"] for p in pairs),
        "opposite_consequences": rate(
            p["affected_nodes_correct"] for p in pairs if p["opposite_consequences"]
        ),
        "baseline_correct_pairs": rate(
            p["all_nodes_correct"] for p in pairs if p["baseline_correct"]
        ),
        "pairs": pairs,
    }


def transfer_selection(metrics, preservation_weight):
    return (
        metrics["paired"]["all_nodes_correct"]["accuracy"]
        - preservation_weight * metrics["protected_damage"]["rate"],
        metrics["paired"]["affected_nodes_correct"]["accuracy"],
        metrics["equation_consistency"]["all_satisfied"],
        metrics["overall"]["accuracy"],
    )


def transfer_agreement(source, target):
    left = {p["pair_id"]: p for p in source["pairs"]}
    right = {p["pair_id"]: p for p in target["pairs"]}
    if (
        not left
        or left.keys() != right.keys()
        or len(left) != len(source["pairs"])
        or len(right) != len(target["pairs"])
    ):
        raise ValueError("Cross-representation metrics require aligned unique pairs")
    pairs = [(p, right[k]) for k, p in left.items()]
    return {
        "joint_all_nodes_correct": rate(
            a["all_nodes_correct"] and b["all_nodes_correct"] for a, b in pairs
        ),
        "joint_affected_nodes_correct": rate(
            a["affected_nodes_correct"] and b["affected_nodes_correct"] for a, b in pairs
        ),
        "target_given_source_correct": rate(
            b["all_nodes_correct"] for a, b in pairs if a["all_nodes_correct"]
        ),
        "baseline_correct_in_both": rate(
            a["all_nodes_correct"] and b["all_nodes_correct"]
            for a, b in pairs
            if a["baseline_correct"] and b["baseline_correct"]
        ),
    }
