"""Preservation and equation checks over complete counterfactual states."""

from .causal_tasks import CHAIN_SCHEMES, Scheme


def protected_damage(predictions, baseline, samples, sequence):
    by_node = {q.node.value: {"eligible": 0, "damaged": 0} for q in samples}
    for prediction, original, q in zip(predictions, baseline, samples, strict=True):
        affected = set().union(*(q.world.affected(op) for op in sequence))
        if q.node not in affected and original == q.answer():
            row = by_node[q.node.value]
            row["eligible"] += 1
            row["damaged"] += int(prediction != q.answer(sequence))
    eligible = sum(v["eligible"] for v in by_node.values())
    damaged = sum(v["damaged"] for v in by_node.values())
    return {
        "eligible": eligible,
        "damaged": damaged,
        "rate": damaged / eligible if eligible else 0.0,
        "by_node": by_node,
        "scope": "all_structurally_unaffected_nodes",
    }


def equation_consistency(predictions, samples, sequence):
    """Check each equation against predicted parents, including do-replacements.

    Root input correctness is deliberately separate: a consistent prediction
    can describe the wrong inputs. all_nodes_correct tests the complete target.
    Missing/invalid predictions fail every equation that needs that value.
    """
    overrides = {op.node.value: op.value for op in sequence}
    groups = {}
    for prediction, q in zip(predictions, samples, strict=True):
        world, values = groups.setdefault(q.state_key, (q.world, {}))
        if q.node.value in values:
            raise ValueError("Equation checks require unique questions per state/style")
        values[q.node.value] = prediction
    by_equation, states = {}, []
    for (name, style), (world, values) in groups.items():
        rules = {
            "relay": (
                ("source", "switch"),
                lambda a, b, scheme=world.scheme: a | b if scheme == Scheme.OR_COPY else a & b,
            ),
        }
        dependent = "bridge" if world.scheme in CHAIN_SCHEMES else "lamp"
        if world.scheme in (Scheme.AND_INVERTED, Scheme.AND_INVERTED_CHAIN):
            rules[dependent] = (("relay",), lambda r: 1 - r)
        elif world.scheme in (Scheme.AND_XOR, Scheme.AND_XOR_CHAIN):
            rules[dependent] = (("relay", "flag"), lambda r, f: r ^ f)
        elif world.scheme == Scheme.AND_GATED:
            rules[dependent] = (("relay", "flag"), lambda r, f: r & f)
        else:
            rules[dependent] = (("relay",), lambda r: r)
        if world.scheme in CHAIN_SCHEMES:
            rules["lamp"] = (("bridge",), lambda b: b)
        for node, value in overrides.items():
            rules[node] = ((), lambda value=value: value)
        statuses = {}
        for node, (parents, evaluate) in rules.items():
            row = by_equation.setdefault(
                node, {"count": 0, "satisfied": 0, "violated": 0, "invalid": 0}
            )
            if any(values.get(n) not in (0, 1) for n in (node, *parents)):
                status = "invalid"
            else:
                status = (
                    "satisfied"
                    if values[node] == evaluate(*(values[p] for p in parents))
                    else "violated"
                )
            row["count"] += 1
            row[status] += 1
            statuses[node] = status
        states.append({"state": name, "style": style.value, "equations": statuses})
    checked = sum(v["count"] for v in by_equation.values())
    satisfied = sum(v["satisfied"] for v in by_equation.values())
    return {
        "state_count": len(states),
        "all_satisfied": sum(
            all(s == "satisfied" for s in row["equations"].values()) for row in states
        )
        / len(states),
        "equation_accuracy": satisfied / checked,
        "by_equation": by_equation,
        "states": states,
    }
