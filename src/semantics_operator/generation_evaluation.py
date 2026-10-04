"""Unconstrained generation metrics for frozen-model causal interventions."""

from .answer_protocol import parse_generation
from .causal_tasks import answer_metrics
from .reft import intervention_generate


def generate_questions(
    lm, samples, site, transform, *, sequence=(), max_new_tokens=16, progress=lambda _: None
):
    outputs = []
    for i, q in enumerate(samples):
        prompt = q.prompt(sequence)
        generated = intervention_generate(
            lm, [prompt], site, transform, max_new_tokens=max_new_tokens
        )[0]
        outputs.append({"prompt": prompt, **parse_generation(generated)})
        if i == 0 or (i + 1) % 20 == 0 or i + 1 == len(samples):
            progress(f"{i + 1}/{len(samples)}")
    return outputs


def generation_metrics(outputs, samples, sequence, baseline):
    records, groups = [], {}
    for output, q, base in zip(outputs, samples, baseline, strict=True):
        record = {
            **output,
            "key": q.key,
            "node": q.node.value,
            "style": q.style.value,
            "expected": q.answer(sequence),
            "base_expected": q.answer(),
            "changed": q.answer(sequence) != q.answer(),
            "base_prediction": base["prediction"],
        }
        records.append(record)
        groups.setdefault(q.state_key, []).append(record)
    result = answer_metrics(records)
    for name, changed in (("changed", True), ("unchanged", False)):
        rows = [r for r in records if r["changed"] == changed]
        result[name] = {
            "count": len(rows),
            "accuracy": sum(r["prediction"] == r["expected"] for r in rows) / len(rows)
            if rows
            else None,
        }
    result["all_nodes_correct"] = sum(
        all(r["prediction"] == r["expected"] for r in rows) for rows in groups.values()
    ) / len(groups)
    result["relay_lamp_correct"] = sum(
        all(r["prediction"] == r["expected"] for r in rows if r["node"] in ("relay", "lamp"))
        for rows in groups.values()
    ) / len(groups)
    by_node = {}
    for node in ("source", "switch", "flag"):
        rows = [
            r for r in records if r["node"] == node and r["base_prediction"] == r["base_expected"]
        ]
        by_node[node] = {
            "eligible": len(rows),
            "damaged": sum(r["prediction"] != r["expected"] for r in rows),
        }
    eligible, damaged = (sum(x[k] for x in by_node.values()) for k in ("eligible", "damaged"))
    result["protected_damage"] = {
        "eligible": eligible,
        "damaged": damaged,
        "rate": damaged / eligible if eligible else 0.0,
        "by_node": by_node,
    }
    return result


def attach_candidate_agreement(generated_metrics, candidate_metrics):
    pairs = [
        (g, c)
        for g, c in zip(generated_metrics["records"], candidate_metrics["records"], strict=True)
        if g["prediction"] is not None
    ]
    matches = sum(g["prediction"] == c["prediction"] for g, c in pairs)
    generated_metrics["agreement_with_candidates"] = {
        "valid_generated_count": len(pairs),
        "matches": matches,
        "rate": matches / len(pairs) if pairs else None,
    }
