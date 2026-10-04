"""Predeclared RU-to-EN temperature steering with protected-property controls."""

import math

import torch

from .heatmap_directions import anchor_deltas
from .heatmaps import DEFAULT_TEMPLATE, compare_pair
from .localization import discover_sites
from .word_transfer_runtime import measure

QUESTIONS = {
    "temperature": ("Is the temperature cold or hot?", ("cold", "hot")),
    "color": ("What color are the objects, blue or red?", ("blue", "red")),
    "object": ("What kind of object is described, cup or bowl?", ("cup", "bowl")),
    "count": ("How many objects are there, two or three?", ("two", "three")),
}


def make_directions(source, native, *, seed):
    if source.ndim != 1 or source.shape != native.shape:
        raise ValueError("Source and native directions must have the same vector shape")
    if not torch.isfinite(source).all() or not torch.isfinite(native).all():
        raise ValueError("Non-finite source/native direction")
    if float(source.norm()) == 0 or float(native.norm()) == 0:
        raise ValueError("Cannot test a zero source/native contrast")
    directions = {"ru_delta": source.clone(), "en_delta": native.clone()}
    for index in range(3):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        random = torch.randn(source.shape, generator=generator)
        directions[f"random_{index}"] = random * (source.norm() / random.norm())
    return directions


def _cases(lm, template, anchor_id):
    cases, expected = [], {}
    for context, description, value in (
        ("blue_cups", "There are two blue cups.\n", 0),
        ("red_bowls", "There are three red bowls.\n", 1),
    ):
        for state, word in enumerate(("cold", "hot")):
            text = description + template.format(word=word)
            ids = lm.tokenizer.encode(text, add_special_tokens=True)
            if not ids or ids[-1] != anchor_id:
                raise ValueError("Recipient must end in the source's shared anchor token")
            key = f"{context}_{word}"
            queries = {}
            for name, (question, candidates) in QUESTIONS.items():
                suffix = "\n" + question + " Reply with exactly one word.\nAnswer:"
                # Preserve the captured prefix byte-for-byte at the token level.
                qids = ids + lm.tokenizer.encode(suffix, add_special_tokens=False)
                queries[name] = {"text": text + suffix, "ids": qids, "candidates": list(candidates)}
            cases.append(
                {
                    "id": key,
                    "context": context,
                    "state": state,
                    "word": word,
                    "prefix": text,
                    "prefix_ids": ids,
                    "anchor_position": len(ids) - 1,
                    "questions": queries,
                }
            )
            expected[key] = {"temperature": state, "color": value, "object": value, "count": value}
    return cases, expected


def _prediction(result, mode):
    if mode == "candidate":
        return result["prediction"]
    g = result["generation"]
    return g["answer"] if g["complete"] else None


def summarize_cases(cases, results, baseline, expected, *, flip=True):
    summary = {}
    for mode in ("candidate", "greedy"):
        temp, protected, joint, paired = [], [], {}, {}
        damaged = eligible = invalid = 0
        by_property = {}
        for case in cases:
            key = case["id"]
            correct, base_correct = [], []
            for name, old in expected[key].items():
                target = 1 - old if name == "temperature" and flip else old
                prediction = _prediction(results[key][name], mode)
                previous = _prediction(baseline[key][name], mode)
                success = prediction == target
                invalid += prediction is None
                correct.append(success)
                base_correct.append(previous == old)
                if name == "temperature":
                    temp.append(success)
                else:
                    protected.append(success)
                    eligible += previous == old
                    damaged += previous == old and not success
                    entry = by_property.setdefault(
                        name, {"correct": 0, "count": 0, "damaged": 0, "eligible": 0}
                    )
                    entry["correct"] += success
                    entry["count"] += 1
                    entry["eligible"] += previous == old
                    entry["damaged"] += previous == old and not success
            joint[key] = all(correct)
            paired.setdefault(case["context"], []).append((all(correct), all(base_correct)))
        pairs = [values for values in paired.values() if len(values) == 2]
        eligible_pairs = [p for p in pairs if all(v[1] for v in p)]
        summary[mode] = {
            "temperature": {"correct": sum(temp), "count": len(temp)},
            "protected": {"correct": sum(protected), "count": len(protected)},
            "protected_by_property": by_property,
            "protected_damage": {"damaged": damaged, "eligible": eligible},
            "joint": {"correct": sum(joint.values()), "count": len(joint)},
            "paired_joint": {
                "correct": sum(all(v[0] for v in p) for p in pairs),
                "count": len(pairs),
            },
            "baseline_correct_paired_joint": {
                "correct": sum(all(v[0] for v in p) for p in eligible_pairs),
                "count": len(eligible_pairs),
            },
            "invalid_or_incomplete": invalid,
        }
    for name in ("color", "object", "count"):
        values = [
            r[name]["first_token_kl"]
            for r in results.values()
            if name in r and r[name]["first_token_kl"] is not None
        ]
        summary.setdefault("protected_first_token_kl", {})[name] = (
            sum(values) / len(values) if values else None
        )
    return summary


@torch.inference_mode()
def run_word_transfer(
    lm,
    cfg,
    *,
    layer=13,
    strengths=None,
    max_new_tokens=16,
    progress=lambda _: None,
):
    strengths = [1.0] if strengths is None else list(strengths)
    if any(not math.isfinite(a) or a < 0 for a in strengths):
        raise ValueError("Use finite nonnegative strengths; reverse-sign control is automatic")
    strengths = sorted({1.0, *strengths} - {0.0})
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    catalog = discover_sites(lm)
    if layer not in catalog:
        raise ValueError(f"Unknown block {layer}; available: {sorted(catalog)}")
    site = catalog[layer]["block"]
    metadata, extracted = [], []
    for pair in (("холодно", "жарко"), ("cold", "hot")):
        progress(f"Extracting block {layer}: {pair[0]} -> {pair[1]}")
        report, tensors = compare_pair(lm, pair, layers=[layer])
        if report["anchor"] is None:
            raise ValueError("Contrast requires a shared final token")
        vector = anchor_deltas(report, tensors)[layer]
        if float(vector.square().mean().sqrt()) <= 2 * report["repeat_a_max_abs"]:
            raise ValueError("Contrast is zero or below observed repeat noise")
        metadata.append(report)
        extracted.append(vector)
    anchor_ids = [m["inputs"]["a"]["ids"][m["anchor"]["a"]] for m in metadata]
    if anchor_ids[0] != anchor_ids[1]:
        raise ValueError("Source and target anchor token IDs differ")
    if anchor_ids[0] in lm.tokenizer.all_special_ids or not lm.tokenizer.decode(
        [anchor_ids[0]], skip_special_tokens=False
    ).endswith(":"):
        raise ValueError(
            "word-transfer requires the final colon anchor; this tokenizer adds a trailing "
            "special token or does not expose the terminal colon at that position"
        )
    directions = make_directions(*extracted, seed=cfg.seed)
    cases, expected = _cases(lm, DEFAULT_TEMPLATE, anchor_ids[0])
    baselines, base_logp, raw_baselines = {}, {}, {}
    for word in ("cold", "hot"):
        ids = lm.tokenizer.encode(DEFAULT_TEMPLATE.format(word=word), add_special_tokens=True)
        raw_baselines[word], _ = measure(
            lm, ids, site, len(ids) - 1, None, max_new_tokens=max_new_tokens
        )
    for case in cases:
        progress(f"Baseline: {case['id']}")
        baselines[case["id"]], base_logp[case["id"]] = {}, {}
        for name, q in case["questions"].items():
            result, logp = measure(
                lm,
                q["ids"],
                site,
                case["anchor_position"],
                None,
                candidates=q["candidates"],
                max_new_tokens=max_new_tokens,
            )
            result["first_token_kl"] = 0.0
            baselines[case["id"]][name], base_logp[case["id"]][name] = result, logp
    # A real zero hook uses the same forward layout and fixed position as trials.
    first_case = cases[0]
    q = first_case["questions"]["temperature"]
    zero, logp = measure(
        lm,
        q["ids"],
        site,
        first_case["anchor_position"],
        torch.zeros_like(extracted[0]),
        candidates=q["candidates"],
        max_new_tokens=max_new_tokens,
    )
    original = baselines[first_case["id"]]["temperature"]
    zero_error = float((logp - base_logp[first_case["id"]]["temperature"]).abs().max())
    if zero_error > 1e-5 or zero["generation"] != original["generation"]:
        raise RuntimeError("Word-transfer zero intervention sanity check failed")
    conditions = {
        "baseline": {
            "direction": None,
            "alpha": 0.0,
            "cases": baselines,
            "raw_completions": raw_baselines,
        }
    }
    configurations = (
        [("ru_delta", a) for a in strengths]
        + [("ru_delta", -1.0), ("en_delta", 1.0)]
        + [(f"random_{i}", 1.0) for i in range(3)]
    )
    for direction, alpha in configurations:
        # Float repr preserves distinct user strengths; :g would round and could
        # overwrite the exact alpha=1 primary with a nearby sweep value.
        key = f"{direction}@{str(alpha).removesuffix('.0')}"
        progress(f"Condition {key} (both cold->hot and hot->cold)")
        results, raw = {}, {}
        for state, word in enumerate(("cold", "hot")):
            vector = directions[direction] * (alpha * (1 if state == 0 else -1))
            ids = metadata[1]["inputs"]["a" if state == 0 else "b"]["ids"]
            raw[word], _ = measure(
                lm, ids, site, len(ids) - 1, vector, max_new_tokens=max_new_tokens
            )
        for case in cases:
            vector = directions[direction] * (alpha * (1 if case["state"] == 0 else -1))
            results[case["id"]] = {}
            for name, q in case["questions"].items():
                result, _ = measure(
                    lm,
                    q["ids"],
                    site,
                    case["anchor_position"],
                    vector,
                    candidates=q["candidates"],
                    max_new_tokens=max_new_tokens,
                    baseline_logp=base_logp[case["id"]][name],
                )
                old, target = expected[case["id"]][name], expected[case["id"]][name]
                if name == "temperature":
                    target = 1 - old
                base = baselines[case["id"]][name]
                result["target_log_odds_change"] = (
                    result["candidate_logp"][target] - result["candidate_logp"][1 - target]
                ) - (base["candidate_logp"][target] - base["candidate_logp"][1 - target])
                results[case["id"]][name] = result
        conditions[key] = {
            "direction": direction,
            "alpha": alpha,
            "cases": results,
            "raw_completions": raw,
        }
    rollback = 0.0
    for case in cases:
        q = case["questions"]["temperature"]
        _, logp = measure(
            lm,
            q["ids"],
            site,
            case["anchor_position"],
            None,
            candidates=q["candidates"],
            max_new_tokens=1,
        )
        rollback = max(rollback, float((logp - base_logp[case["id"]]["temperature"]).abs().max()))
    if rollback > 1e-5:
        raise RuntimeError("Word-transfer rollback check failed")
    for condition in conditions.values():
        condition["metrics"] = summarize_cases(cases, condition["cases"], baselines, expected)
    report = {
        "experiment": "word_direction_transfer_v1",
        "model": {
            "profile": cfg.profile,
            "path": str(cfg.model_path),
            "device": str(lm.device),
            "dtype": str(next(lm.model.parameters()).dtype),
        },
        "site": {
            "layer": layer,
            "path": site,
            "position": "last prefix token, before every question",
        },
        "seed": cfg.seed,
        "template": DEFAULT_TEMPLATE,
        "prompt_format": "raw",
        "primary_condition": "ru_delta@1",
        "max_new_tokens": max_new_tokens,
        "strengths": [0.0, *strengths],
        "extraction": {"ru": metadata[0], "en": metadata[1]},
        "directions": {
            k: {
                "norm": float(v.norm()),
                "cosine_to_ru": float(
                    torch.nn.functional.cosine_similarity(v, extracted[0], dim=0)
                ),
            }
            for k, v in directions.items()
        },
        "cases": cases,
        "expected_before": expected,
        "conditions": conditions,
        "baseline_understanding": summarize_cases(
            cases, baselines, baselines, expected, flip=False
        ),
        "zero_hook_max_logp_difference": zero_error,
        "rollback_max_logp_difference": rollback,
        "limitations": [
            "One temperature pair and two small protected-property contexts; exploratory causal steering, not evidence of general reasoning.",
            "All words use the same Russian wrapper. Only the contrasted words change language; audit questions and metadata are English.",
            "Raw tokenization matches heatmaps. Prefix/question/answer segments are encoded separately to preserve the intervention token exactly. No chat template or forced space.",
            "Source and target token counts/positions can differ; inspect extraction token tables. Contextual audits add position/context shift.",
            "Block and strengths are predeclared; alpha=1 is always primary. No target-based selection. Repeated manual runs can introduce selection bias.",
            "en_delta is a target-informed diagnostic, not RU-only transfer. Random controls use three local RNG seeds with the Russian delta norm.",
            "Candidate probabilities are normalized only over the supplied alternatives; scores sum all continuation tokens without EOS. Free generation is scored separately and must reach EOS with exactly one allowed answer.",
            "KL covers the full vocabulary at the first answer token only, not the complete generated sequence. Protected damage counts only baseline-correct questions; report denominators.",
            "For hot->cold the vector sign is reversed. A successful pair requires both directions and all protected properties.",
            "Frozen model weights; only one block-output token is edited, including during cache-free generation.",
        ],
    }
    return report, directions
