"""Signed cross-pair delta directions at a common suffix-token identity."""

from itertools import combinations

import torch


def anchor_deltas(pair, tensors):
    """Retain only small copies; do not keep views into full token captures."""
    if pair["anchor"] is None:
        return {}
    return {
        layer["index"]: (
            tensors[layer["key"] + ".b"][pair["anchor"]["b"]]
            - tensors[layer["key"] + ".a"][pair["anchor"]["a"]]
        ).clone()
        for layer in pair["layers"]
    }


def compare_directions(pairs, vectors):
    if not pairs or len(pairs) != len(vectors):
        raise ValueError("Provide matching nonempty pair metadata and delta vectors")
    indices = [layer["index"] for layer in pairs[0]["layers"]]
    if any([layer["index"] for layer in p["layers"]] != indices for p in pairs):
        raise ValueError("Cross-pair directions require the same block indices")
    descriptions, warnings = [], []
    for i, pair in enumerate(pairs):
        anchor = pair["anchor"]
        ids = [pair["inputs"][s]["ids"][anchor[s]] for s in ("a", "b")] if anchor else []
        if ids and ids[0] != ids[1]:
            raise ValueError("A pair anchor must be the same token in both inputs")
        descriptions.append(
            {
                "index": i,
                "label": " -> ".join(pair["inputs"][s]["word"] for s in ("a", "b")),
                "anchor_token_id": ids[0] if ids else None,
                "anchor_positions": anchor,
                "token_counts": {s: len(pair["inputs"][s]["ids"]) for s in ("a", "b")},
            }
        )
        if anchor and anchor["a"] != anchor["b"]:
            warnings.append(
                f"Pair {i + 1}: A/B anchor positions differ; delta includes positional effects."
            )
    comparisons = []
    for i, j in combinations(range(len(pairs)), 2):
        a, b = descriptions[i], descriptions[j]
        missing = a["anchor_token_id"] is None or b["anchor_token_id"] is None
        status = (
            "no_shared_suffix"
            if missing
            else "different_anchor_token"
            if a["anchor_token_id"] != b["anchor_token_id"]
            else "ok"
        )
        same_positions = a["anchor_positions"] == b["anchor_positions"] if not missing else None
        comparisons.append(
            {"left": i, "right": j, "status": status, "same_anchor_positions": same_positions}
        )
        if status == "ok" and not same_positions:
            warnings.append(
                f"Pairs {i + 1}/{j + 1}: anchor positions differ across pairs; cosine may include positional effects."
            )
    layers = []
    for index in indices:
        norms, statuses, normalized = [], [], []
        width = None
        for p, v in zip(pairs, vectors, strict=True):
            if p["anchor"] is None:
                norms.append(None)
                statuses.append("no_shared_suffix")
                normalized.append(None)
                continue
            delta = v[index].double()
            if delta.ndim != 1 or delta.numel() == 0:
                raise ValueError("Expected nonempty one-dimensional delta vectors")
            if not torch.isfinite(delta).all():
                raise FloatingPointError("Non-finite delta direction")
            if width is not None and delta.numel() != width:
                raise ValueError("Delta dimensions differ between pairs")
            width = delta.numel()
            norm = float(delta.norm())
            rms = float(delta.square().mean().sqrt())
            status = (
                "zero_delta"
                if norm == 0
                else "below_repeat_noise"
                if rms <= 2 * p["repeat_a_max_abs"]
                else "ok"
            )
            norms.append(norm)
            statuses.append(status)
            normalized.append(delta / norm if status == "ok" else None)
        matrix = [[None for _ in pairs] for _ in pairs]
        for i, a in enumerate(normalized):
            for j, b in enumerate(normalized):
                if (
                    a is not None
                    and b is not None
                    and descriptions[i]["anchor_token_id"] == descriptions[j]["anchor_token_id"]
                ):
                    matrix[i][j] = float((a @ b).clamp(-1, 1))
        layers.append(
            {"index": index, "cosine": matrix, "delta_l2": norms, "direction_status": statuses}
        )
    return {
        "definition": "cosine(delta_i, delta_j), delta = B - A at last shared suffix token",
        "noise_rule": "Direction undefined when delta RMS <= 2 * repeat_A_max_abs; conservative observed-noise screen, not a statistical confidence interval.",
        "pairs": descriptions,
        "comparisons": comparisons,
        "layers": layers,
        "warnings": warnings,
        "interpretation": "Agreement is descriptive and depends on tokenization, context and anchor position. It does not establish a causal or language-universal operator.",
    }
