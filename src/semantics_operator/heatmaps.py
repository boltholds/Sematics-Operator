"""Read-only, full-coordinate comparisons of a minimal input pair at every block."""

from string import Formatter

import torch

from .localization import discover_sites, hidden

DEFAULT_TEMPLATE = "Слово: {word}.\nЗначение:"


def align_tokens(a, b):
    """Anchor exact prefix/suffix IDs; never zip unequal replacement spans.

    Equal-length replacements are positional comparisons, not semantic matches.
    Unmatched rows carry no delta. Indices always refer to the original sequence.
    """
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(a), len(b)) - prefix and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1
    rows = [{"a": i, "b": i, "kind": "prefix"} for i in range(prefix)]
    middle_a = list(range(prefix, len(a) - suffix))
    middle_b = list(range(prefix, len(b) - suffix))
    if len(middle_a) == len(middle_b):
        rows.extend(
            {"a": i, "b": j, "kind": "replacement"} for i, j in zip(middle_a, middle_b, strict=True)
        )
    else:
        rows.extend({"a": i, "b": None, "kind": "unmatched_a"} for i in middle_a)
        rows.extend({"a": None, "b": j, "kind": "unmatched_b"} for j in middle_b)
    rows.extend(
        {"a": len(a) - suffix + i, "b": len(b) - suffix + i, "kind": "suffix"}
        for i in range(suffix)
    )
    return rows


def aligned_matrices(a, b, rows):
    x = torch.full((len(rows), a.shape[-1]), float("nan"))
    y = torch.full_like(x, float("nan"))
    for k, row in enumerate(rows):
        if row["a"] is not None:
            x[k] = a[row["a"]]
        if row["b"] is not None:
            y[k] = b[row["b"]]
    return x, y, y - x


@torch.inference_mode()
def capture_blocks(lm, ids, sites):
    captures, handles = {}, []
    try:
        for index, site in sites.items():

            def hook(module, args, output, index=index):
                if index in captures:
                    raise ValueError("A block executed more than once")
                tensor = hidden(output)
                if tensor.shape[:2] != (1, len(ids)):
                    raise ValueError("Expected one complete unpadded token sequence")
                value = tensor[0].detach().float().cpu().clone()
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Non-finite block activations")
                captures[index] = value

            handles.append(lm.model.get_submodule(site).register_forward_hook(hook))
        input_ids, mask = lm._batch([ids])
        lm.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        if captures.keys() != sites.keys():
            raise ValueError("Some requested blocks did not execute")
    finally:
        for handle in handles:
            handle.remove()
    return captures


def _vector_metrics(a, b):
    a, b = a.double(), b.double()
    denom = float(a.norm() * b.norm())
    return {
        "delta_rms": float((b - a).square().mean().sqrt()),
        "delta_l2": float((b - a).norm()),
        "cosine": float(((a @ b) / denom).clamp(-1, 1)) if denom else None,
    }


def compare_pair(lm, pair, *, template=DEFAULT_TEMPLATE, prompt_format="raw", layers=None):
    fields = [
        (name, spec, conv)
        for _, name, spec, conv in Formatter().parse(template)
        if name is not None
    ]
    if fields != [("word", "", None)]:
        raise ValueError("template must contain exactly one unformatted {word} placeholder")
    if len(pair) != 2 or any(not isinstance(w, str) or not w.strip() for w in pair):
        raise ValueError("A pair requires two nonempty words or values")
    if prompt_format not in ("raw", "chat"):
        raise ValueError("prompt_format must be raw or chat")
    if prompt_format == "chat" and not getattr(lm.tokenizer, "chat_template", None):
        raise ValueError("chat format requires a tokenizer chat_template; use raw")
    catalog = discover_sites(lm)
    chosen = sorted(catalog) if layers is None else sorted(set(layers))
    if not chosen or any(i not in catalog for i in chosen):
        raise ValueError(f"Unknown/empty block selection; available: {sorted(catalog)}")
    sites = {i: catalog[i]["block"] for i in chosen}
    inputs = {}
    for side, word in zip(("a", "b"), pair, strict=True):
        text = template.format(word=word)
        ids = (
            lm._prompt_ids(text)
            if prompt_format == "chat"
            else lm.tokenizer.encode(text, add_special_tokens=True)
        )
        if not ids:
            raise ValueError("Input encoded to an empty sequence")
        inputs[side] = {
            "word": word,
            "text": text,
            "ids": ids,
            "tokens": lm.tokenizer.convert_ids_to_tokens(ids),
            "decoded": lm.tokenizer.decode(ids, skip_special_tokens=False),
        }
    a_ids, b_ids = inputs["a"]["ids"], inputs["b"]["ids"]
    rows = align_tokens(a_ids, b_ids)
    same = a_ids == b_ids
    anchor = (
        {"a": len(a_ids) - 1, "b": len(b_ids) - 1} if same or rows[-1]["kind"] == "suffix" else None
    )
    a = capture_blocks(lm, a_ids, sites)
    b = capture_blocks(lm, b_ids, sites)
    repeat = capture_blocks(lm, a_ids, sites)
    repeat_error = max(float((a[i] - repeat[i]).abs().max()) for i in sites)
    del repeat
    tensors, summaries = {}, []
    amplitude, delta_amplitude = 0.0, 0.0
    for index, site in sites.items():
        if a[index].shape[-1] != b[index].shape[-1]:
            raise ValueError("Block hidden widths differ between paired inputs")
        _, _, delta = aligned_matrices(a[index], b[index], rows)
        key = f"layer_{index:03d}"
        tensors.update({f"{key}.a": a[index], f"{key}.b": b[index], f"{key}.delta": delta})
        amplitude = max(amplitude, float(a[index].abs().max()), float(b[index].abs().max()))
        valid = delta[torch.isfinite(delta)]
        if valid.numel():
            delta_amplitude = max(delta_amplitude, float(valid.abs().max()))
        summaries.append(
            {
                "index": index,
                "module": site,
                "key": key,
                "shape_a": list(a[index].shape),
                "shape_b": list(b[index].shape),
                "anchor_metrics": _vector_metrics(a[index][anchor["a"]], b[index][anchor["b"]])
                if anchor
                else None,
            }
        )
    warnings = []
    if pair[0] != pair[1] and same:
        warnings.append(
            "Different texts produced identical token IDs; no lexical contrast reached the model."
        )
    if len(a_ids) != len(b_ids):
        warnings.append(
            "Token lengths differ: suffix positions shift; deltas include positional effects."
        )
    if anchor is None:
        warnings.append("No shared suffix token: the final-position overview is omitted.")
    if repeat_error > 0:
        warnings.append(
            "Repeated A differs numerically: compare effect sizes with repeat_a_max_abs."
        )
    if lm.tokenizer.unk_token_id is not None and any(
        lm.tokenizer.unk_token_id in ids for ids in (a_ids, b_ids)
    ):
        warnings.append("Input contains unknown-token IDs; inspect the token tables.")
    return {
        "inputs": inputs,
        "prompt_format": prompt_format,
        "alignment": rows,
        "anchor": anchor,
        "identical_token_ids": same,
        "repeat_a_max_abs": repeat_error,
        "layers": summaries,
        "warnings": warnings,
        "scales": {"activation_max_abs": amplitude, "delta_max_abs": delta_amplitude},
    }, tensors
