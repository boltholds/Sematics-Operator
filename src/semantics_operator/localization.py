"""Privileged donor localization across sites, layer sets and token boundaries."""

import json
import re
from datetime import UTC, datetime
from uuid import uuid4

import torch

from .steering import evaluate, measurement, selection_score, validation_questions
from .world import OPERATORS, questions


def common_prefix(lm):
    candidates = [lm.tokenizer.encode(f" {v}", add_special_tokens=False) for v in (0, 1)]
    prefix = []
    for a, b in zip(*candidates):
        if a != b:
            break
        prefix.append(a)
    if not candidates[0] or not candidates[1] or candidates[0] == candidates[1]:
        raise ValueError("Candidates must be nonempty and distinguishable")
    return prefix


def hidden(output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError("Localization requires a [batch, token, hidden] output")
    return tensor


@torch.no_grad()
def capture_tail(lm, prompts, sites, window, prefix):
    sequences = [lm._prompt_ids(p) + prefix for p in prompts]
    if window < 1 or min(map(len, sequences)) < window:
        raise ValueError("Token window exceeds available sequence")
    ids, mask = lm._batch(sequences)
    captures, handles = {}, []
    try:
        for site in sites:

            def hook(module, args, output, site=site):
                tensor = hidden(output)
                captures[site] = torch.stack(
                    [
                        tensor[row, len(seq) - window : len(seq)].detach().float().cpu()
                        for row, seq in enumerate(sequences)
                    ]
                )

            handles.append(lm.model.get_submodule(site).register_forward_hook(hook))
        lm.model(input_ids=ids, attention_mask=mask, use_cache=False)
        if set(captures) != set(sites):
            raise ValueError("Some localization sites did not execute")
    finally:
        for handle in handles:
            handle.remove()
    return captures


@torch.no_grad()
def capture_scoring_tail(lm, prompts, sites, window, *, candidates):
    """Capture prompt tails in the exact candidate-scoring forward layout.

    Returns [prompt, candidate, token, hidden] per site. The candidate axis
    retains each row's floating-point result; the captured positions precede
    every answer token. Reusing scores() keeps padding, row order and sequence
    lengths identical to the recipient's scoring pass for a self-patch.
    """
    ends = [len(lm._prompt_ids(p)) for p in prompts for _ in candidates]
    if not ends or window < 1 or min(ends) < window:
        raise ValueError("Token window exceeds available sequence")
    captures, handles = {}, []
    try:
        for site in sites:

            def hook(module, args, output, site=site):
                tensor = hidden(output)
                if tensor.shape[0] != len(ends):
                    raise ValueError("Captured rows must match candidate scoring")
                captures[site] = torch.stack(
                    [
                        tensor[row, end - window : end].detach().float().cpu()
                        for row, end in enumerate(ends)
                    ]
                ).reshape(len(prompts), len(candidates), window, tensor.shape[-1])

            handles.append(lm.model.get_submodule(site).register_forward_hook(hook))
        lm.scores(prompts, candidates=candidates)
        if set(captures) != set(sites):
            raise ValueError("Some localization sites did not execute")
    finally:
        for handle in handles:
            handle.remove()
    return captures


@torch.no_grad()
def patched_scores(
    lm, prompts, donor, sites, window, prefix, *, replace=True, candidates=(" 0", " 1")
):
    ends = [len(lm._prompt_ids(p)) + len(prefix) for p in prompts for _ in (0, 1)]
    handles = []
    try:
        for site in sites:

            def hook(module, args, output, site=site):
                tensor = hidden(output).clone()
                states = donor[site]
                if states.ndim == 4:
                    if states.shape[:2] != (len(prompts), 2):
                        raise ValueError("Scoring donors need two candidate rows per prompt")
                    values = states[:, :, -window:].flatten(0, 1).to(tensor)
                else:
                    values = states[:, -window:].repeat_interleave(2, dim=0).to(tensor)
                for row, end in enumerate(ends):
                    if replace:
                        tensor[row, end - window : end] = values[row]
                    else:
                        tensor[row, end - window : end] += values[row]
                if isinstance(output, tuple):
                    return (tensor, *output[1:])
                if isinstance(output, list):
                    return [tensor, *output[1:]]
                return tensor

            handles.append(lm.model.get_submodule(site).register_forward_hook(hook))
        return lm.scores(prompts, candidates=candidates).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()


def discover_sites(lm):
    result = {}
    for name in lm.linear_modules():
        if name.endswith(("down_proj", "feed_forward.w2", "fc2", "dense_4h_to_h")):
            match = re.match(r"(.+\.layers\.(\d+))\.", name)
            if match:
                result[int(match[2])] = {"mlp": name, "block": match[1]}
    if not result:
        raise ValueError("No supported decoder layer paths found")
    return result


def aligned_window(lm, sources, recipients, requested, prefix):
    # Match identical suffix tokens, never align unrelated positions by index alone.
    lengths = []
    for source, recipient in zip(sources, recipients, strict=True):
        a, b = lm._prompt_ids(source) + prefix, lm._prompt_ids(recipient) + prefix
        count = 0
        for x, y in zip(reversed(a), reversed(b)):
            if x != y:
                break
            count += 1
        lengths.append(min(requested, count))
    if min(lengths) < 1:
        raise ValueError("Donor and recipient need a shared suffix for patch alignment")
    return min(lengths)


@torch.no_grad()
def run_localization(
    lm, cfg, *, layer_sets=None, windows=None, boundaries=None, progress=lambda _: None
):
    catalog = discover_sites(lm)
    indices = sorted(catalog)
    if layer_sets is None:
        layer_sets = (
            [(i,) for i in indices]
            + [tuple(indices[i : i + 3]) for i in range(len(indices) - 2)]
            + [tuple(indices)]
        )
    groups = list(dict.fromkeys(tuple(sorted(set(g))) for g in layer_sets))
    if not groups or any(not g or any(i not in catalog for i in g) for g in groups):
        raise ValueError("Layer sets must contain existing layer indices")
    windows = [1, 4] if windows is None else sorted(set(windows))
    boundaries = ["prompt", "decision"] if boundaries is None else list(dict.fromkeys(boundaries))
    if (
        not windows
        or any(w < 1 for w in windows)
        or not boundaries
        or any(b not in ("prompt", "decision") for b in boundaries)
    ):
        raise ValueError("Use positive windows and prompt/decision boundaries")
    shared = common_prefix(lm)
    validation, test = validation_questions(), questions("test")
    baseline_val, baseline_test = evaluate(lm, validation, {}), evaluate(lm, test, {})
    trials, selected, tests = {}, {}, {}
    cache = {}

    def score(samples, op, config, self_patch=False):
        boundary = config["boundary"]
        prefix = shared if boundary == "decision" else []
        recipients = [q.prompt() for q in samples]
        sources = recipients if self_patch else [q.prompt((op,)) for q in samples]
        # Store all sites together in each donor forward, reuse across groups/windows.
        used_indices = sorted({i for group in groups for i in group})
        sites = [catalog[i][config["kind"]] for i in config["layers"]]
        actual = aligned_window(lm, sources, recipients, config["window"], prefix)
        key = (samples[0].world.name, op.key, boundary, self_patch)
        if key not in cache:
            max_window = aligned_window(lm, sources, recipients, max(windows), prefix)
            all_sites = [catalog[i][kind] for i in used_indices for kind in ("mlp", "block")]
            chunks = {site: [] for site in all_sites}
            for start in range(0, len(samples), 2):
                values = capture_tail(lm, sources[start : start + 2], all_sites, max_window, prefix)
                for site in all_sites:
                    chunks[site].append(values[site])
            cache[key] = {site: torch.cat(parts) for site, parts in chunks.items()}
        donor = cache[key]
        rows = []
        for start in range(0, len(samples), 2):
            rows.append(
                patched_scores(
                    lm,
                    recipients[start : start + 2],
                    {site: donor[site][start : start + 2] for site in sites},
                    sites,
                    actual,
                    prefix,
                )
            )
        return torch.cat(rows), actual

    for op in OPERATORS:
        trials[op.key] = []
        best = None
        for group in groups:
            for kind in ("mlp", "block"):
                for boundary in boundaries:
                    for window in windows:
                        config = {
                            "layers": list(group),
                            "kind": kind,
                            "boundary": boundary,
                            "window": window,
                        }
                        progress(f"Validation {op.key}: {config}")
                        scores, actual = score(validation, op, config)
                        objective = list(selection_score(scores, validation, op, baseline_val))
                        entry = {
                            **config,
                            "actual_window": actual,
                            "validation_objective": objective,
                        }
                        trials[op.key].append(entry)
                        # On ties prefer fewer layers and fewer positions, then stable first config.
                        rank = (*objective, -len(group), -actual)
                        if best is None or rank > best:
                            best, selected[op.key] = rank, entry
        config = selected[op.key]
        progress(f"Test {op.key}: {config}")
        scores, actual = score(test, op, config)
        modes = {
            "base": baseline_test,
            "explicit_prompt": evaluate(lm, test, {}, (op,)),
            "selected": scores,
        }
        modes["deepest_only"], _ = score(test, op, {**config, "layers": [max(config["layers"])]})
        # Every member evaluated, to distinguish a joint effect from one strong site.
        for member in config["layers"]:
            modes[f"single_{member}"], _ = score(test, op, {**config, "layers": [member]})
        modes["self_patch"], _ = score(test, op, config, self_patch=True)
        self_error = float((modes["self_patch"] - baseline_test).abs().max())
        if not torch.allclose(modes["self_patch"], baseline_test, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Self-patch sanity check failed")
        tests[op.key] = {
            name: measurement(value, test, (op,), baseline_test) for name, value in modes.items()
        }
        selected[op.key]["test_actual_window"] = actual
        selected[op.key]["self_patch_max_score_difference"] = self_error
        cache.clear()
    restored = evaluate(lm, test, {})
    error = float((restored - baseline_test).abs().max())
    if not torch.allclose(restored, baseline_test, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Localization rollback failed")
    return {
        "experiment": "donor_localization_v1",
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "candidate_tokens": {
            str(v): lm.tokenizer.encode(f" {v}", add_special_tokens=False) for v in (0, 1)
        },
        "common_candidate_prefix": shared,
        "layer_sets": [list(g) for g in groups],
        "windows": windows,
        "boundaries": boundaries,
        "validation_trials": trials,
        "selected": selected,
        "test": tests,
        "rollback": {"max_score_difference": error},
        "limitations": [
            "Privileged paired donor prompts contain the intervention; this is localization, not learned generalization.",
            "Decision boundary includes only the shared candidate prefix, never the distinguishing answer token.",
            "Token windows are restricted to the identical donor/recipient suffix; actual sizes are recorded.",
            "Later complete block replacements may overwrite earlier effects; group success alone is not synergy.",
            "Default search covers singletons, contiguous triples and all layers, not all subsets.",
            "MLP and block sites are tested separately; only three primitive interventions are localized.",
            "One validation wording and one test wording on the same Boolean support; no significance claim.",
        ],
    }


def save_localization(root, report):
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-localize-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Privileged donor localization",
        "",
        "| Operation | Method | Accuracy | Pair correct | All nodes correct |",
        "|---|---|---:|---:|---:|",
    ]
    for op, methods in report["test"].items():
        for method, m in methods.items():
            c = m["consistency"]
            lines.append(
                f"| {op} | {method} | {m['overall']['accuracy']:.3f} | {c['relay_lamp_correct']['accuracy']:.3f} | {c['all_nodes_correct']['accuracy']:.3f} |"
            )
    lines += [
        "",
        "## Selected on validation",
        "",
        "```json",
        json.dumps(report["selected"], indent=2),
        "```",
        "",
        "## Limits",
        "",
    ] + ["- " + s for s in report["limitations"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
