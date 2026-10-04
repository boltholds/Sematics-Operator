"""Matched constant versus state-conditioned activation transitions.

Fit only on train pairs. The conditional map is an affine kernel-ridge predictor
in the span of centered training states, avoiding a hidden_dim-squared matrix.
"""

import torch

from .experiment import SCENARIOS, capture_probes, resolve_sequence
from .localization import capture_tail, common_prefix, discover_sites, patched_scores
from .steering import evaluate, measurement, selection_score, steered_scores, validation_questions
from .world import OPERATORS, Node, questions


def protected_damage(scores, samples, baseline):
    """New errors only, on upstream/independent nodes correct before editing."""
    predicted, original = scores.argmax(-1).tolist(), baseline.argmax(-1).tolist()
    by_node = {}
    for node in (Node.SOURCE, Node.SWITCH, Node.FLAG):
        eligible = [
            i for i, q in enumerate(samples) if q.node == node and original[i] == q.answer()
        ]
        damaged = sum(predicted[i] != samples[i].answer() for i in eligible)
        by_node[node.value] = {"eligible": len(eligible), "damaged": damaged}
    count = sum(m["eligible"] for m in by_node.values())
    damage = sum(m["damaged"] for m in by_node.values())
    return {
        "eligible": count,
        "damaged": damage,
        "rate": damage / count if count else 0.0,
        "by_node": by_node,
    }


def penalized_selection(scores, samples, intervention, baseline, weight):
    joint, pair, overall = selection_score(scores, samples, intervention, baseline)
    damage = protected_damage(scores, samples, baseline)["rate"]
    return (joint - weight * damage, pair, overall)


@torch.no_grad()
def fit_map(states, deltas, ridge=0.1):
    if states.ndim != 2 or states.shape != deltas.shape or len(states) < 2:
        raise ValueError("Expected matching [samples, hidden] states and deltas")
    if not torch.isfinite(torch.tensor(ridge)) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if not torch.isfinite(states).all() or not torch.isfinite(deltas).all():
        raise ValueError("Non-finite training activations")
    x, y = states.double().cpu(), deltas.double().cpu()
    center, mean = x.mean(0), y.mean(0)
    centered = x - center
    scale = centered.square().sum(1).mean().sqrt().clamp_min(1e-8)
    features = centered / scale
    kernel = features @ features.T
    coefficients = torch.linalg.solve(kernel + ridge * torch.eye(len(x)), y - mean)
    return {
        "center": center,
        "scale": scale,
        "features": features,
        "coefficients": coefficients,
        "mean_delta": mean,
    }


@torch.no_grad()
def fit_pca_map(states, deltas, components, ridge=0.1):
    if not isinstance(components, int) or components < 1:
        raise ValueError("PCA components must be positive integers")
    full = fit_map(states, deltas, ridge)
    centered = deltas.double().cpu() - full["mean_delta"]
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    tolerance = singular.max() * max(centered.shape) * torch.finfo(torch.float64).eps
    rank = int((singular > tolerance).sum())
    k = min(components, rank, len(states) - 1)
    basis = vh[:k].contiguous()
    total = singular.square().sum()
    energy = singular[:k].square().sum() / total if total > 0 else total
    return {
        **full,
        "coefficients": full["coefficients"] @ basis.T,
        "basis": basis,
        "explained_variance_ratio": energy,
    }


def predict(model, states):
    x = (states.double().cpu() - model["center"]) / model["scale"]
    residual = (x @ model["features"].T) @ model["coefficients"]
    if "basis" in model:
        residual = residual @ model["basis"]
    return (model["mean_delta"] + residual).float()


def row_scores(lm, samples, edits, *, replace=False, prefix=None):
    """Each question gets its own delta, duplicated identically for its two candidates."""
    if prefix is not None:
        return torch.cat(
            [
                patched_scores(
                    lm,
                    [q.prompt() for q in samples[i : i + 2]],
                    {site: delta[i : i + 2, None, :] for site, delta in edits.items()},
                    list(edits),
                    1,
                    prefix,
                    replace=replace,
                )
                for i in range(0, len(samples), 2)
            ]
        )
    return torch.cat(
        [
            steered_scores(
                lm,
                [q.prompt() for q in samples[i : i + 2]],
                {
                    layer: delta[i : i + 2].repeat_interleave(2, dim=0)
                    for layer, delta in edits.items()
                },
                replace=replace,
            )
            for i in range(0, len(samples), 2)
        ]
    )


@torch.no_grad()
def run_conditional(
    lm,
    cfg,
    *,
    layers=None,
    strengths=None,
    ridge=0.1,
    pca_components=(),
    site_kind="mlp",
    boundary="prompt",
    preservation_weight=0.0,
    progress=lambda _: None,
):
    # Hold the injection site fixed by default; optional identical layer search for all methods.
    if site_kind not in ("mlp", "block") or boundary not in ("prompt", "decision"):
        raise ValueError("Use mlp/block sites and prompt/decision boundary")
    if not torch.isfinite(torch.tensor(preservation_weight)) or preservation_weight < 0:
        raise ValueError("preservation_weight must be finite and nonnegative")
    if layers is None and site_kind == "block":
        raise ValueError("Specify block indices with --layers, e.g. --layers 12")
    layers = [lm.choose_target(cfg.target_module)] if layers is None else layers
    if not layers:
        raise ValueError("At least one layer is required")
    if site_kind == "block":
        catalog = discover_sites(lm)
        blocks = {str(i): sites["block"] for i, sites in catalog.items()}
        layers = [blocks.get(str(layer), str(layer)) for layer in layers]
        if any(layer not in blocks.values() for layer in layers):
            raise ValueError("Unknown decoder block; use an existing index or full block path")
        layers = list(dict.fromkeys(layers))
    else:
        layers = list(dict.fromkeys(lm.choose_target(layer) for layer in layers))
    prefix = common_prefix(lm) if boundary == "decision" else []
    extended = site_kind != "mlp" or boundary != "prompt" or preservation_weight != 0

    def capture(prompts, layer):
        if not extended:
            return capture_probes(lm, prompts, layer)
        return torch.cat(
            [
                capture_tail(lm, prompts[i : i + 2], [layer], 1, prefix)[layer][:, 0]
                for i in range(0, len(prompts), 2)
            ]
        )

    def score_rows(qs, edits, *, replace=False):
        return row_scores(lm, qs, edits, replace=replace, prefix=prefix if extended else None)

    def measure(scores, qs, sequence, baseline):
        result = measurement(scores, qs, sequence, baseline)
        result["protected_damage"] = protected_damage(scores, qs, baseline)
        return result

    strengths = [0.0, 0.5, 1.0, 2.0] if strengths is None else strengths
    if not strengths or any(not torch.isfinite(torch.tensor(a)) for a in strengths):
        raise ValueError("Provide finite strengths")
    strengths = sorted({0.0, *strengths}, key=lambda a: (abs(a), a))
    train, validation, test = questions("train"), validation_questions(), questions("test")
    base_validation, base_test = evaluate(lm, validation, {}), evaluate(lm, test, {})
    samples = {"train": train, "validation": validation, "test": test}
    states = {split: {} for split in samples}
    models, tensors, fit_metrics = {}, {}, {}
    if any(not isinstance(k, int) or k < 1 for k in pca_components):
        raise ValueError("PCA components must be positive integers")
    pca_components = sorted(set(pca_components))
    methods = ("constant", "conditional", "shuffled", *(f"pca_{k}" for k in pca_components))
    trials = {op.key: {method: [] for method in methods} for op in OPERATORS}
    selected = {op.key: {} for op in OPERATORS}
    for layer in layers:
        progress(f"Collecting neutral states: {layer}")
        for split, qs in samples.items():
            states[split][layer] = capture([q.prompt() for q in qs], layer)
        for op_index, op in enumerate(OPERATORS):
            progress(f"Fitting transitions: {op.key}, {layer}")
            x = states["train"][layer]
            target = capture([q.prompt((op,)) for q in train], layer)
            delta = target - x
            generator = torch.Generator().manual_seed(cfg.seed + op_index)
            permutation = torch.randperm(len(x), generator=generator)
            fitted = {
                "conditional": fit_map(x, delta, ridge),
                "shuffled": fit_map(x, delta[permutation], ridge),
            }
            fitted.update({f"pca_{k}": fit_pca_map(x, delta, k, ridge) for k in pca_components})
            for method in methods:
                model = fitted["conditional"] if method == "constant" else fitted[method]
                models[op.key, layer, method] = model
                artifact_prefix = f"{op.key}.{layer}.{method}"
                fields = {"mean_delta": model["mean_delta"]} if method == "constant" else model
                for key, tensor in fields.items():
                    tensors[f"{artifact_prefix}.{key}"] = tensor.contiguous().clone()
                fitted_delta = (
                    model["mean_delta"].float().expand_as(x)
                    if method == "constant"
                    else predict(model, x)
                )
                fit_metrics[artifact_prefix] = {
                    "train_delta_mse_against_true_pairs": float(
                        (fitted_delta - delta).square().mean()
                    )
                }
                if "basis" in model:
                    fit_metrics[artifact_prefix].update(
                        {
                            "effective_components": model["basis"].shape[0],
                            "train_explained_variance_ratio": float(
                                model["explained_variance_ratio"]
                            ),
                        }
                    )
                val_delta = (
                    model["mean_delta"].float().expand_as(states["validation"][layer])
                    if method == "constant"
                    else predict(model, states["validation"][layer])
                )
                for alpha in strengths:
                    scores = (
                        base_validation
                        if alpha == 0
                        else score_rows(validation, {layer: alpha * val_delta})
                    )
                    objective = list(
                        penalized_selection(
                            scores, validation, op, base_validation, preservation_weight
                        )
                    )
                    entry = {
                        "layer": layer,
                        "alpha": alpha,
                        "validation_objective": objective,
                        "unpenalized_objective": list(
                            selection_score(scores, validation, op, base_validation)
                        ),
                        "protected_damage": protected_damage(scores, validation, base_validation),
                    }
                    trials[op.key][method].append(entry)
                    best = selected[op.key].get(method)
                    if best is None or objective > best["validation_objective"]:
                        selected[op.key][method] = entry
            progress(f"Selected so far: {op.key}: {selected[op.key]}")

    if pca_components:
        for op in OPERATORS:
            variant = max(
                (f"pca_{k}" for k in pca_components),
                key=lambda m: selected[op.key][m]["validation_objective"],
            )
            selected[op.key]["pca_selected"] = {**selected[op.key][variant], "variant": variant}
        methods = (*methods, "pca_selected")

    def edits_for(sequence, method):
        edits = {}
        for op in resolve_sequence(sequence):
            choice = selected[op.key][method]
            layer = choice["layer"]
            model = models[op.key, layer, choice.get("variant", method)]
            x = states["test"][layer]
            delta = (
                model["mean_delta"].float().expand_as(x)
                if method == "constant"
                else predict(model, x)
            )
            edits[layer] = edits.get(layer, torch.zeros_like(delta)) + choice["alpha"] * delta
        return edits

    scenarios = {}
    for name, sequence in SCENARIOS.items():
        progress(f"Test: {name}")
        scores = {
            "base": base_test,
            **{method: score_rows(test, edits_for(sequence, method)) for method in methods},
            "explicit_prompt": evaluate(lm, test, {}, sequence),
        }
        scenarios[name] = {
            method: measure(value, test, sequence, base_test) for method, value in scores.items()
        }
    # Privileged diagnostic: paired donor prompts are allowed ONLY here, after selection.
    oracle = {}
    if pca_components or extended:
        for layer in layers:
            oracle[layer] = {}
            for name, sequence in SCENARIOS.items():
                progress(f"Exact donor diagnostic: {layer}, {name}")
                donor = capture([q.prompt(sequence) for q in test], layer)
                scores = score_rows(test, {layer: donor}, replace=True)
                oracle[layer][name] = measure(scores, test, sequence, base_test)
    self_patch = {}
    if extended:
        for layer in layers:
            scores = score_rows(test, {layer: states["test"][layer]}, replace=True)
            self_patch[layer] = float((scores - base_test).abs().max())
            if not torch.allclose(scores, base_test, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Comparison self-patch sanity check failed")
    restored = evaluate(lm, test, {})
    difference = float((restored - base_test).abs().max())
    if not torch.allclose(restored, base_test, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Conditional steering rollback failed")
    report = {
        "experiment": "state_conditioned_activation_v3"
        if extended
        else (
            "state_conditioned_activation_v2"
            if pca_components
            else "state_conditioned_activation_v1"
        ),
        "pca_components": pca_components,
        "site": {
            "kind": site_kind,
            "boundary": boundary,
            "window": 1,
            "common_candidate_prefix": prefix,
        },
        "preservation_weight": preservation_weight,
        "self_patch_max_score_difference": self_patch,
        "oracle_diagnostics": oracle,
        "model": {
            "profile": cfg.profile,
            "path": str(cfg.model_path),
            "device": str(lm.device),
            "dtype": str(next(lm.model.parameters()).dtype),
        },
        "seed": cfg.seed,
        "ridge": ridge,
        "layers": layers,
        "strengths": strengths,
        "split": {
            "train": [q.key for q in train],
            "validation": [q.key for q in validation],
            "test": [q.key for q in test],
            "shift": "Distinct names/templates, same truth-table support.",
        },
        "fit": "Delta = mean_delta + ((z-center)/scale @ features.T) @ coefficients. Ridge fit on train only.",
        "selection": "Identical validation grid per method. Lexicographic (all-nodes accuracy - preservation_weight * protected damage rate), pair, overall. Zero included. Damage counts new errors on previously correct source/switch/flag answers; zero if no eligible answers. Penalty applies to validation selection, not ridge training.",
        "selected": selected,
        "validation_trials": trials,
        "fit_metrics": fit_metrics,
        "scenarios": scenarios,
        "rollback": {"max_score_difference": difference},
        "limitations": [
            "Conditional map has more parameters than constant baseline; shared data and selection budget do not imply equal capacity.",
            "Targets are prompted activations, not guaranteed correct semantic states; wording and answer bias can confound them.",
            "Learned methods see neutral test states only. Oracle diagnostics use paired intervention prompts and are not a deployable operator.",
            "PCA is fit on centered train deltas only. Explained variance is not semantic correctness.",
            "Each fixed PCA rank has the same selection grid; pca_selected searches more candidates across ranks.",
            "Exact donor replacement covers one selected module output at one position, not the entire model state.",
            "Composition sums deltas predicted from unedited states; it is not recursive state evolution.",
            "One seed and one shuffled-pair control; no significance claim.",
            "Activation edits only; no weight edits. Decision position includes only the shared candidate prefix, never the distinguishing answer token.",
            "Choosing this site followed earlier test inspection; this is an exploratory follow-up, not a fresh held-out confirmation.",
        ],
    }
    return report, tensors
