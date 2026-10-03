"""Matched constant versus state-conditioned activation transitions.

Fit only on train pairs. The conditional map is an affine kernel-ridge predictor
in the span of centered training states, avoiding a hidden_dim-squared matrix.
"""

import torch

from .experiment import SCENARIOS, capture_probes, resolve_sequence
from .steering import evaluate, measurement, selection_score, steered_scores, validation_questions
from .world import OPERATORS, questions


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


def predict(model, states):
    x = (states.double().cpu() - model["center"]) / model["scale"]
    return (model["mean_delta"] + (x @ model["features"].T) @ model["coefficients"]).float()


def row_scores(lm, samples, edits):
    """Each question gets its own delta, duplicated identically for its two candidates."""
    return torch.cat(
        [
            steered_scores(
                lm,
                [q.prompt() for q in samples[i : i + 2]],
                {
                    layer: delta[i : i + 2].repeat_interleave(2, dim=0)
                    for layer, delta in edits.items()
                },
            )
            for i in range(0, len(samples), 2)
        ]
    )


@torch.no_grad()
def run_conditional(lm, cfg, *, layers=None, strengths=None, ridge=0.1, progress=lambda _: None):
    # Hold the injection site fixed by default; optional identical layer search for all methods.
    layers = [lm.choose_target(cfg.target_module)] if layers is None else layers
    if not layers:
        raise ValueError("At least one layer is required")
    layers = list(dict.fromkeys(lm.choose_target(layer) for layer in layers))
    strengths = [0.0, 0.5, 1.0, 2.0] if strengths is None else strengths
    if not strengths or any(not torch.isfinite(torch.tensor(a)) for a in strengths):
        raise ValueError("Provide finite strengths")
    strengths = sorted({0.0, *strengths}, key=lambda a: (abs(a), a))
    train, validation, test = questions("train"), validation_questions(), questions("test")
    base_validation, base_test = evaluate(lm, validation, {}), evaluate(lm, test, {})
    samples = {"train": train, "validation": validation, "test": test}
    states = {split: {} for split in samples}
    models, tensors, fit_metrics = {}, {}, {}
    methods = ("constant", "conditional", "shuffled")
    trials = {op.key: {method: [] for method in methods} for op in OPERATORS}
    selected = {op.key: {} for op in OPERATORS}
    for layer in layers:
        progress(f"Collecting neutral states: {layer}")
        for split, qs in samples.items():
            states[split][layer] = capture_probes(lm, [q.prompt() for q in qs], layer)
        for op_index, op in enumerate(OPERATORS):
            progress(f"Fitting transitions: {op.key}, {layer}")
            x = states["train"][layer]
            target = capture_probes(lm, [q.prompt((op,)) for q in train], layer)
            delta = target - x
            generator = torch.Generator().manual_seed(cfg.seed + op_index)
            permutation = torch.randperm(len(x), generator=generator)
            fitted = {
                "conditional": fit_map(x, delta, ridge),
                "shuffled": fit_map(x, delta[permutation], ridge),
            }
            for method in methods:
                model = fitted["conditional"] if method == "constant" else fitted[method]
                models[op.key, layer, method] = model
                prefix = f"{op.key}.{layer}.{method}"
                fields = {"mean_delta": model["mean_delta"]} if method == "constant" else model
                for key, tensor in fields.items():
                    tensors[f"{prefix}.{key}"] = tensor.contiguous().clone()
                fitted_delta = (
                    model["mean_delta"].float().expand_as(x)
                    if method == "constant"
                    else predict(model, x)
                )
                fit_metrics[prefix] = {
                    "train_delta_mse_against_true_pairs": float(
                        (fitted_delta - delta).square().mean()
                    )
                }
                val_delta = (
                    model["mean_delta"].float().expand_as(states["validation"][layer])
                    if method == "constant"
                    else predict(model, states["validation"][layer])
                )
                for alpha in strengths:
                    scores = row_scores(lm, validation, {layer: alpha * val_delta})
                    objective = list(selection_score(scores, validation, op, base_validation))
                    entry = {"layer": layer, "alpha": alpha, "validation_objective": objective}
                    trials[op.key][method].append(entry)
                    best = selected[op.key].get(method)
                    if best is None or objective > best["validation_objective"]:
                        selected[op.key][method] = entry
            progress(f"Selected so far: {op.key}: {selected[op.key]}")

    def edits_for(sequence, method):
        edits = {}
        for op in resolve_sequence(sequence):
            choice = selected[op.key][method]
            layer = choice["layer"]
            model = models[op.key, layer, method]
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
            **{method: row_scores(lm, test, edits_for(sequence, method)) for method in methods},
            "explicit_prompt": evaluate(lm, test, {}, sequence),
        }
        scenarios[name] = {
            method: measurement(value, test, sequence, base_test)
            for method, value in scores.items()
        }
    restored = evaluate(lm, test, {})
    difference = float((restored - base_test).abs().max())
    if not torch.allclose(restored, base_test, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Conditional steering rollback failed")
    report = {
        "experiment": "state_conditioned_activation_v1",
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
        "selection": "Identical validation grid per method. Lexicographic all-nodes, pair, overall correctness. Zero included.",
        "selected": selected,
        "validation_trials": trials,
        "fit_metrics": fit_metrics,
        "scenarios": scenarios,
        "rollback": {"max_score_difference": difference},
        "limitations": [
            "Conditional map has more parameters than constant baseline; shared data and selection budget do not imply equal capacity.",
            "Targets are prompted activations, not guaranteed correct semantic states; wording and answer bias can confound them.",
            "Test states contain only unedited prompts, never intervention prompts or answers.",
            "Composition sums deltas predicted from unedited states; it is not recursive state evolution.",
            "One seed and one shuffled-pair control; no significance claim.",
            "Same MLP last-prompt injection site as prior experiment; no weight edits.",
        ],
    }
    return report, tensors
