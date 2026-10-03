"""Contrastive activation experiment; all selection uses validation, never test."""

from dataclasses import dataclass

import torch

from .experiment import SCENARIOS, resolve_sequence, summarize
from .world import OPERATORS, Node, Question, World, questions


@dataclass(frozen=True)
class ValidationQuestion(Question):
    def prompt(self, interventions=()):
        w = self.world
        text = (
            f"In system {w.name}, input source has value {w.source} and switch has value {w.switch}. "
            f"Independent flag has value {w.flag}. Compute relay by Boolean AND of source and switch. "
            "Set lamp equal to relay. "
        )
        if interventions:
            text += (
                "Replace the indicated assignments: "
                + "; ".join(
                    f"{op.node.value} := {op.value}" for op in resolve_sequence(interventions)
                )
                + ". "
            )
        return text + f"Return the value of {self.node.value}, either 0 or 1.\nAnswer:"


def validation_questions():
    return tuple(
        ValidationQuestion(
            World(
                q.world.name.replace("training", "validation"),
                q.world.source,
                q.world.switch,
                q.world.flag,
            ),
            q.node,
        )
        for q in questions("train")
    )


@torch.no_grad()
def steered_scores(lm, prompts, edits, *, replace=False):
    """Edit only the last prompt token, identically for both answer candidates.

    No candidate tokens enter extraction or the intervention-position choice.
    This is a last-prompt MLP-output adaptation, not response-average persona steering.
    """
    positions = [len(lm._prompt_ids(p)) - 1 for p in prompts for _ in (0, 1)]
    handles = []
    try:
        for target, vector in edits.items():

            def hook(module, args, output, vector=vector):
                result = output.clone()
                rows = torch.arange(len(positions), device=output.device)
                change = vector.to(device=output.device, dtype=output.dtype)
                if replace:
                    result[rows, positions] = change
                else:
                    result[rows, positions] += change
                return result

            handles.append(lm.model.get_submodule(target).register_forward_hook(hook))
        return lm.scores(prompts).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()


def evaluate(lm, samples, edits, interventions=()):
    return torch.cat(
        [
            steered_scores(lm, [q.prompt(interventions) for q in samples[i : i + 2]], edits)
            for i in range(0, len(samples), 2)
        ]
    )


def consistency(scores, samples, interventions):
    worlds = {}
    for q, prediction in zip(samples, scores.argmax(-1).tolist(), strict=True):
        worlds.setdefault(q.world.name, (q.world, {}))[1][q.node] = prediction
    overrides = {op.node: op.value for op in interventions}
    pair, complete, relation = [], [], []
    for world, predicted in worlds.values():
        expected = world.values(interventions)
        pair.append(all(predicted[n] == expected[n] for n in (Node.RELAY, Node.LAMP)))
        complete.append(all(predicted[n] == expected[n] for n in Node))
        relation.append(predicted[Node.LAMP] == overrides.get(Node.LAMP, predicted[Node.RELAY]))

    def metric(values):
        return {"count": len(values), "accuracy": sum(values) / len(values)}

    return {
        "relay_lamp_correct": metric(pair),
        "all_nodes_correct": metric(complete),
        "lamp_mechanism_satisfied": metric(relation),
    }


def measurement(scores, samples, sequence, baseline):
    result = summarize(scores, samples, sequence, baseline)
    result["consistency"] = consistency(scores, samples, sequence)
    return result


def selection_score(scores, samples, intervention, baseline):
    metrics = measurement(scores, samples, (intervention,), baseline)
    # Joint world correctness avoids rewarding always-1 solutions or pair-only collapse.
    return (
        metrics["consistency"]["all_nodes_correct"]["accuracy"],
        metrics["consistency"]["relay_lamp_correct"]["accuracy"],
        metrics["overall"]["accuracy"],
    )


@torch.no_grad()
def run_steering(lm, cfg, *, layers=None, strengths=None, progress=lambda _: None):
    if layers is None:
        layers = [
            n
            for n in lm.linear_modules()
            if n.endswith(("down_proj", "fc2", "dense_4h_to_h", "feed_forward.w2"))
        ]
    strengths = [0.0, 0.5, 1.0, 2.0] if strengths is None else strengths
    if not layers or not strengths or any(not torch.isfinite(torch.tensor(a)) for a in strengths):
        raise ValueError("Provide recognized MLP layers and finite steering strengths")
    layers = list(dict.fromkeys(lm.choose_target(n) for n in layers))
    strengths = sorted({0.0, *strengths}, key=lambda a: (abs(a), a))
    train, validation, test = questions("train"), validation_questions(), questions("test")
    base_validation = evaluate(lm, validation, {})
    base_test = evaluate(lm, test, {})
    vectors, selected, trials, controls = {}, {}, {}, {}
    for op_index, op in enumerate(OPERATORS):
        best_score = None
        trials[op.key] = []
        for layer in layers:
            progress(f"Extracting {op.key}: {layer}")
            differences = []
            for start in range(0, len(train), 2):
                batch = train[start : start + 2]
                positive = lm.representations([q.prompt((op,)) for q in batch], layer)
                neutral = lm.representations([q.prompt() for q in batch], layer)
                differences.append(positive - neutral)
            vector = torch.cat(differences).mean(0)
            if not torch.isfinite(vector).all():
                raise FloatingPointError("Non-finite extracted vector")
            vectors[f"{op.key}.{layer}"] = vector
            for alpha in strengths:
                scores = evaluate(lm, validation, {layer: alpha * vector})
                metric = selection_score(scores, validation, op, base_validation)
                trials[op.key].append({"layer": layer, "alpha": alpha, "objective": list(metric)})
                if best_score is None or metric > best_score:
                    best_score = metric
                    selected[op.key] = {
                        "layer": layer,
                        "alpha": alpha,
                        "validation_objective": list(metric),
                    }
        choice = selected[op.key]
        vector = vectors[f"{op.key}.{choice['layer']}"]
        generator = torch.Generator().manual_seed(cfg.seed + op_index)
        noise = torch.randn(vector.shape, generator=generator)
        controls[op.key] = noise * (vector.norm() / noise.norm().clamp_min(1e-12))
        progress(f"Selected {op.key}: {choice}")

    def edits_for(sequence, mode):
        edits = {}
        for op in resolve_sequence(sequence):
            choice = selected[op.key]
            layer, alpha = choice["layer"], choice["alpha"]
            vector = controls[op.key] if mode == "random" else vectors[f"{op.key}.{layer}"]
            delta = vector * alpha * (-1 if mode == "opposite" else 1)
            edits[layer] = edits.get(layer, torch.zeros_like(delta)) + delta
        return edits

    scenarios = {}
    for name, sequence in SCENARIOS.items():
        progress(f"Test: {name}")
        modes = {
            "base": base_test,
            "steering": evaluate(lm, test, edits_for(sequence, "steering")),
            "random_norm_matched": evaluate(lm, test, edits_for(sequence, "random")),
            "opposite": evaluate(lm, test, edits_for(sequence, "opposite")),
            "explicit_prompt": evaluate(lm, test, {}, sequence),
        }
        scenarios[name] = {k: measurement(v, test, sequence, base_test) for k, v in modes.items()}
    restored = evaluate(lm, test, {})
    difference = float((restored - base_test).abs().max())
    if not torch.allclose(restored, base_test, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Steering rollback verification failed")
    key_sets = [{q.key for q in split} for split in (train, validation, test)]
    report = {
        "experiment": "contrastive_activation_v1",
        "model": {
            "profile": cfg.profile,
            "path": str(cfg.model_path),
            "device": str(lm.device),
            "dtype": str(next(lm.model.parameters()).dtype),
        },
        "seed": cfg.seed,
        "split": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "overlap": len(
                (key_sets[0] & key_sets[1])
                | (key_sets[0] & key_sets[2])
                | (key_sets[1] & key_sets[2])
            ),
            "shift": "Distinct names/templates; same Boolean truth-table support.",
        },
        "extraction": "Mean paired intervention-minus-neutral MLP outputs at last prompt token; no answers used.",
        "selection": "Validation lexicographic all-nodes correctness, pair correctness, overall accuracy; zero included.",
        "selected": selected,
        "validation_trials": trials,
        "scenarios": scenarios,
        "rollback": {"max_score_difference": difference},
        "limitations": [
            "Adaptation of contrastive activation steering, not a reproduction of response-average persona vectors.",
            "Explicit intervention wording and answer bias can confound extracted vectors.",
            "Random and opposite directions are controls, not proof of semantic specificity.",
            "No training of weights; future weight distillation requires a successful steering result.",
            "One seed, one random direction per primitive; no significance claim.",
            "Compositions sum primitive vectors and are never used for selection.",
        ],
    }
    return report, vectors


def save_steering(root, report, vectors):
    import json
    from datetime import UTC, datetime
    from uuid import uuid4

    from safetensors.torch import save_file

    conditional = report["experiment"].startswith("state_conditioned_activation_")
    kind = "compare" if conditional else "steering"
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{kind}-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    save_file(
        vectors,
        folder / ("transitions.safetensors" if conditional else "vectors.safetensors"),
        metadata={
            "position": "last prompt token",
            "space": "MLP output",
            "experiment": report["experiment"],
        },
    )
    lines = [
        "# " + report["experiment"],
        "",
        "| Scenario | Method | Overall | Changed | Preserved correct | Pair correct | All nodes correct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    def value(metric):
        return "n/a" if metric["accuracy"] is None else f"{metric['accuracy']:.3f}"

    for name, scenario in report["scenarios"].items():
        for method, metrics in scenario.items():
            cells = [
                metrics["overall"],
                metrics["changed"],
                metrics["preservation_of_correct_base_answers"],
                metrics["consistency"]["relay_lamp_correct"],
                metrics["consistency"]["all_nodes_correct"],
            ]
            lines.append(
                "| " + name + " | " + method + " | " + " | ".join(map(value, cells)) + " |"
            )
    lines += [
        "",
        "## Selected on validation",
        "",
        "```json",
        json.dumps(report["selected"], indent=2),
        "```",
        "",
        "## Limitations",
        "",
    ] + ["- " + s for s in report["limitations"]]
    if report.get("oracle_diagnostics"):
        lines += [
            "",
            "## Privileged exact-donor diagnostics",
            "",
            "Donor prompts contain the intervention; these are not held-out learned predictions.",
            "",
            "| Layer | Scenario | Pair correct | All nodes correct |",
            "|---|---|---:|---:|",
        ]
        for layer, scenarios in report["oracle_diagnostics"].items():
            for name, metrics in scenarios.items():
                c = metrics["consistency"]
                lines.append(
                    f"| {layer} | {name} | {value(c['relay_lamp_correct'])} | {value(c['all_nodes_correct'])} |"
                )
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
