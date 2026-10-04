"""LoReFT + locality, matched donor-regression baselines, graph-shift evaluation."""

import json
import random
from datetime import UTC, datetime
from uuid import uuid4

import torch
from safetensors.torch import save_file

from .causal_tasks import Scheme, circuit_questions, metrics, selection
from .conditional import fit_map, fit_pca_map, predict
from .experiment import SCENARIOS, resolve_sequence
from .localization import capture_tail, common_prefix, discover_sites
from .reft import LoReFT, frozen_model, intervention_scores, task_locality_loss
from .world import OPERATORS


def block_site(lm, layer):
    catalog = discover_sites(lm)
    if layer not in catalog:
        raise ValueError(f"Unknown block {layer}; available: {sorted(catalog)}")
    return catalog[layer]["block"]


@torch.no_grad()
def capture_states(lm, prompts, site, prefix):
    return torch.cat(
        [
            capture_tail(lm, prompts[i : i + 2], [site], 1, prefix)[site][:, 0]
            for i in range(0, len(prompts), 2)
        ]
    )


@torch.no_grad()
def evaluate(lm, samples, site, prefix, transform=lambda h: h, sequence=()):
    return torch.cat(
        [
            intervention_scores(
                lm, [q.prompt(sequence) for q in samples[i : i + 2]], site, prefix, transform
            ).cpu()
            for i in range(0, len(samples), 2)
        ]
    )


def train_batches(samples, op, steps, seed):
    """One affected + one unaffected query; alternate changed/stable affected pools."""
    rng = random.Random(seed)
    affected = [i for i, q in enumerate(samples) if q.node in q.world.affected(op)]
    local = [i for i in range(len(samples)) if i not in affected]
    changed = [i for i in affected if samples[i].answer((op,)) != samples[i].answer()]
    stable = [i for i in affected if i not in changed]
    pools = [changed or affected, stable or affected, local]
    for pool in pools:
        rng.shuffle(pool)
    return [
        (pools[t % 2][(t // 2) % len(pools[t % 2])], local[t % len(local)]) for t in range(steps)
    ]


def train_loreft(lm, cfg, samples, baseline, site, prefix, op, hidden_size, locality, progress):
    model = LoReFT(hidden_size, cfg.rank, cfg.seed).to(lm.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    batches = train_batches(samples, op, cfg.steps, cfg.seed)
    losses = []
    for step, indices in enumerate(batches):
        batch = [samples[i] for i in indices]
        optimizer.zero_grad(set_to_none=True)
        scores = intervention_scores(lm, [q.prompt() for q in batch], site, prefix, model)
        labels = torch.tensor([q.answer((op,)) for q in batch], device=lm.device)
        affected = torch.tensor([q.node in q.world.affected(op) for q in batch], device=lm.device)
        loss, parts = task_locality_loss(
            scores, baseline[list(indices)], labels, affected, locality
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LoReFT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"step": step + 1, "loss": float(loss.detach()), **parts})
        if step == 0 or (step + 1) % 10 == 0:
            progress(f"LoReFT {op.key} locality={locality}: {step + 1}/{cfg.steps} {parts}")
    return model.eval(), {
        "locality_weight": locality,
        "losses": losses,
        "question_keys": [[samples[i].key for i in b] for b in batches],
        "unique_questions": len({i for b in batches for i in b}),
    }


def run_reft_suite(
    lm,
    cfg,
    *,
    layer=12,
    strengths=None,
    pca_components=None,
    preservation_weight=1.0,
    progress=lambda _: None,
):
    strengths = [0, 0.5, 1, 2] if strengths is None else strengths
    if not strengths or any(not torch.isfinite(torch.tensor(a)) for a in strengths):
        raise ValueError("Provide finite strengths")
    if not torch.isfinite(torch.tensor(preservation_weight)) or preservation_weight < 0:
        raise ValueError("preservation_weight must be finite and nonnegative")
    strengths = sorted({0.0, *strengths}, key=lambda a: (abs(a), a))
    pca_components = [cfg.rank] if pca_components is None else pca_components
    if any(type(k) is not int or k < 1 for k in pca_components):
        raise ValueError("PCA ranks must be positive integers")
    site, prefix = block_site(lm, layer), common_prefix(lm)
    train, val = circuit_questions("train"), circuit_questions("validation")
    methods = (
        "constant",
        "ridge",
        *(f"pca_{k}" for k in sorted(set(pca_components))),
        "loreft_task",
        "loreft_locality",
    )
    fitted, tensors, training, trials, selected = {}, {}, {}, {}, {}
    with frozen_model(lm):
        base_train, base_val = evaluate(lm, train, site, prefix), evaluate(lm, val, site, prefix)
        x = capture_states(lm, [q.prompt() for q in train], site, prefix)
        for op in OPERATORS:
            progress(f"Training and selecting {op.key}")
            donor = capture_states(lm, [q.prompt((op,)) for q in train], site, prefix)
            delta = donor - x
            fitted[op.key] = {"ridge": fit_map(x, delta), "constant": delta.mean(0)}
            for k in sorted(set(pca_components)):
                fitted[op.key][f"pca_{k}"] = fit_pca_map(x, delta, k)
            training[op.key] = {}
            for method, locality in (
                ("loreft_task", 0.0),
                ("loreft_locality", cfg.locality_weight),
            ):
                model, log = train_loreft(
                    lm, cfg, train, base_train, site, prefix, op, x.shape[1], locality, progress
                )
                fitted[op.key][method], training[op.key][method] = model, log
            selected[op.key], trials[op.key] = {}, {}
            for method in methods:
                model = fitted[op.key][method]
                fields = (
                    model.state_dict()
                    if isinstance(model, LoReFT)
                    else {"mean_delta": model}
                    if method == "constant"
                    else model
                )
                for key, tensor in fields.items():
                    tensors[f"{op.key}.{method}.{key}"] = tensor.detach().cpu().contiguous().clone()
                trials[op.key][method] = []
                for alpha in strengths:
                    scores = (
                        base_val
                        if alpha == 0
                        else evaluate(lm, val, site, prefix, make_transform(model, method, alpha))
                    )
                    m = metrics(scores, val, (op,), base_val)
                    entry = {
                        "alpha": alpha,
                        "objective": list(selection(m, preservation_weight)),
                        "all_nodes_correct": m["all_nodes_correct"],
                        "relay_lamp_correct": m["relay_lamp_correct"],
                        "protected_damage": m["protected_damage"],
                    }
                    trials[op.key][method].append(entry)
                    best = selected[op.key].get(method)
                    if best is None or entry["objective"] > best["objective"]:
                        selected[op.key][method] = entry
        tests, exact = {}, {}
        for scheme in Scheme:
            progress(f"Held-out evaluation: {scheme.value}")
            samples = circuit_questions("test", scheme)
            base = evaluate(lm, samples, site, prefix)
            tests[scheme.value], exact[scheme.value] = {}, {}
            for name, sequence in SCENARIOS.items():
                modes = {
                    "base": base,
                    "explicit_prompt": evaluate(lm, samples, site, prefix, sequence=sequence),
                }
                for method in methods:
                    transforms = [
                        make_transform(
                            fitted[op.key][method], method, selected[op.key][method]["alpha"]
                        )
                        for op in resolve_sequence(sequence)
                    ]

                    def compose(h, transforms=transforms):
                        for transform in transforms:
                            h = transform(h)
                        return h

                    modes[method] = evaluate(lm, samples, site, prefix, compose)
                tests[scheme.value][name] = {
                    m: metrics(s, samples, sequence, base) for m, s in modes.items()
                }
                donor = capture_states(lm, [q.prompt(sequence) for q in samples], site, prefix)
                rows = []
                for i in range(0, len(samples), 2):
                    values = donor[i : i + 2].repeat_interleave(2, 0).to(lm.device)
                    with torch.no_grad():
                        rows.append(
                            intervention_scores(
                                lm,
                                [q.prompt() for q in samples[i : i + 2]],
                                site,
                                prefix,
                                lambda h, values=values: values.to(h),
                            ).cpu()
                        )
                exact[scheme.value][name] = metrics(torch.cat(rows), samples, sequence, base)
        restored = evaluate(lm, train, site, prefix)
        difference = float((restored - base_train).abs().max())
        if not torch.allclose(restored, base_train, atol=1e-5, rtol=1e-5):
            raise RuntimeError("LoReFT rollback failed")
    return {
        "experiment": "loreft_causal_suite_v1",
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "seed": cfg.seed,
        "rank": cfg.rank,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "locality_weight": cfg.locality_weight,
        "preservation_weight": preservation_weight,
        "strengths": strengths,
        "site": {"path": site, "kind": "block", "boundary": "decision", "prefix": prefix},
        "train_scheme": Scheme.AND_COPY.value,
        "split": {
            "train": [q.key for q in train],
            "validation": [q.key for q in val],
            "test": {s.value: [q.key for q in circuit_questions("test", s)] for s in Scheme},
        },
        "methods": list(methods),
        "training": training,
        "selected": selected,
        "validation_trials": trials,
        "test": tests,
        "oracle_diagnostics": exact,
        "rollback": {"max_score_difference": difference},
        "sources": ["https://arxiv.org/abs/2404.03592", "https://arxiv.org/abs/2110.11309"],
        "limitations": [
            "Native LoReFT equation with QR basis and identity initialization; not a reproduction of paper benchmarks or the pyreft optimizer setup.",
            "CE and forward locality KL use normalized scores for only 0/1 candidates, not the full vocabulary distribution.",
            "Task labels supervise affected variables; locality preserves every unaffected query, including relay for lamp edits. Symbolic masks are used only in training.",
            "Locality is a soft training objective, not a guarantee. loreft_task is an identically initialized zero-locality ablation.",
            "Donor-regression baselines have different supervision from LoReFT; they share train worlds, site and validation alpha grid.",
            "Compositions sequentially transform the same hidden vector; they do not rerun the network between primitive operators.",
            "Revision uses external latest-write-wins resolution, not learned contradiction detection.",
            "New schemes are test-only, but all use Boolean variables and the same names. No arbitrary-text reasoning claim.",
            "The layer was chosen after earlier experiments. One seed, small truth tables; no statistical significance claim.",
            "Exact donors are privileged controls and do not participate in fitting or selection. Base model weights remain frozen.",
        ],
    }, tensors


def make_transform(model, method, alpha):
    def transform(h):
        if alpha == 0:
            return h
        if isinstance(model, LoReFT):
            return h + alpha * (model(h) - h)
        delta = model.to(h) if method == "constant" else predict(model, h.detach()).to(h)
        return h + alpha * delta

    return transform


def save_research(root, report, tensors):
    kind = "das" if report["experiment"].startswith("das_") else "reft"
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{kind}-" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    save_file(
        tensors,
        folder / "operators.safetensors",
        metadata={"experiment": report["experiment"], "site": json.dumps(report["site"])},
    )
    lines = ["# " + report["experiment"], ""]
    if kind == "reft":
        lines += [
            "| Scheme | Scenario | Method | Accuracy | All nodes | New protected errors |",
            "|---|---|---|---:|---:|---:|",
        ]
        for scheme, scenarios in report["test"].items():
            for name, methods in scenarios.items():
                for method, m in {
                    **methods,
                    "exact_donor": report["oracle_diagnostics"][scheme][name],
                }.items():
                    lines.append(
                        f"| {scheme} | {name} | {method} | {m['overall']['accuracy']:.3f} | {m['all_nodes_correct']:.3f} | {m['protected_damage']['damaged']} |"
                    )
        lines += [
            "",
            "## Selected on validation",
            "",
            "```json",
            json.dumps(report["selected"], indent=2),
            "```",
        ]
    else:
        lines += [
            "| Scheme | Variable | Method | IIA | Changed cases | Protected damage |",
            "|---|---|---|---:|---:|---:|",
        ]
        for scheme, nodes in report["test"].items():
            for node, methods in nodes.items():
                for method, m in methods.items():
                    lines.append(
                        f"| {scheme} | {node} | {method} | {m['interchange_accuracy']:.3f} | {m['changed_accuracy']:.3f} | {m['protected_damage']['damaged']} |"
                    )
    lines += ["", "## Limits", ""] + ["- " + s for s in report["limitations"]]
    lines += ["", "## Sources", ""] + ["- " + s for s in report["sources"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder
