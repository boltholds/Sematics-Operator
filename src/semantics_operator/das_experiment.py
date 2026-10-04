"""Low-rank DAS on ordered neutral base/source pairs and explicit SCM targets."""

import random

import torch
from torch.nn import functional as F

from .causal_tasks import Scheme, circuit_questions, interchange_pairs
from .localization import common_prefix
from .reft import DAS, frozen_model, intervention_scores
from .reft_experiment import block_site, capture_states, evaluate
from .world import Node


def pair_metrics(scores, pairs, natural, lookup):
    predictions = scores.argmax(-1).tolist()
    probabilities = scores.softmax(-1)[:, 1].tolist()
    natural_predictions = natural.argmax(-1).tolist()
    correct, changed, eligible, damage, natural_subset, records, worlds = [], [], [], [], [], [], {}
    for pair, prediction, p1 in zip(pairs, predictions, probabilities, strict=True):
        base_pred = natural_predictions[lookup[pair.base.key]]
        source_pred = natural_predictions[lookup[pair.source.key]]
        ok = prediction == pair.expected
        correct.append(ok)
        if pair.expected != pair.base.answer():
            changed.append(ok)
        if pair.base.node in ("source", "switch", "flag") and base_pred == pair.base.answer():
            eligible.append(True)
            damage.append(not ok)
        if base_pred == pair.base.answer() and source_pred == pair.source.answer():
            natural_subset.append(ok)
        worlds.setdefault((pair.base.world.name, pair.source.world.name), []).append(ok)
        records.append(
            {
                "base": pair.base.key,
                "source": pair.source.key,
                "variable": pair.node.value,
                "source_value": pair.intervention.value,
                "expected": pair.expected,
                "prediction": prediction,
                "p1": p1,
            }
        )
    return {
        "count": len(pairs),
        "interchange_accuracy": sum(correct) / len(correct),
        "changed_count": len(changed),
        "changed_accuracy": sum(changed) / len(changed) if changed else 0.0,
        "all_nodes_correct": sum(all(v) for v in worlds.values()) / len(worlds),
        "protected_damage": {
            "eligible": len(eligible),
            "damaged": sum(damage),
            "rate": sum(damage) / len(eligible) if eligible else 0.0,
        },
        "both_natural_correct": {
            "count": len(natural_subset),
            "accuracy": sum(natural_subset) / len(natural_subset) if natural_subset else None,
        },
        "records": records,
    }


def score_pairs(lm, pairs, state, lookup, site, prefix, transform):
    rows = []
    for start in range(0, len(pairs), 2):
        batch = pairs[start : start + 2]
        source = state[[lookup[p.source.key] for p in batch]].repeat_interleave(2, 0).to(lm.device)
        scores = intervention_scores(
            lm,
            [p.base.prompt() for p in batch],
            site,
            prefix,
            lambda h, source=source: transform(h, source),
        )
        rows.append(scores.detach().cpu())
    return torch.cat(rows)


def run_das(lm, cfg, *, layer=12, progress=lambda _: None):
    site, prefix = block_site(lm, layer), common_prefix(lm)
    train = circuit_questions("train")
    pairs = interchange_pairs(train)
    lookup = {q.key: i for i, q in enumerate(train)}
    fitted, random_controls, logs, tensors = {}, {}, {}, {}
    with frozen_model(lm):
        natural_train = evaluate(lm, train, site, prefix)
        state = capture_states(lm, [q.prompt() for q in train], site, prefix)
        for node_index, node in enumerate((Node.RELAY, Node.LAMP)):
            model = DAS(state.shape[1], cfg.rank, cfg.seed + node_index).to(lm.device)
            random_controls[node] = DAS(state.shape[1], cfg.rank, cfg.seed + node_index).to(
                lm.device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            node_pairs = [p for p in pairs if p.node == node]
            pools = [
                [p for p in node_pairs if (p.expected != p.base.answer()) == changed]
                for changed in (True, False)
            ]
            rng = random.Random(cfg.seed + node_index)
            for pool in pools:
                rng.shuffle(pool)
            losses, used = [], set()
            for step in range(cfg.steps):
                batch = [pool[step % len(pool)] for pool in pools]
                used.update((p.base.key, p.source.key) for p in batch)
                source = (
                    state[[lookup[p.source.key] for p in batch]]
                    .repeat_interleave(2, 0)
                    .to(lm.device)
                )
                optimizer.zero_grad(set_to_none=True)
                scores = intervention_scores(
                    lm,
                    [p.base.prompt() for p in batch],
                    site,
                    prefix,
                    lambda h, model=model, source=source: model(h, source),
                )
                labels = torch.tensor([p.expected for p in batch], device=lm.device)
                loss = F.cross_entropy(scores, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite DAS loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if step == 0 or (step + 1) % 10 == 0:
                    progress(f"DAS {node}: {step + 1}/{cfg.steps}, loss={losses[-1]:.5f}")
            fitted[node], logs[node.value] = (
                model.eval(),
                {
                    "losses": losses,
                    "unique_pairs": len(used),
                    "available_pairs": len(node_pairs),
                    "pair_keys": sorted(used),
                },
            )
            tensors[f"{node.value}.raw_basis"] = model.raw_basis.detach().cpu().clone()
            tensors[f"{node.value}.orthonormal_basis"] = (
                model.basis().detach().cpu().contiguous().clone()
            )

        def assess(samples):
            lookup = {q.key: i for i, q in enumerate(samples)}
            state = capture_states(lm, [q.prompt() for q in samples], site, prefix)
            natural = evaluate(lm, samples, site, prefix)
            pairs = interchange_pairs(samples)
            result = {}
            for node in (Node.RELAY, Node.LAMP):
                node_pairs = [p for p in pairs if p.node == node]
                modes = {"base": natural[[lookup[p.base.key] for p in node_pairs]]}
                with torch.no_grad():
                    for method, transform in (
                        ("das", fitted[node]),
                        ("random_subspace", random_controls[node]),
                        ("full_donor", lambda h, source: source.to(h)),
                    ):
                        modes[method] = score_pairs(
                            lm, node_pairs, state, lookup, site, prefix, transform
                        )
                result[node.value] = {
                    name: pair_metrics(s, node_pairs, natural, lookup) for name, s in modes.items()
                }
            return result

        progress("DAS validation: report only, no model/strength selection")
        validation = assess(circuit_questions("validation"))
        tests = {}
        for scheme in Scheme:
            progress(f"DAS test: {scheme.value}")
            tests[scheme.value] = assess(circuit_questions("test", scheme))
        restored = evaluate(lm, train, site, prefix)
        difference = float((restored - natural_train).abs().max())
        if not torch.allclose(restored, natural_train, atol=1e-5, rtol=1e-5):
            raise RuntimeError("DAS rollback failed")
    return {
        "experiment": "das_causal_alignment_v1",
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "seed": cfg.seed,
        "rank": cfg.rank,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "train_scheme": Scheme.AND_COPY.value,
        "site": {"path": site, "kind": "block", "boundary": "decision", "prefix": prefix},
        "training": logs,
        "validation": validation,
        "test": tests,
        "rollback": {"max_score_difference": difference},
        "sources": [
            "https://proceedings.mlr.press/v236/geiger24a.html",
            "https://arxiv.org/abs/2404.03592",
        ],
        "limitations": [
            "Low-rank DAS with QR basis on a frozen model. Train one shared subspace per variable for both binary values.",
            "Train targets come from the explicit base SCM with the source variable substituted. Source prompts are neutral, without override instructions or answer digits.",
            "This diagnostic uses donor activations at test time; it is not a donor-free operator.",
            "Validation is reported after fixed-step training; no selection or training uses validation or shifted schemes.",
            "All ordered 8x8 world pairs and query roles are evaluated, including self and unchanged-value swaps. These observations are correlated, not independent trials.",
            "Cross-entropy over normalized 0/1 candidates; no full-vocabulary loss. Both-naturally-correct subset is reported because the base model may not implement the SCM.",
            "A query-conditioned late answer state can encode the answer rather than an intermediate causal variable; IIA alone cannot establish faithful localization.",
            "No claim of reproducing original DAS benchmarks. One seed and a small Boolean family.",
        ],
    }, tensors
