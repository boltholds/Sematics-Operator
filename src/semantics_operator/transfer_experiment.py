"""Train in one representation; freeze site, weights and strength before transfer."""

import math

import torch

from .answer_protocol import protocol_metadata
from .causal_tasks import metrics
from .full_vocab import LossMode, answer_sequences, reference_distributions
from .generation_evaluation import (
    attach_candidate_agreement,
    generate_questions,
    generation_metrics,
)
from .localization import discover_sites
from .positions import ReftPosition, validate_state_prefixes
from .reft import frozen_model
from .reft_experiment import block_site, capture_states, evaluate, make_transform, train_loreft
from .transfer_metrics import paired_metrics, transfer_agreement, transfer_selection
from .transfer_tasks import Family, Names, Representation, transfer_questions
from .world import OPERATORS, Node

TRANSFER_OPERATORS = tuple(op for op in OPERATORS if op.node == Node.RELAY)
METHODS = ("loreft_task", "loreft_locality")


def candidate_metrics(scores, samples, sequence, baseline):
    if not torch.isfinite(scores).all() or not torch.isfinite(baseline).all():
        raise FloatingPointError("Non-finite transfer candidate or baseline scores")
    result = metrics(scores, samples, sequence, baseline)
    result["paired"] = paired_metrics(
        scores.argmax(-1).tolist(), samples, sequence, baseline.argmax(-1).tolist()
    )
    return result


def greedy_metrics(outputs, samples, sequence, baseline, candidate):
    result = generation_metrics(outputs, samples, sequence, baseline)
    result["paired"] = paired_metrics(
        [r["prediction"] for r in outputs],
        samples,
        sequence,
        [r["prediction"] for r in baseline],
    )
    attach_candidate_agreement(result, candidate)
    return result


def _cross_representation(groups, source):
    results = {}
    for key, group in groups.items():
        representation, names, family = key.split("/")
        if representation == source:
            continue
        reference = groups[f"{source}/{names}/{family}"]
        results[key] = {
            op: {
                mode: {
                    evaluation: transfer_agreement(
                        reference["operators"][op][mode][evaluation]["paired"],
                        result[evaluation]["paired"],
                    )
                    for evaluation in ("candidate", "greedy")
                    if evaluation in result
                }
                for mode, result in modes.items()
            }
            for op, modes in group["operators"].items()
        }
    return results


def run_transfer(
    lm,
    cfg,
    *,
    layer=8,
    train_representation="en",
    strengths=None,
    preservation_weight=1.0,
    position=ReftPosition.STATE,
    max_new_tokens=16,
    loss_mode=LossMode.FULL_VOCAB,
    progress=lambda _: None,
):
    source, position, loss_mode = (
        Representation(train_representation),
        ReftPosition(position),
        LossMode(loss_mode),
    )
    strengths = [0, 0.5, 1, 2] if strengths is None else strengths
    if not strengths or any(not math.isfinite(a) for a in strengths):
        raise ValueError("Provide finite strengths")
    strengths = sorted({0.0, *strengths}, key=lambda a: (abs(a), a))
    if not math.isfinite(preservation_weight) or preservation_weight < 0:
        raise ValueError("preservation_weight must be finite and nonnegative")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    site = block_site(lm, layer)
    if position == ReftPosition.STATE and layer == max(discover_sites(lm)):
        raise ValueError("State intervention requires an earlier block; final block has no path")
    protocol = protocol_metadata(lm)
    protocol["instruction"] = "Representation-specific binary request; see saved prompts."
    protocol["boundary"] = (
        "last_token_inside_state_marker" if position == ReftPosition.STATE else "last_prompt_token"
    )
    protocol["generation_policy"] = (
        f"Reapply at the original {protocol['boundary']} on every uncached forward; "
        "never move to generated tokens."
    )
    if loss_mode == LossMode.FULL_VOCAB:
        protocol["training_completion_ids"] = answer_sequences(lm, [0, 1])
    train = transfer_questions("train", source, seed=cfg.seed)
    val = transfer_questions("validation", source, seed=cfg.seed)
    test = {
        f"{rep}/{names}/{family}": transfer_questions(
            "test", rep, names=names, family=family, seed=cfg.seed
        )
        for rep in Representation
        for names in Names
        for family in Family
    }
    checks = {
        "train": validate_state_prefixes(lm, train),
        "validation": validate_state_prefixes(lm, val),
        "test": {},
    }
    for key, samples in test.items():
        checks["test"][key] = validate_state_prefixes(lm, samples)

    def score(samples, transform=lambda h: h, sequence=()):
        return evaluate(lm, samples, site, [], transform, sequence, position=position)

    def generate(samples, label, transform=lambda h: h, sequence=()):
        return generate_questions(
            lm,
            samples,
            site,
            transform,
            sequence=sequence,
            max_new_tokens=max_new_tokens,
            position=position,
            progress=lambda msg: progress(f"{label}: {msg}"),
        )

    fitted, tensors, training, trials, selected, groups, cache = {}, {}, {}, {}, {}, {}, {}
    with frozen_model(lm):
        # This audit is descriptive only. Nothing from target representations enters fitting,
        # model selection, naming generation, or strength calibration.
        progress("Auditing natural understanding in all representations before training")
        for key, samples in test.items():
            base = score(samples)
            generated = generate(samples, f"Read {key}")
            natural = candidate_metrics(base, samples, (), base)
            groups[key] = {
                "understanding": {
                    "natural": {
                        "candidate": natural,
                        "greedy": greedy_metrics(generated, samples, (), generated, natural),
                    }
                },
                "operators": {},
            }
            cache[key] = (base, generated)
            g = groups[key]["understanding"]["natural"]["greedy"]
            progress(
                f"Reading {key}: accuracy={g['overall']['accuracy']:.3f}, "
                f"both worlds={g['paired']['all_nodes_correct']['correct']}/"
                f"{g['paired']['all_nodes_correct']['count']}"
            )

        progress(f"Training only on {source}; fixed site {site}")
        base_train, base_val = score(train), score(val)
        # Only the hidden width is needed; these are not donor targets.
        x = capture_states(lm, [train[0].prompt()], site, [], position=position)
        references = (
            reference_distributions(lm, train) if loss_mode == LossMode.FULL_VOCAB else None
        )
        for op in TRANSFER_OPERATORS:
            fitted[op.key], training[op.key], selected[op.key], trials[op.key] = {}, {}, {}, {}
            for method in METHODS:
                locality = cfg.locality_weight if method == "loreft_locality" else 0.0
                operator, log = train_loreft(
                    lm,
                    cfg,
                    train,
                    base_train,
                    site,
                    [],
                    op,
                    x.shape[1],
                    locality,
                    progress,
                    loss_mode=loss_mode,
                    full_reference=references,
                    position=position,
                )
                fitted[op.key][method], training[op.key][method] = operator, log
                log["representation"] = source.value
                for name, tensor in operator.state_dict().items():
                    tensors[f"{op.key}.{method}.{name}"] = (
                        tensor.detach().cpu().contiguous().clone()
                    )
                rows = []
                for alpha in strengths:
                    values = (
                        base_val
                        if alpha == 0
                        else score(val, make_transform(operator, method, alpha))
                    )
                    m = candidate_metrics(values, val, (op,), base_val)
                    rows.append(
                        {
                            "alpha": alpha,
                            "objective": list(transfer_selection(m, preservation_weight)),
                            "metrics": m,
                        }
                    )
                # Stable tie breaking: zero, then smallest absolute strength. Only source val.
                choice = max(rows, key=lambda row: tuple(row["objective"]))
                selected[op.key][method] = {k: choice[k] for k in ("alpha", "objective")}
                trials[op.key][method] = rows
                progress(f"Frozen {op.key}/{method}: alpha={choice['alpha']} ({source} validation)")
        del references, x

        # Evaluation starts only after ALL operator choices have been frozen.
        for key, samples in test.items():
            base, base_generated = cache[key]
            for op in TRANSFER_OPERATORS:
                sequence = (op,)
                baseline = candidate_metrics(base, samples, sequence, base)
                modes = {
                    "base": {
                        "candidate": baseline,
                        "greedy": greedy_metrics(
                            base_generated, samples, sequence, base_generated, baseline
                        ),
                    }
                }
                explicit = score(samples, sequence=sequence)
                explicit_m = candidate_metrics(explicit, samples, sequence, base)
                outputs = generate(samples, f"Explicit {key}/{op.key}", sequence=sequence)
                modes["explicit_prompt"] = {
                    "candidate": explicit_m,
                    "greedy": greedy_metrics(
                        outputs, samples, sequence, base_generated, explicit_m
                    ),
                }
                # Output-only constant controls, deliberately not labelled as model generations.
                for value in (0, 1):
                    constant = torch.zeros_like(base)
                    constant[:, value] = 1
                    modes[f"always_{value}"] = {
                        "kind": "output_only_control",
                        "candidate": candidate_metrics(constant, samples, sequence, base),
                    }
                for method in METHODS:
                    operator = fitted[op.key][method]
                    by_alpha = {0.0: modes["base"]}
                    for label, alpha in (
                        (method, selected[op.key][method]["alpha"]),
                        (f"{method}_fixed_1", 1.0),
                    ):
                        if alpha not in by_alpha:
                            transform = make_transform(operator, method, alpha)
                            values = score(samples, transform)
                            candidate = candidate_metrics(values, samples, sequence, base)
                            outputs = generate(
                                samples, f"Transfer {key}/{op.key}/{label}", transform
                            )
                            by_alpha[alpha] = {
                                "candidate": candidate,
                                "greedy": greedy_metrics(
                                    outputs, samples, sequence, base_generated, candidate
                                ),
                            }
                        modes[label] = {**by_alpha[alpha], "alpha": alpha}
                groups[key]["operators"][op.key] = modes
        restored = score(train)
        difference = float((restored - base_train).abs().max())
        if not torch.allclose(restored, base_train, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Transfer rollback failed")
    return {
        "experiment": "cross_representation_transfer_v1",
        "train_representation": source.value,
        "selection_scope": "source_validation_only",
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "seed": cfg.seed,
        "rank": cfg.rank,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "locality_weight": cfg.locality_weight,
        "preservation_weight": preservation_weight,
        "strengths": strengths,
        "loss_mode": loss_mode.value,
        "max_new_tokens": max_new_tokens,
        "site": {
            "path": site,
            "position": position.value,
            "selection": "predeclared",
            "prefix": [],
        },
        "answer_protocol": protocol,
        "state_prefix_checks": checks,
        "split": {
            "train": [q.key for q in train],
            "validation": [q.key for q in val],
            "test": {k: [q.key for q in qs] for k, qs in test.items()},
        },
        "dataset": {
            "train_schemes": ["and_copy", "and_inverted"],
            "held_out_topologies": ["and_chain", "and_inverted_chain"],
            "representations": [r.value for r in Representation],
            "aliases": {
                k: {n.value: name for n, name in qs[0].aliases.items()} for k, qs in test.items()
            },
            "source_train_test_prompt_overlap": {
                k: len({q.prompt() for q in qs} & {q.prompt() for q in train})
                for k, qs in test.items()
            },
        },
        "training": training,
        "validation_trials": trials,
        "selected": selected,
        "test": groups,
        "cross_representation": _cross_representation(groups, source.value),
        "rollback": {"max_score_difference": difference},
        "limitations": [
            "Base weights are frozen; this tests activation operators, not temporary weight dynamics.",
            "The site is predeclared; no donor localization or target-language calibration is performed.",
            "All 8 Boolean inputs occur in every split. Source/direct/seen test prompts overlap training; this is an in-domain control, not unseen-example generalization. Other languages, renamed identifiers and chain topologies test separate transfer axes.",
            "Validation uses separate identifiers in the source representation; renamed test identifiers are a third disjoint alphabet. Names are neutral Latin letters in all representations, with seed-shuffled role assignment shared by both mechanisms.",
            "Only one naming assignment per condition and three training layouts are used. Rule order and neutral ### boundary are shared across languages; symbolic notation is a representation, not a natural language.",
            "Target natural-understanding scores are audited before training but never used for fitting, selection, early stopping or filtering the dataset.",
            "Pair success requires both COPY and NOT states. Conditional baseline-correct and source-correct rates retain their denominators; an empty subset has null accuracy.",
            "Equation consistency uses predicted parents. Complete-state correctness separately checks original inputs and all intermediate nodes.",
            "Alpha is selected once per operator/method on source validation using paired candidate metrics. Fixed-alpha=1 diagnostics never select a target-specific operator.",
            "Candidate scoring ranks bare 0/1. Greedy diagnostics are unconstrained full-vocabulary generation; incomplete and invalid answers count as failures. Constant controls are synthetic predictions, not generated text.",
            "This suite tests relay=0/1 primitives on paired COPY/NOT; it does not test XOR, operator composition or autonomous choice of an intervention.",
            "Training and inference reuse one state position per prompt. The state is recomputed for each question, with no persistent shared KV world. One run is one seed; repeat runs to assess stability.",
        ],
    }, tensors
