"""LoReFT + locality, matched donor-regression baselines, graph-shift evaluation."""

import json
import random
from datetime import UTC, datetime
from functools import partial
from uuid import uuid4

import torch
from safetensors.torch import save_file

from .answer_protocol import protocol_metadata
from .causal_tasks import (
    TRAIN_SCHEMES,
    InterventionMode,
    PromptLayout,
    PromptStyle,
    Scheme,
    circuit_questions,
    metrics,
    selection,
    training_questions,
)
from .conditional import fit_map, fit_pca_map, predict
from .experiment import SCENARIOS, resolve_sequence
from .full_vocab import (
    LossMode,
    answer_sequences,
    full_vocab_loss,
    reference_distributions,
    teacher_forced_logits,
)
from .generation_evaluation import (
    attach_candidate_agreement,
    generate_questions,
    generation_metrics,
)
from .localization import capture_tail, discover_sites
from .positions import ReftPosition, intervention_positions, validate_state_prefixes
from .reft import LoReFT, frozen_model, intervention_scores, task_locality_loss
from .reft_localization import locate_block
from .world import OPERATORS


def block_site(lm, layer):
    catalog = discover_sites(lm)
    if layer not in catalog:
        raise ValueError(f"Unknown block {layer}; available: {sorted(catalog)}")
    return catalog[layer]["block"]


@torch.no_grad()
def capture_states(lm, prompts, site, prefix, *, position=ReftPosition.ANSWER):
    return torch.cat(
        [
            capture_tail(
                lm,
                prompts[i : i + 2],
                [site],
                1,
                prefix,
                positions=intervention_positions(lm, prompts[i : i + 2], position),
            )[site][:, 0]
            for i in range(0, len(prompts), 2)
        ]
    )


@torch.no_grad()
def evaluate(
    lm, samples, site, prefix, transform=lambda h: h, sequence=(), *, position=ReftPosition.ANSWER
):
    return torch.cat(
        [
            intervention_scores(
                lm,
                [q.prompt(sequence) for q in samples[i : i + 2]],
                site,
                prefix,
                transform,
                position=position,
            ).cpu()
            for i in range(0, len(samples), 2)
        ]
    )


def train_batches(samples, op, steps, seed):
    """Complete world/style batches; alternate worlds changed and unchanged by do(op)."""
    rng = random.Random(seed)
    groups = {}
    for i, q in enumerate(samples):
        groups.setdefault(q.state_key, []).append(i)
    if not groups:
        raise ValueError("Training requires complete states")
    for indices in groups.values():
        if {samples[i].node for i in indices} != set(samples[indices[0]].world.values()):
            raise ValueError("Training requires every question role for each state/style")
    schemes = list(dict.fromkeys(q.world.scheme for q in samples))
    pools = {}
    for scheme in schemes:
        changed, stable = [], []
        for indices in groups.values():
            if samples[indices[0]].world.scheme != scheme:
                continue
            pool = (
                changed
                if any(samples[i].answer((op,)) != samples[i].answer() for i in indices)
                else stable
            )
            pool.append(tuple(indices))
        pools[scheme] = [changed or stable.copy(), stable or changed.copy()]
        for pool in pools[scheme]:
            rng.shuffle(pool)
    result = []
    for t in range(steps):
        cycle, scheme = t // len(schemes), schemes[t % len(schemes)]
        pool = pools[scheme][cycle % 2]
        result.append(pool[(cycle // 2) % len(pool)])
    return result


def train_loreft(
    lm,
    cfg,
    samples,
    baseline,
    site,
    prefix,
    op,
    hidden_size,
    locality,
    progress,
    *,
    loss_mode=LossMode.BINARY,
    full_reference=None,
    position=ReftPosition.ANSWER,
):
    loss_mode = LossMode(loss_mode)
    if loss_mode == LossMode.FULL_VOCAB and full_reference is None:
        raise ValueError("Full-vocabulary training requires protected reference distributions")
    model = LoReFT(hidden_size, cfg.rank, cfg.seed).to(lm.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    batches = train_batches(samples, op, cfg.steps, cfg.seed)
    losses = []
    for step, indices in enumerate(batches):
        batch = [samples[i] for i in indices]
        optimizer.zero_grad(set_to_none=True)
        n_task = sum(q.node in q.world.affected(op) for q in batch)
        counts = n_task, len(batch) - n_task
        parts, by_node = {"task_ce": 0.0, "locality_kl": 0.0}, {}
        total = 0.0
        # Accumulate the exact full-state mean CE + mean KL, without retaining all graphs.
        for start in range(0, len(batch), 2):
            micro = batch[start : start + 2]
            if loss_mode == LossMode.FULL_VOCAB:
                targets = answer_sequences(lm, [q.answer((op,)) for q in micro])
                token_logits = teacher_forced_logits(
                    lm,
                    [q.prompt() for q in micro],
                    targets,
                    site=site,
                    transform=model,
                    position=position,
                )
                loss, chunk_parts, details = full_vocab_loss(
                    token_logits,
                    [full_reference[i] for i in indices[start : start + 2]],
                    targets,
                    [q.node in q.world.affected(op) for q in micro],
                    locality,
                    normalization_counts=counts,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite full-vocabulary LoReFT loss")
                loss.backward()
                total += float(loss.detach())
                for key, value in chunk_parts.items():
                    parts[key] += value
                for q, detail in zip(micro, details, strict=True):
                    by_node[q.node.value] = {
                        "count": 1,
                        "affected": q.node in q.world.affected(op),
                        **detail,
                    }
                continue
            scores = intervention_scores(
                lm, [q.prompt() for q in micro], site, prefix, model, position=position
            )
            labels = torch.tensor([q.answer((op,)) for q in micro], device=lm.device)
            affected = torch.tensor(
                [q.node in q.world.affected(op) for q in micro], device=lm.device
            )
            original = baseline[list(indices[start : start + 2])]
            loss, chunk_parts = task_locality_loss(
                scores,
                original,
                labels,
                affected,
                locality,
                normalization_counts=counts,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite LoReFT loss")
            loss.backward()
            total += float(loss.detach())
            for key, value in chunk_parts.items():
                parts[key] += value
            with torch.no_grad():
                logp = scores.detach().float().log_softmax(-1).cpu()
                original_logp = original.float().log_softmax(-1).cpu()
                kl = (original_logp.exp() * (original_logp - logp)).sum(-1)
                for row, q in enumerate(micro):
                    is_affected = q.node in q.world.affected(op)
                    eligible = not is_affected and int(original[row].argmax()) == q.answer()
                    by_node[q.node.value] = {
                        "count": 1,
                        "affected": is_affected,
                        "target_ce": float(-logp[row, q.answer((op,))]),
                        "locality_kl": None if is_affected else float(kl[row]),
                        "protected_eligible": int(eligible),
                        "protected_damaged": int(
                            eligible and int(logp[row].argmax()) != q.answer()
                        ),
                    }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"step": step + 1, "loss": total, **parts, "by_node": by_node})
        if step == 0 or (step + 1) % 10 == 0:
            progress(f"LoReFT {op.key} locality={locality}: {step + 1}/{cfg.steps} {parts}")
    return model.eval(), {
        "locality_weight": locality,
        "loss_mode": loss_mode.value,
        "position": ReftPosition(position).value,
        "schemes": [s.value for s in dict.fromkeys(q.world.scheme for q in samples)],
        "batching": "full_state",
        "microbatch_size": 2,
        "styles": list(dict.fromkeys(q.style.value for q in samples)),
        "losses": losses,
        "question_keys": [[samples[i].key for i in b] for b in batches],
        "unique_questions": len({i for b in batches for i in b}),
    }


def run_reft_suite(
    lm,
    cfg,
    *,
    layer=12,
    localization_layers=None,
    max_new_tokens=16,
    strengths=None,
    pca_components=None,
    preservation_weight=1.0,
    loss_mode=LossMode.FULL_VOCAB,
    train_schemes=TRAIN_SCHEMES,
    position=ReftPosition.STATE,
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
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    answer_protocol = protocol_metadata(lm)
    loss_mode = LossMode(loss_mode)
    if loss_mode == LossMode.FULL_VOCAB:
        answer_protocol["training_completion_ids"] = answer_sequences(lm, [0, 1])
    prefix = []
    position = ReftPosition(position)
    evaluate_at = partial(evaluate, position=position)
    capture_at = partial(capture_states, position=position)
    train = training_questions(
        "train", train_schemes, styles=tuple(PromptStyle), layout=PromptLayout.STATE_FIRST
    )
    val = training_questions("validation", train_schemes, layout=PromptLayout.STATE_FIRST)
    prefix_checks = {
        "train": validate_state_prefixes(lm, train),
        "validation": validate_state_prefixes(lm, val),
        "test": {},
    }
    train_schemes = tuple(dict.fromkeys(q.world.scheme for q in train))
    methods = (
        "constant",
        "ridge",
        *(f"pca_{k}" for k in sorted(set(pca_components))),
        "loreft_task",
        "loreft_locality",
    )
    fitted, tensors, training, trials, selected = {}, {}, {}, {}, {}
    with frozen_model(lm):
        site, localization = locate_block(
            lm,
            [layer] if localization_layers is None else localization_layers,
            preservation_weight,
            progress,
            samples=val,
            position=position,
        )
        base_train, base_val = (
            evaluate_at(lm, train, site, prefix),
            evaluate_at(lm, val, site, prefix),
        )
        x = capture_at(lm, [q.prompt() for q in train], site, prefix)
        full_reference = None
        if loss_mode == LossMode.FULL_VOCAB:
            progress("Caching full-vocabulary protected distributions (CPU)")
            full_reference = reference_distributions(lm, train)
        for op in OPERATORS:
            progress(f"Training and selecting {op.key}")
            donor = capture_at(lm, [q.prompt((op,)) for q in train], site, prefix)
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
                    lm,
                    cfg,
                    train,
                    base_train,
                    site,
                    prefix,
                    op,
                    x.shape[1],
                    locality,
                    progress,
                    loss_mode=loss_mode,
                    full_reference=full_reference,
                    position=position,
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
                        else evaluate_at(
                            lm, val, site, prefix, make_transform(model, method, alpha)
                        )
                    )
                    m = metrics(scores, val, (op,), base_val)
                    entry = {
                        "alpha": alpha,
                        "objective": list(selection(m, preservation_weight)),
                        "all_nodes_correct": m["all_nodes_correct"],
                        "relay_lamp_correct": m["relay_lamp_correct"],
                        "protected_damage": m["protected_damage"],
                        "equation_consistency": m["equation_consistency"],
                        **{k: m[k] for k in ("overall", "changed", "by_node", "by_label")},
                    }
                    trials[op.key][method].append(entry)
                    best = selected[op.key].get(method)
                    if best is None or entry["objective"] > best["objective"]:
                        selected[op.key][method] = entry
        progress("Fixed alpha=1 LoReFT diagnostics: train and validation")
        fixed = {
            "alpha": 1.0,
            "used_for_selection": False,
            "train": fixed_loreft_metrics(
                lm, train, base_train, site, prefix, fitted, position=position
            ),
            "validation": fixed_loreft_metrics(
                lm, val, base_val, site, prefix, fitted, position=position
            ),
            "test": {},
        }
        tests, exact, greedy = {}, {}, {}
        for scheme in Scheme:
            progress(f"Held-out evaluation: {scheme.value}")
            samples = circuit_questions("test", scheme, layout=PromptLayout.STATE_FIRST)
            prefix_checks["test"][scheme.value] = validate_state_prefixes(lm, samples)
            base = evaluate_at(lm, samples, site, prefix)
            fixed["test"][scheme.value] = fixed_loreft_metrics(
                lm,
                samples,
                base,
                site,
                prefix,
                fitted,
                position=position,
            )
            tests[scheme.value], exact[scheme.value] = {}, {}
            for name, sequence in SCENARIOS.items():
                modes = {
                    "base": base,
                    "explicit_prompt": evaluate_at(lm, samples, site, prefix, sequence=sequence),
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

                    modes[method] = evaluate_at(lm, samples, site, prefix, compose)
                tests[scheme.value][name] = {
                    m: metrics(s, samples, sequence, base) for m, s in modes.items()
                }
                donor = capture_at(lm, [q.prompt(sequence) for q in samples], site, prefix)
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
                                position=position,
                            ).cpu()
                        )
                exact[scheme.value][name] = metrics(torch.cat(rows), samples, sequence, base)
            progress(f"Unconstrained generation: {scheme.value}")
            greedy[scheme.value] = greedy_loreft_metrics(
                lm,
                samples,
                site,
                fitted,
                selected,
                tests[scheme.value],
                fixed["test"][scheme.value],
                max_new_tokens,
                progress,
                position=position,
            )
        restored = evaluate_at(lm, train, site, prefix)
        difference = float((restored - base_train).abs().max())
        if not torch.allclose(restored, base_train, atol=1e-5, rtol=1e-5):
            raise RuntimeError("LoReFT rollback failed")
    return {
        "experiment": "loreft_causal_suite_v5",
        "prompt_layout": PromptLayout.STATE_FIRST.value,
        "state_prefix_checks": prefix_checks,
        "loss_mode": loss_mode.value,
        "intervention_mode": InterventionMode.REPLACE_EQUATION.value,
        "answer_protocol": answer_protocol,
        "localization": localization,
        "max_new_tokens": max_new_tokens,
        "model": {"profile": cfg.profile, "device": str(lm.device)},
        "seed": cfg.seed,
        "rank": cfg.rank,
        "steps": cfg.steps,
        "learning_rate": cfg.learning_rate,
        "locality_weight": cfg.locality_weight,
        "preservation_weight": preservation_weight,
        "strengths": strengths,
        "site": {
            "path": site,
            "kind": "block",
            "boundary": "state" if position == ReftPosition.STATE else "prompt",
            "position": position.value,
            "prefix": prefix,
        },
        "train_schemes": [s.value for s in train_schemes],
        "test_groups": {
            "seen_mechanisms": [s.value for s in train_schemes],
            "held_out_mechanisms_or_topologies": [
                s.value for s in Scheme if s not in train_schemes
            ],
        },
        "split": {
            "train": [q.key for q in train],
            "validation": [q.key for q in val],
            "test": {s.value: [q.key for q in circuit_questions("test", s)] for s in Scheme},
        },
        "methods": list(methods),
        "training": training,
        "train_styles": [style.value for style in PromptStyle],
        "selected": selected,
        "validation_trials": trials,
        "test": tests,
        "oracle_diagnostics": exact,
        "fixed_strength_diagnostics": fixed,
        "greedy_test": greedy,
        "rollback": {"max_score_difference": difference},
        "sources": ["https://arxiv.org/abs/2404.03592", "https://arxiv.org/abs/2110.11309"],
        "limitations": [
            "Native LoReFT equation with QR basis and identity initialization; not a reproduction of paper benchmarks or the pyreft optimizer setup.",
            "Full-vocabulary mode uses token CE including one EOS and forward KL over the complete vocabulary; binary mode retains the two-candidate objective. Alpha selection and non-greedy metrics still use bare 0/1 ranking without EOS.",
            "Protected KL uses the same teacher-forced natural answer history for baseline and intervention, averaged over its tokens including EOS. It does not constrain arbitrary generated histories.",
            "The first configured EOS ID is the training target; generation accepts all configured stop IDs. The intervention position is fixed during decoding. An edit at the final block cannot influence later positions; this is a negative control for state-position experiments.",
            "Task labels supervise affected variables; locality and preservation metrics cover all structurally unaffected nodes, including relay and bridge for lamp edits. Symbolic masks are used in supervision and evaluation, never given to the operator at inference.",
            "Equation consistency checks predicted parents under the intervened equations. Consistent but wrong input states can pass this check; all_nodes_correct independently requires the complete correct counterfactual state. Invalid/missing answers fail dependent equations.",
            "Selection first maximizes all_nodes_correct minus the full unaffected-node damage penalty, then equation consistency, relay/lamp accuracy and overall accuracy. Equation checks are evaluation criteria, not an additional training loss.",
            "Locality is a soft training objective, not a guarantee. loreft_task is an identically initialized zero-locality ablation.",
            "Each step contains every role for one world/style, with microbatch accumulation. Training schemes alternate; within each scheme changed/stable worlds alternate. Validation and held-out evaluation use the default wording; train uses three styles.",
            "Training by-node losses are measured before each update on the scheduled batch; fixed-alpha train metrics evaluate the final operator on every training question.",
            "Fixed-alpha=1 diagnostics cover primitive operations on train, validation and test, independently of validation selection; they are not used to choose operators. Compositions in test use selected strengths only.",
            "Donor-regression baselines have different supervision from LoReFT; they share train worlds, site and validation alpha grid.",
            "Compositions sequentially transform the same hidden vector; they do not rerun the network between primitive operators.",
            "Revision uses external latest-write-wins resolution, not learned contradiction detection.",
            "COPY and NOT are the default training mechanisms; other mechanisms and all bridge topologies remain test-only. Seen-mechanism test wording is reported separately from held-out structures. All tasks use Boolean variables and the same names; no arbitrary-text reasoning claim.",
            "Candidate layer list may reflect earlier experiments. One shared block is selected by validation donor transfer; this need not be the best layer for learned LoReFT. The final block is excluded from state-position selection. Training continues if no eligible donor site beats baseline, with that outcome recorded.",
            "Validation donors select the intervention site; test donors remain separate privileged diagnostics. Shifted schemes never participate in fitting or selection. Base model weights remain frozen.",
            "v5 uses the same state-first prompts for state and answer positions. State edits use the last token wholly inside the marked description before the question; answer edits use the last prompt token. Fast-tokenizer offsets must match actual chat tokenization. QUERY_FIRST wording moves its query after the description in this layout.",
            "State-mode token prefixes are checked to be identical across question roles for every world/style. Causal forwards recompute that state per question; tiny activation differences from numerical kernels are reported at validation localization. No shared KV cache or persistent edited world is claimed.",
            "One state token may not encode the whole circuit. This experiment tests whether later blocks can use its intervention across questions; it does not guarantee a causal abstraction. Names and scheme identifiers are not randomized.",
            "Scoring, teacher forcing and free generation edit the same resolved token with bare 0/1 candidates and no forced answer prefix. Greedy decoding uses the full vocabulary and can disagree with binary ranking or fail the required format.",
            "Greedy diagnostics evaluate primitive LoReFT operators at both selected strength and alpha=1. Compositions are still evaluated by candidate scoring only.",
            "Truncated generation counts as incomplete even when its partial text is a digit; it is not a completed binary answer. Unaffected-answer damage is relative to correct baseline generations, not candidate predictions.",
        ],
    }, tensors


def greedy_loreft_metrics(
    lm,
    samples,
    site,
    fitted,
    selected,
    selected_scores,
    fixed_scores,
    max_new_tokens,
    progress,
    *,
    position=ReftPosition.ANSWER,
):
    def generate(label, transform=lambda h: h, sequence=()):
        return generate_questions(
            lm,
            samples,
            site,
            transform,
            sequence=sequence,
            max_new_tokens=max_new_tokens,
            progress=lambda message: progress(f"Greedy {label}: {message}"),
            position=position,
        )

    base = generate("base")
    result = {}
    for op in OPERATORS:
        outputs = {"base": base, "explicit_prompt": generate(op.key + " explicit", sequence=(op,))}
        candidates = {
            "base": selected_scores[op.key]["base"],
            "explicit_prompt": selected_scores[op.key]["explicit_prompt"],
        }
        alphas = {}
        for method in ("loreft_task", "loreft_locality"):
            cache = {0.0: base}
            for mode, alpha, ranked in (
                ("selected", selected[op.key][method]["alpha"], selected_scores[op.key][method]),
                ("fixed", 1.0, fixed_scores[op.key][method]),
            ):
                name = f"{method}_{mode}"
                if alpha not in cache:
                    cache[alpha] = generate(
                        op.key + " " + name, make_transform(fitted[op.key][method], method, alpha)
                    )
                outputs[name], candidates[name], alphas[name] = cache[alpha], ranked, alpha
        result[op.key] = {}
        for name, generated in outputs.items():
            m = generation_metrics(generated, samples, (op,), base)
            attach_candidate_agreement(m, candidates[name])
            if name in alphas:
                m["alpha"] = alphas[name]
            result[op.key][name] = m
    return result


def fixed_loreft_metrics(lm, samples, base, site, prefix, fitted, *, position=ReftPosition.ANSWER):
    """Predeclared unit-strength diagnostics; no donor and no selection side effects."""
    result = {}
    for op in OPERATORS:
        result[op.key] = {"base": metrics(base, samples, (op,), base)}
        for method in ("loreft_task", "loreft_locality"):
            scores = evaluate(
                lm,
                samples,
                site,
                prefix,
                make_transform(fitted[op.key][method], method, 1.0),
                position=position,
            )
            result[op.key][method] = metrics(scores, samples, (op,), base)
    return result


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
        metadata={
            "experiment": report["experiment"],
            "site": json.dumps(report["site"]),
            "answer_protocol": json.dumps(report.get("answer_protocol", {})),
        },
    )
    lines = ["# " + report["experiment"], ""]
    if kind == "reft":
        lines += [
            "## Run protocol",
            "",
            f"Loss: `{report.get('loss_mode', 'binary')}`; intervention: `{report.get('intervention_mode', 'legacy_override')}`.",
            f"Position: `{report['site'].get('position', 'answer')}`; prompt layout: `{report.get('prompt_layout', 'legacy')}`.",
            "",
            "Training / validation mechanisms: "
            + ", ".join(report.get("train_schemes", ["and_copy"]))
            + ".",
            "",
            "Held-out mechanisms / topologies: "
            + ", ".join(report.get("test_groups", {}).get("held_out_mechanisms_or_topologies", []))
            + ".",
            "",
        ]
        lines += [
            "| Scheme | Scenario | Method | Accuracy | All nodes | Equations consistent | New protected errors |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for scheme, scenarios in report["test"].items():
            for name, methods in scenarios.items():
                for method, m in {
                    **methods,
                    "exact_donor": report["oracle_diagnostics"][scheme][name],
                }.items():
                    lines.append(
                        f"| {scheme} | {name} | {method} | {m['overall']['accuracy']:.3f} | {m['all_nodes_correct']:.3f} | {m['equation_consistency']['all_satisfied']:.3f} | {m['protected_damage']['damaged']} |"
                    )
        lines += [
            "",
            "## Selected on validation",
            "",
            "```json",
            json.dumps(report["selected"], indent=2),
            "```",
        ]
        if "fixed_strength_diagnostics" in report:
            fixed = report["fixed_strength_diagnostics"]
            lines += [
                "",
                "## Fixed alpha=1 diagnostics (not used for selection)",
                "",
                "| Split / scheme | Operation | Method | Accuracy | Changed | All nodes | Equations consistent | Protected damage by node |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
            groups = {
                "train": fixed["train"],
                "validation": fixed["validation"],
                **{"test/" + s: value for s, value in fixed["test"].items()},
            }
            for split, operations in groups.items():
                for op, methods in operations.items():
                    for method, m in methods.items():
                        damage = m["protected_damage"]["by_node"]
                        values = " / ".join(
                            f"{n}: {v['damaged']}/{v['eligible']}"
                            for n, v in damage.items()
                            if v["eligible"]
                        )
                        changed = m["changed"]["accuracy"]
                        changed_text = "n/a" if changed is None else f"{changed:.3f}"
                        lines.append(
                            f"| {split} | {op} | {method} | {m['overall']['accuracy']:.3f} | {changed_text} | {m['all_nodes_correct']:.3f} | {m['equation_consistency']['all_satisfied']:.3f} | {values or 'n/a'} |"
                        )
        if "localization" in report:
            lines += [
                "",
                "## Bare-digit validation localization",
                "",
                "```json",
                json.dumps(
                    {k: v for k, v in report["localization"].items() if k != "trials"}, indent=2
                ),
                "```",
            ]
        if "greedy_test" in report:
            lines += [
                "",
                "## Unconstrained greedy generation",
                "",
                "| Scheme | Operation | Mode | Accuracy | All nodes | Equations consistent | Format errors | Incomplete | Protected damage by node |",
                "|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
            for scheme, operations in report["greedy_test"].items():
                for op, modes in operations.items():
                    for mode, m in modes.items():
                        damage = m["protected_damage"]["by_node"]
                        values = " / ".join(
                            f"{n}: {v['damaged']}/{v['eligible']}"
                            for n, v in damage.items()
                            if v["eligible"]
                        )
                        lines.append(
                            f"| {scheme} | {op} | {mode} | {m['overall']['accuracy']:.3f} | {m['all_nodes_correct']:.3f} | {m['equation_consistency']['all_satisfied']:.3f} | {m['overall']['format_errors']} | {m['overall']['incomplete']} | {values or 'n/a'} |"
                        )
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
