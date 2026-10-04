"""Train neural COPY/NOT execution, then test unseen topology and state interventions."""

import platform
import random
import time
from dataclasses import asdict, replace

import torch
from torch.nn import functional as F

from .executor_model import LocalExecutor, pack, rollout
from .executor_tasks import Rule, episodes_for, make_dataset, oracle_step


def _score(correct, count):
    return {"correct": int(correct), "count": int(count)}


def trajectory_metrics(episodes, predicted, gold):
    predicted, gold = predicted.detach().cpu(), gold.detach().cpu()
    bits = predicted >= 0.5
    metrics = {
        k: _score(0, 0)
        for k in (
            "all_nodes_final",
            "free_nodes_final",
            "changed_free_nodes",
            "all_steps_exact",
            "paired_both_values",
            "mechanisms_final",
        )
    }
    metrics["preservation_damage"] = {"damaged": 0, "count": 0}
    metrics["free_preservation_damage"] = {"damaged": 0, "count": 0}
    metrics["per_step_free"] = [_score(0, 0) for _ in range(predicted.shape[1])]
    groups = {}
    for j, e in enumerate(episodes):
        n = len(e.graph.parents)
        correct = bits[j, :, :n] == gold[j, :, :n].bool()
        free = torch.tensor(e.free())
        baseline = torch.tensor(e.graph.equilibrium()).bool()
        changed = gold[j, -1, :n].bool() != baseline
        final_ok = bool(correct[-1].all())
        values = {
            "all_nodes_final": (final_ok, 1),
            "free_nodes_final": (correct[-1, free].sum(), free.sum()),
            "changed_free_nodes": (correct[-1, changed & free].sum(), (changed & free).sum()),
            "all_steps_exact": (correct.all(), 1),
        }
        for key, (a, b) in values.items():
            metrics[key]["correct"] += int(a)
            metrics[key]["count"] += int(b)
        for key, mask in (
            ("preservation_damage", ~changed),
            ("free_preservation_damage", ~changed & free),
        ):
            metrics[key]["damaged"] += int((~correct[-1, mask]).sum())
            metrics[key]["count"] += int(mask.sum())
        for t, score in enumerate(metrics["per_step_free"]):
            score["correct"] += int(correct[t, free].sum())
            score["count"] += int(free.sum())
        for i, rule in enumerate(e.graph.rules):
            if free[i]:
                expected = bool(bits[j, -1, e.graph.parents[i]]) ^ (rule == Rule.NOT)
                metrics["mechanisms_final"]["correct"] += int(bool(bits[j, -1, i]) == expected)
                metrics["mechanisms_final"]["count"] += 1
        groups.setdefault(e.id.rsplit("/", 1)[0], []).append(final_ok)
    pairs = [v for v in groups.values() if len(v) == 2]
    metrics["paired_both_values"] = _score(sum(all(v) for v in pairs), len(pairs))
    return metrics


def _loss(predicted, gold, inputs, protected, weight):
    weights = inputs.free.to(predicted.dtype) * (1 + weight * protected)
    loss = F.binary_cross_entropy(predicted[:, 1:], gold[:, 1:], reduction="none")
    return (loss * weights[:, None]).sum() / (weights.sum() * loss.shape[1]).clamp_min(1)


@torch.no_grad()
def evaluate(model, episodes, device):
    steps = max(e.graph.depth() for e in episodes) + 1
    inputs, gold, _ = pack(episodes, steps, device)
    predicted = rollout(model, inputs, steps)
    disconnected = replace(
        inputs,
        parents=torch.arange(inputs.parents.shape[1], device=device).expand_as(inputs.parents),
    )
    controls = {
        "oracle": gold,
        "clamp_only": inputs.initial[:, None].expand_as(gold),
        "one_step": torch.cat((predicted[:, :1], predicted[:, 1:2].expand(-1, steps, -1)), dim=1),
        "disconnected_parent": rollout(model, disconnected, steps),
    }
    records = []
    for j, e in enumerate(episodes):
        n = len(e.graph.parents)
        records.append(
            {
                "id": e.id,
                "parents": e.graph.parents,
                "rules": [r.name for r in e.graph.rules],
                "interventions": e.interventions,
                "predicted": predicted[j, :, :n].cpu().tolist(),
                "gold": gold[j, :, :n].int().cpu().tolist(),
            }
        )
    return {
        "metrics": trajectory_metrics(episodes, predicted, gold),
        "controls": {k: trajectory_metrics(episodes, p, gold) for k, p in controls.items()},
        "trajectories": records,
    }


@torch.no_grad()
def truth_table(model, device):
    rows = []
    for rule in (Rule.COPY, Rule.NOT):
        for own in (0, 1):
            for parent in (0, 1):
                p = model(
                    torch.tensor([[own, parent]], dtype=torch.float32, device=device),
                    torch.tensor([[1, 1]], device=device),
                    torch.tensor([[rule, Rule.ROOT]], device=device),
                )[0, 0].item()
                rows.append(
                    {
                        "rule": rule.name,
                        "own": own,
                        "parent": parent,
                        "probability_one": p,
                        "expected": parent ^ (rule == Rule.NOT),
                    }
                )
    return rows


@torch.no_grad()
def pulse_audit(model, episodes, device):
    """Edit actual recurrent memory once; never feed oracle states into subsequent steps."""
    eligible, exact, self_patch_ok, descendant_ok, descendant_count = 0, 0, 0, 0, 0
    max_permutation_error, records = 0.0, []
    for e in episodes:
        steps = e.graph.depth() + 1
        inputs, gold, _ = pack([e], steps, device)
        original = rollout(model, inputs, steps)
        order = tuple(reversed(range(len(e.graph.parents))))
        p_inputs, _, _ = pack([e.permute(order)], steps, device)
        permuted = rollout(model, p_inputs, steps)
        error = (permuted - original[:, :, list(order)]).abs().max().item()
        max_permutation_error = max(error, max_permutation_error)
        candidates = [i for i, free in enumerate(e.free()) if free and e.graph.descendants(i)]
        if not candidates:
            continue
        node = min(candidates, key=lambda i: (e.graph.depths()[i], i))
        prefix = 2
        state = original[:, prefix].clone()
        clean = rollout(model, inputs, steps, start=state)
        self_patch = rollout(model, inputs, steps, start=state.clone())
        self_patch_ok += int(torch.equal(clean, self_patch))
        is_eligible = bool(torch.equal(state >= 0.5, gold[:, prefix].bool()))
        altered = state.clone()
        # Even failed pre-states receive a real flip of their decoded memory.
        pulse_value = 1 - int(state[0, node] >= 0.5)
        altered[0, node] = pulse_value
        actual = rollout(model, inputs, steps, start=altered)
        oracle = gold[0, prefix].int().tolist()
        oracle[node] = pulse_value
        expected = [tuple(oracle)]
        for _ in range(steps):
            expected.append(oracle_step(e, expected[-1]))
        expected_t = torch.tensor(expected, device=device).bool()
        agreement = bool(torch.equal(actual[0] >= 0.5, expected_t))
        # Compare the predicted and oracle *effects*, not just absolute answers.
        clean_expected = [tuple(gold[0, prefix].int().tolist())]
        for _ in range(steps):
            clean_expected.append(oracle_step(e, clean_expected[-1]))
        effect = expected_t != torch.tensor(clean_expected, device=device).bool()
        descendants = torch.zeros_like(effect)
        descendants[:, list(e.graph.descendants(node))] = True
        mask = effect & descendants
        if is_eligible:
            eligible += 1
            exact += int(agreement)
            actual_effect = (actual[0] >= 0.5) != (clean[0] >= 0.5)
            descendant_ok += int((actual_effect & mask).sum())
            descendant_count += int(mask.sum())
        records.append(
            {
                "id": e.id,
                "node": node,
                "prefix_steps": prefix,
                "eligible": is_eligible,
                "all_steps_exact": agreement,
                "predicted": actual[0].cpu().tolist(),
                "gold": expected,
            }
        )
    return {
        "eligible": eligible,
        "considered": len(records),
        "all_steps_exact": _score(exact, eligible),
        "expected_descendant_changes": _score(descendant_ok, descendant_count),
        "self_patch": _score(self_patch_ok, len(records)),
        "max_permutation_error": max_permutation_error,
        "trajectories": records,
    }


def _train(cfg, kind, seed, train, validation, device, progress):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = LocalExecutor(kind, cfg.width, cfg.blocks, cfg.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    v_inputs, v_gold, v_protected = pack(validation, 4, device)
    best_key, best_weights, selected_step, curves = None, None, None, []
    for step in range(1, cfg.steps + 1):
        model.train()
        batch = [train[rng.randrange(len(train))] for _ in range(cfg.batch_size)]
        inputs, gold, protected = pack(batch, 4, device)
        optimizer.zero_grad(set_to_none=True)
        predicted = rollout(model, inputs, 4)
        loss = _loss(predicted, gold, inputs, protected, cfg.preservation_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {kind}/{seed}/{step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % cfg.validation_every == 0 or step == cfg.steps:
            model.eval()
            with torch.no_grad():
                v_pred = rollout(model, v_inputs, 4)
                v_loss = _loss(
                    v_pred, v_gold, v_inputs, v_protected, cfg.preservation_weight
                ).item()
            scores = trajectory_metrics(validation, v_pred, v_gold)
            key = (scores["all_steps_exact"]["correct"], -v_loss)
            curves.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "validation_loss": v_loss,
                    "validation_all_steps": scores["all_steps_exact"],
                }
            )
            if best_key is None or key > best_key:
                best_key, selected_step = key, step
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            progress(
                f"executor {kind}/{seed}: {step}/{cfg.steps}, "
                f"validation trajectories {scores['all_steps_exact']['correct']}/{len(validation)}"
            )
    model.load_state_dict(best_weights)
    model.eval()
    return model, best_weights, selected_step, curves


def run_executor(cfg, progress=print):
    cfg.validate()
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this PyTorch installation")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available in this PyTorch installation")
    data = make_dataset(cfg.dataset_seed, cfg.train_graphs, cfg.eval_graphs)
    splits = {k: episodes_for(v, k) for k, v in data.items()}
    splits["composition"] = episodes_for(data["test"], "composition", composition=True)
    report = {
        "experiment": "inspectable_causal_executor_v1",
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "runtime": {
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "device": device,
        },
        "supplied": [
            "directed graph",
            "ROOT/COPY/NOT labels",
            "root bits",
            "initial equilibrium",
            "hard root/do clamps",
            "one-hop routing",
            "oracle training trajectories",
        ],
        "learned": "shared free-node transition; one scalar probability per persistent node",
        "limits": "Known unary mechanisms only; no language understanding or causal discovery. "
        "Locality and clamping are architectural. Composition means simultaneous clamps.",
        "dataset": {
            k: [{**asdict(g), "topology_key": g.topology_key(), "depth": g.depth()} for g in graphs]
            for k, graphs in data.items()
        },
        "episodes": {
            k: [{"id": e.id, "interventions": e.interventions} for e in es]
            for k, es in splits.items()
        },
        "runs": {},
    }
    weights = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.threads)
    try:
        for kind in ("transformer", "mlp"):
            for seed in cfg.seeds:
                start = time.monotonic()
                model, state, selected, curves = _train(
                    cfg, kind, seed, splits["train"], splits["validation"], device, progress
                )
                scores = {k: evaluate(model, es, device) for k, es in splits.items()}
                audit = pulse_audit(model, splits["long"], device)
                report["runs"][f"{kind}/{seed}"] = {
                    "selected_step": selected,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "curves": curves,
                    "evaluation": scores,
                    "truth_table": truth_table(model, device),
                    "pulse_audit": audit,
                    "elapsed_seconds": time.monotonic() - start,
                }
                weights[f"{kind}-{seed}"] = state
    finally:
        torch.set_num_threads(old_threads)
    return report, weights
