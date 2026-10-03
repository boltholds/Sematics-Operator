"""Supervised primitive weight edits, held-out composition, locality and rollback."""

import hashlib
import platform
import random
from collections.abc import Callable
from dataclasses import asdict

import torch
import transformers
from torch import Tensor
from torch.nn import functional as F

from .config import Settings
from .model import LocalLanguageModel
from .weights import WeightPatch, WeightSession, randomized
from .world import OPERATORS, Intervention, Node, Question, questions

SCENARIOS = {
    "relay_0": (OPERATORS[0],),
    "relay_1": (OPERATORS[1],),
    "lamp_1": (OPERATORS[2],),
    "composition": (OPERATORS[0], OPERATORS[2]),
    "composition_reversed": (OPERATORS[2], OPERATORS[0]),
    "revision": (OPERATORS[0], OPERATORS[1]),
    "rollback": (),
}


def resolve_sequence(sequence: tuple[Intervention, ...]) -> tuple[Intervention, ...]:
    """Explicit latest-write-wins revision; not learned contradiction detection."""
    state: dict[Node, Intervention] = {}
    for intervention in sequence:
        state[intervention.node] = intervention
    return tuple(state.values())


def fingerprint(tensor: Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def score_questions(
    lm: LocalLanguageModel,
    samples: tuple[Question, ...],
    interventions: tuple[Intervention, ...] = (),
) -> Tensor:
    return torch.cat(
        [
            lm.scores([q.prompt(interventions) for q in samples[i : i + 2]]).cpu()
            for i in range(0, len(samples), 2)
        ]
    )


@torch.no_grad()
def capture_probes(lm: LocalLanguageModel, prompts: list[str], target: str) -> Tensor:
    return torch.cat(
        [lm.representations(prompts[i : i + 2], target) for i in range(0, len(prompts), 2)]
    )


def summarize(
    scores: Tensor,
    samples: tuple[Question, ...],
    interventions: tuple[Intervention, ...],
    baseline: Tensor,
) -> dict:
    probabilities = scores.softmax(-1)
    records = []
    for i, q in enumerate(samples):
        expected, base = q.answer(interventions), q.answer()
        records.append(
            {
                "key": q.key,
                "node": q.node.value,
                "expected": expected,
                "base_expected": base,
                "prediction": int(scores[i].argmax()),
                "changed": expected != base,
                "p1": float(probabilities[i, 1]),
                "candidate_logp": scores[i].tolist(),
                "base_prediction": int(baseline[i].argmax()),
            }
        )

    def group(rows):
        count = len(rows)
        return {
            "count": count,
            "accuracy": sum(r["prediction"] == r["expected"] for r in rows) / count
            if count
            else None,
        }

    unchanged = [r for r in records if not r["changed"]]
    preserved_correct = [r for r in unchanged if r["base_prediction"] == r["expected"]]
    return {
        "overall": group(records),
        "changed": group([r for r in records if r["changed"]]),
        "unchanged": group(unchanged),
        "preservation_of_correct_base_answers": group(preserved_correct),
        "by_node": {n.value: group([r for r in records if r["node"] == n.value]) for n in Node},
        "records": records,
    }


def fit_operator(
    lm: LocalLanguageModel,
    session: WeightSession,
    cfg: Settings,
    intervention: Intervention,
    samples: tuple[Question, ...],
    baseline: Tensor,
    progress: Callable[[str], None],
) -> tuple[WeightPatch, list[float]]:
    session.reset(cfg.seed)  # Equal initialization and budget across primitive operators.
    optimizer = torch.optim.Adam(session.parameters(), lr=cfg.learning_rate)
    rng = random.Random(cfg.seed)
    order = list(range(len(samples)))
    losses = []
    for step in range(cfg.steps):
        if step % (len(samples) // 2) == 0:
            rng.shuffle(order)
        offset = (step * 2) % len(samples)
        indices = order[offset : offset + 2]
        batch = [samples[i] for i in indices]
        scores = lm.scores([q.prompt() for q in batch])
        targets = torch.tensor([q.answer((intervention,)) for q in batch], device=lm.device)
        loss = F.cross_entropy(scores, targets)
        unchanged = torch.tensor(
            [q.answer((intervention,)) == q.answer() for q in batch], device=lm.device
        )
        if unchanged.any():
            base_prob = baseline[indices].to(lm.device).softmax(-1)
            locality = F.kl_div(
                scores[unchanged].log_softmax(-1), base_prob[unchanged], reduction="batchmean"
            )
            loss = loss + cfg.locality_weight * locality
        a, b = session.parameters()
        delta_mean_square = ((b.T @ b) * (a @ a.T)).sum() * session.layer.scale**2
        delta_mean_square = delta_mean_square / (b.shape[0] * a.shape[1])
        loss = loss + cfg.regularization * delta_mean_square
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite training loss; reduce learning_rate or use float32"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(session.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == cfg.steps:
            progress(f"{intervention.key}: step {step + 1}/{cfg.steps}, loss={losses[-1]:.5f}")
    return session.snapshot(), losses


def run_experiment(
    lm: LocalLanguageModel, cfg: Settings, progress: Callable[[str], None] = lambda _: None
):
    torch.manual_seed(cfg.seed)
    target = lm.choose_target(cfg.target_module)
    train, test = questions("train"), questions("test")
    base_hash = fingerprint(lm.model.get_submodule(target).weight)
    progress(f"Target: {target}; computing original-model scores")
    base_train, base_test = score_questions(lm, train), score_questions(lm, test)
    probe_prompts = [q.prompt() for q in test if q.node == Node.LAMP]
    representations = {"baseline": capture_probes(lm, probe_prompts, target)}
    patches, histories, scenarios = {}, {}, {}
    with WeightSession(lm.model, target, cfg.rank) as session:
        for intervention in OPERATORS:
            patch, history = fit_operator(
                lm, session, cfg, intervention, train, base_train, progress
            )
            patches[intervention.key], histories[intervention.key] = patch, history
        controls = {
            key: randomized(p, cfg.seed + 100 + i) for i, (key, p) in enumerate(patches.items())
        }
        for name, sequence in SCENARIOS.items():
            progress(f"Evaluating {name} on unseen names and wording")
            active = resolve_sequence(sequence)
            with session.branch(tuple(patches[i.key] for i in active)):
                edited = score_questions(lm, test)
                representations[name] = capture_probes(lm, probe_prompts, target)
            with session.branch(tuple(controls[i.key] for i in active)):
                control = score_questions(lm, test)
            with session.branch(()):
                prompted = score_questions(lm, test, sequence)
            scenarios[name] = {
                "sequence": [i.key for i in sequence],
                "active_operators": [i.key for i in active],
                "base": summarize(base_test, test, sequence, base_test),
                "weight_edit": summarize(edited, test, sequence, base_test),
                "random_norm_matched": summarize(control, test, sequence, base_test),
                "explicit_prompt": summarize(prompted, test, sequence, base_test),
            }
    restored = score_questions(lm, test)
    max_difference = float((restored - base_test).abs().max())
    unchanged = fingerprint(lm.model.get_submodule(target).weight) == base_hash
    if not unchanged or not torch.allclose(restored, base_test, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Rollback verification failed; experiment results are invalid")
    effective_settings = asdict(cfg)
    effective_settings["model_path"] = str(cfg.model_path)
    effective_settings["output_dir"] = str(cfg.output_dir)
    report = {
        "schema_version": 1,
        "experiment": "supervised_weight_operator_v1",
        "claim": "Infrastructure and intervention-transfer benchmark; not evidence of general reasoning.",
        "settings": effective_settings,
        "model": {
            "checkpoint_format": "gguf"
            if cfg.model_path.suffix.lower() == ".gguf"
            else "huggingface",
            "dequantized": cfg.model_path.suffix.lower() == ".gguf",
            "class": type(lm.model).__name__,
            "target_module": target,
            "target_weight_sha256": base_hash,
            "device": str(lm.device),
            "dtype": str(next(lm.model.parameters()).dtype),
            "parameters": sum(p.numel() for p in lm.model.parameters()),
            "config": lm.model.config.to_dict(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "split": {
            "train_questions": len(train),
            "test_questions": len(test),
            "overlap": len({q.key for q in train} & {q.key for q in test}),
            "test_shift": "New circuit names and wording; same Boolean truth-table support.",
            "training": "Primitive operations only; no composition/revision examples.",
        },
        "training_loss": histories,
        "patch_norms": {key: patch.norm() for key, patch in patches.items()},
        "scenarios": scenarios,
        "rollback": {"max_score_difference": max_difference, "base_weight_unchanged": unchanged},
        "limitations": [
            "Primitive operators are supervised task-specific adapters, not discovered universal semantics.",
            "Composition is addition of independently trained effective weight deltas; success is measured.",
            "Revision is explicit last-write-wins, not autonomous contradiction detection.",
            "Evaluation measures restricted binary candidate likelihoods, not free-form generation.",
            "Held-out names/templates test limited transfer, not unseen mechanisms or long reasoning chains.",
            "Single seed and one norm-matched random control per operator; no significance claim.",
            "Activation probes describe changes and do not independently establish causal mediation.",
        ],
    }
    return report, patches, representations
