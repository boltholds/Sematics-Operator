from collections import defaultdict

import pytest
import torch

from semantics_operator.causal_tasks import CircuitNode, PromptStyle
from semantics_operator.world import Intervention, Node


@pytest.mark.parametrize("field", ["scores", "baseline"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_predictions_cannot_enter_transfer_calibration(field, value):
    from semantics_operator.transfer_experiment import candidate_metrics
    from semantics_operator.transfer_tasks import transfer_questions

    qs = transfer_questions("validation", "en")
    tensors = {"scores": torch.zeros(len(qs), 2), "baseline": torch.zeros(len(qs), 2)}
    tensors[field][0, 1] = value
    with pytest.raises(FloatingPointError, match="finite"):
        candidate_metrics(
            tensors["scores"], qs, (Intervention(Node.RELAY, 1),), tensors["baseline"]
        )


def test_translations_and_renamings_preserve_pairs_without_prompt_identifiers():
    from semantics_operator.transfer_tasks import transfer_questions

    aligned = defaultdict(list)
    for representation in ("en", "ru", "symbolic"):
        for names in ("seen", "renamed"):
            samples = transfer_questions("test", representation, names=names, seed=42)
            assert len(samples) == 80
            assert len({q.key for q in samples}) == 80
            for q in samples:
                aligned[q.pair_id, q.world.scheme, q.node].append(q)
                assert q.world.name not in q.prompt()
                assert q.world.scheme.value not in q.prompt()
                assert q.aliases[q.node] in q.prompt().split("\n###")[1]
            pairs = defaultdict(list)
            for q in samples:
                if q.node == CircuitNode.LAMP:
                    pairs[q.pair_id].append(q)
            for copy, inverted in pairs.values():
                assert copy.aliases == inverted.aliases
                assert copy.answer((Intervention(Node.RELAY, 1),)) == 1
                assert inverted.answer((Intervention(Node.RELAY, 1),)) == 0
                # Exactly one equation differs; legends and inputs stay identical.
                c, n = copy.prompt().splitlines(), inverted.prompt().splitlines()
                assert sum(a != b for a, b in zip(c, n, strict=True)) == 1
    for rows in aligned.values():
        assert len(rows) == 6
        assert len({q.answer() for q in rows}) == 1
    train = transfer_questions("train", "en")
    val = transfer_questions("validation", "en")
    test = transfer_questions("test", "en", names="renamed")
    vocab = lambda rows: {name for q in rows for name in q.aliases.values()}
    assert not vocab(train) & vocab(val)
    assert not vocab(train) & vocab(test)
    assert not vocab(val) & vocab(test)
    assert all(q.representation == "en" for q in train)
    assert len(train) == 240
    assert {q.style for q in train} == set(PromptStyle)


def test_transfer_prefixes_and_equation_replacement_on_all_renderings():
    from test_model import tiny_model

    from semantics_operator.positions import ReftPosition, intervention_positions
    from semantics_operator.transfer_tasks import transfer_questions

    lm = tiny_model()
    lm.tokenizer.chat_template = "user: {{ messages[0]['content'] }}\nassistant:"
    for representation in ("en", "ru", "symbolic"):
        qs = transfer_questions("test", representation, family="chain")
        group = qs[:6]
        prompts = [q.prompt() for q in group]
        pos = intervention_positions(lm, prompts, ReftPosition.STATE)
        prefixes = [lm._prompt_ids(p)[: i + 1] for p, i in zip(prompts, pos, strict=True)]
        assert all(p == prefixes[0] for p in prefixes)
        for q in qs:
            raw = q.prompt().splitlines()
            edited = q.prompt((Intervention(Node.LAMP, 1),)).splitlines()
            differences = [(a, b) for a, b in zip(raw, edited, strict=True) if a != b]
            assert len(differences) == 1
            assert q.aliases[CircuitNode.LAMP] in differences[0][1]
            if q.node == CircuitNode.LAMP:
                assert q.answer((Intervention(Node.LAMP, 1),)) == 1
            else:
                assert q.answer((Intervention(Node.LAMP, 1),)) == q.answer()
    with pytest.raises(ValueError, match="test"):
        transfer_questions("train", "ru", family="chain")


def test_pair_metric_rejects_constant_consequences_and_keeps_invalid_answers():
    from semantics_operator.transfer_metrics import paired_metrics
    from semantics_operator.transfer_tasks import transfer_questions

    qs = transfer_questions("test", "en", family="chain")
    op = (Intervention(Node.RELAY, 1),)
    base = [q.answer() for q in qs]
    perfect = [q.answer(op) for q in qs]
    result = paired_metrics(perfect, qs, op, base)
    assert result["all_nodes_correct"] == {"count": 8, "correct": 8, "accuracy": 1.0}
    assert result["opposite_consequences"]["count"] == 8
    assert result["baseline_correct_pairs"]["count"] == 8
    biased = [
        0 if q.node in (CircuitNode.LAMP, CircuitNode.BRIDGE) else value
        for q, value in zip(qs, perfect, strict=True)
    ]
    result = paired_metrics(biased, qs, op, base)
    assert result["affected_nodes_correct"]["accuracy"] == 0
    assert result["all_nodes_correct"]["accuracy"] == 0
    assert result["opposite_consequences"]["accuracy"] == 0
    perfect[0] = None
    result = paired_metrics(perfect, qs, op, [None] * len(qs))
    assert result["all_nodes_correct"]["correct"] == 7
    assert result["baseline_correct_pairs"] == {"count": 0, "correct": 0, "accuracy": None}
    with pytest.raises(ValueError, match="pair|complete"):
        paired_metrics(perfect[:6], qs[:6], op, base[:6])
    with pytest.raises(ValueError, match="unique|duplicate"):
        paired_metrics(perfect + perfect[:1], qs + qs[:1], op, base + base[:1])


def test_transfer_selection_rewards_both_halves_and_alignment_requires_same_pairs():
    from semantics_operator.transfer_metrics import (
        paired_metrics,
        transfer_agreement,
        transfer_selection,
    )
    from semantics_operator.transfer_tasks import transfer_questions

    qs = transfer_questions("test", "en")
    op = (Intervention(Node.RELAY, 1),)
    base = [q.answer() for q in qs]
    target = [q.answer(op) for q in qs]
    pair = paired_metrics(target, qs, op, base)
    assert transfer_agreement(pair, pair)["joint_all_nodes_correct"]["correct"] == 8
    assert transfer_agreement(pair, pair)["baseline_correct_in_both"]["count"] == 8
    weak = paired_metrics([0] * len(qs), qs, op, base)
    assert transfer_agreement(weak, pair)["target_given_source_correct"]["accuracy"] is None
    with pytest.raises(ValueError, match="align"):
        transfer_agreement(pair, {**pair, "pairs": []})
    m = {
        "paired": pair,
        "protected_damage": {"rate": 0.1},
        "equation_consistency": {"all_satisfied": 0.5},
        "overall": {"accuracy": 0.9},
    }
    assert transfer_selection(m, 1)[0] == pytest.approx(0.9)


def test_transfer_cli_trains_only_source_and_saves_fixed_choices(tmp_path):
    from test_model import tiny_model

    from semantics_operator.cli import main

    lm = tiny_model()
    lm.model.save_pretrained(tmp_path / "weights")
    lm.tokenizer.save_pretrained(tmp_path / "weights")
    config = tmp_path / "config.toml"
    config.write_text(
        '[models.tiny]\npath="weights"\ndevice="cpu"\n[experiment]\noutput_dir="runs"\n'
    )
    env = tmp_path / ".env"
    env.write_text(f"SO_MODELS_DIR={tmp_path.as_posix()}\nSO_MODEL=tiny\n")
    assert (
        main(
            [
                "transfer",
                "--config",
                str(config),
                "--env-file",
                str(env),
                "--train-representation",
                "ru",
                "--layers",
                "0",
                "--steps",
                "2",
                "--rank",
                "2",
                "--strengths",
                "0",
                "--max-new-tokens",
                "1",
            ]
        )
        == 0
    )
    import json

    from safetensors.torch import load_file

    path = next((tmp_path / "runs").glob("*-transfer-*/report.json"))
    report = json.loads(path.read_text())
    assert report["train_representation"] == "ru"
    assert report["preservation_weight"] == 1.0
    assert report["answer_protocol"]["boundary"] == "last_token_inside_state_marker"
    assert report["selection_scope"] == "source_validation_only"
    assert report["site"]["path"] == "model.layers.0"
    assert report["rollback"]["max_score_difference"] == 0
    assert all("/ru/" in k for k in report["split"]["train"])
    assert all("/ru/" in k for k in report["split"]["validation"])
    assert len(report["test"]) == 12
    for choices in report["selected"].values():
        assert all(x["alpha"] == 0 for x in choices.values())
    for group in report["test"].values():
        assert "natural" in group["understanding"]
        for modes in group["operators"].values():
            assert modes["loreft_locality"]["greedy"] == modes["base"]["greedy"]
            assert modes["loreft_locality"]["alpha"] == 0
            assert "paired" in modes["loreft_locality"]["candidate"]
            assert "always_0" in modes and "always_1" in modes
    assert report["cross_representation"]
    assert load_file(path.parent / "operators.safetensors")
    assert (path.parent / "summary.md").is_file()


def test_transfer_rejects_last_state_block_and_invalid_strengths_before_training(tmp_path):
    from test_model import tiny_model

    from semantics_operator.config import Settings
    from semantics_operator.transfer_experiment import run_transfer

    lm = tiny_model()
    cfg = Settings("tiny", tmp_path, steps=1, rank=2)
    with pytest.raises(ValueError, match="final|earlier"):
        run_transfer(lm, cfg, layer=1)
    with pytest.raises(ValueError, match="finite"):
        run_transfer(lm, cfg, layer=0, strengths=[float("nan")])
    assert all(p.grad is None for p in lm.model.parameters())
    assert not lm.model.model.layers[0]._forward_hooks
