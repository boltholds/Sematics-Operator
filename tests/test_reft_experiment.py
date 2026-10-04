import json

import pytest
from test_model import tiny_model

from semantics_operator.config import Settings


@pytest.mark.parametrize("position", ["state", "answer"])
def test_reft_suite_keeps_shifted_schemes_out_of_training_and_saves(tmp_path, position):
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
                "reft",
                "--reft-position",
                position,
                "--config",
                str(config),
                "--env-file",
                str(env),
                "--layers",
                "0",
                "1",
                "--max-new-tokens",
                "1",
                "--rank",
                "2",
                "--steps",
                "2",
                "--strengths",
                "0",
                "--pca-components",
                "2",
                "--locality-weight",
                "1",
                "--preservation-weight",
                "1",
            ]
        )
        == 0
    )
    path = next((tmp_path / "runs").glob("*-reft-*/report.json"))
    report = json.loads(path.read_text())
    assert report["locality_weight"] == report["preservation_weight"] == 1
    assert report["train_schemes"] == ["and_copy", "and_inverted"]
    assert report["loss_mode"] == "full_vocab"
    assert report["intervention_mode"] == "replace_equation"
    assert report["answer_protocol"]["training_completion_ids"] == [[2, 0], [3, 0]]
    assert report["test_groups"]["seen_mechanisms"] == ["and_copy", "and_inverted"]
    assert "and_xor_chain" in report["test_groups"]["held_out_mechanisms_or_topologies"]
    assert len(report["split"]["train"]) == 240
    assert len(report["split"]["validation"]) == 80
    assert all(
        key.startswith(("train_and_copy_", "train_and_inverted_"))
        for key in report["split"]["train"]
    )
    assert set(report["test"]) == {
        "and_copy",
        "or_copy",
        "and_gated",
        "and_chain",
        "and_inverted",
        "and_xor",
        "and_inverted_chain",
        "and_xor_chain",
    }
    assert set(report["training"]["relay_0"]) == {"loreft_task", "loreft_locality"}
    assert all(
        entry["alpha"] == 0 for choices in report["selected"].values() for entry in choices.values()
    )
    assert report["rollback"]["max_score_difference"] == 0
    assert report["experiment"] == "loreft_causal_suite_v5"
    assert report["answer_protocol"]["candidates"] == ["0", "1"]
    assert report["site"]["position"] == position
    assert report["prompt_layout"] == "state_first"
    assert report["state_prefix_checks"]["train"]["identical_prefix_per_state"]
    assert report["site"]["prefix"] == []
    location = report["localization"]
    assert location["candidate_layers"] == [0, 1]
    assert location["selected_layer"] in (0, 1)
    if position == "state":
        assert location["selected_layer"] == 0
    for trial in location["trials"]:
        assert trial["self_patch_max_score_difference"] < 1e-5
        assert all(
            record["key"].startswith(("validation_and_copy_", "validation_and_inverted_"))
            for m in trial["by_operator"].values()
            for record in m["records"]
        )
    assert set(report["greedy_test"]) == set(report["test"])
    for ops in report["greedy_test"].values():
        for modes in ops.values():
            assert modes["loreft_locality_selected"]["records"] == modes["base"]["records"]
            assert "incomplete" in modes["loreft_locality_fixed"]["overall"]
            assert "source" in modes["loreft_locality_fixed"]["protected_damage"]["by_node"]
            assert "relay" in modes["loreft_locality_fixed"]["protected_damage"]["by_node"]
            assert "equation_consistency" in modes["loreft_locality_fixed"]
    fixed = report["fixed_strength_diagnostics"]
    assert fixed["alpha"] == 1
    assert set(fixed["test"]) == set(report["test"])
    for split in (fixed["train"], fixed["validation"], *fixed["test"].values()):
        assert set(split) == {"relay_0", "relay_1", "lamp_1"}
        for methods in split.values():
            assert set(methods) == {"base", "loreft_task", "loreft_locality"}
            assert "by_label" in methods["loreft_locality"]
    assert any(
        methods["loreft_locality"]["records"] != methods["base"]["records"]
        for methods in fixed["validation"].values()
    )
    for method in report["training"]["relay_0"].values():
        assert method["batching"] == "full_state"
        assert all(len(batch) == 5 for batch in method["question_keys"])
        assert set(method["losses"][0]["by_node"]) == {"source", "switch", "relay", "lamp", "flag"}
    for scenarios in report["test"].values():
        assert "composition" in scenarios and "composition_reversed" in scenarios
        for methods in scenarios.values():
            assert methods["loreft_locality"]["records"] == methods["base"]["records"]
    folder = path.parent
    assert json.loads((folder / "report.json").read_text())["site"]["kind"] == "block"
    from safetensors.torch import load_file

    assert any(key.endswith("raw_basis") for key in load_file(folder / "operators.safetensors"))


def test_das_targets_are_base_counterfactuals_not_source_answers(tmp_path):
    from semantics_operator.causal_tasks import circuit_questions, interchange_pairs
    from semantics_operator.das_experiment import run_das

    pairs = interchange_pairs(circuit_questions("train"))
    assert len(pairs) == 640
    # A full donor carries unrelated source/flag values: must preserve recipient instead.
    assert any(p.expected != p.source.answer() for p in pairs)
    assert {p.intervention.value for p in pairs} == {0, 1}
    lm = tiny_model()
    report, tensors = run_das(lm, Settings("tiny", tmp_path, steps=2, rank=2), layer=0)
    assert report["experiment"] == "das_causal_alignment_v2"
    assert report["answer_protocol"]["candidates"] == ["0", "1"]
    assert report["site"]["prefix"] == []
    assert report["rollback"]["max_score_difference"] == 0
    for scheme, nodes in report["test"].items():
        for methods in nodes.values():
            assert {"base", "das", "random_subspace", "full_donor"} == set(methods)
            assert methods["das"]["count"] == (384 if scheme.endswith("chain") else 320)
            assert 0 <= methods["das"]["interchange_accuracy"] <= 1
            assert "both_natural_correct" in methods["das"]
    assert tensors
    assert all(p.grad is None for p in lm.model.parameters())
