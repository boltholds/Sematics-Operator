import torch
from test_model import tiny_model

from semantics_operator.config import Settings
from semantics_operator.steering import consistency, run_steering, save_steering, steered_scores
from semantics_operator.world import OPERATORS, questions


def test_steering_changes_scores_and_restores_model_even_on_error():
    lm = tiny_model()
    target = "model.layers.0.mlp.down_proj"
    prompts = [q.prompt() for q in questions("test")[:2]]
    before = lm.scores(prompts).detach()
    vector = torch.arange(16).float() / 10
    after = steered_scores(lm, prompts, {target: vector})
    assert not torch.allclose(before, after)
    assert torch.equal(before, lm.scores(prompts).detach())
    try:
        steered_scores(lm, prompts, {target: torch.zeros(7)})
    except RuntimeError:
        pass
    assert not lm.model.get_submodule(target)._forward_hooks


def test_consistency_distinguishes_relation_from_correctness():
    samples = questions("test")
    scores = torch.tensor([[0.0, 1.0]] * 40)
    result = consistency(scores, samples, (OPERATORS[0],))
    assert result["relay_lamp_correct"]["accuracy"] == 0
    assert result["lamp_mechanism_satisfied"]["accuracy"] == 1
    result = consistency(scores, samples, (OPERATORS[0], OPERATORS[2]))
    assert result["relay_lamp_correct"]["accuracy"] == 0
    assert result["lamp_mechanism_satisfied"]["accuracy"] == 1


def test_real_experiment_selects_on_validation_and_saves_controls(tmp_path):
    torch.set_num_threads(1)
    lm = tiny_model()
    original = {k: v.clone() for k, v in lm.model.state_dict().items()}
    report, vectors = run_steering(
        lm,
        Settings("tiny", tmp_path),
        layers=["model.layers.0.mlp.down_proj"],
        strengths=[0.0, 1.0],
    )
    assert report["rollback"]["max_score_difference"] == 0
    assert report["split"]["overlap"] == 0
    assert len(vectors) == 3
    for scenario in report["scenarios"].values():
        assert {
            "base",
            "steering",
            "random_norm_matched",
            "opposite",
            "explicit_prompt",
        } <= scenario.keys()
        assert "consistency" in scenario["steering"]
    for k, v in original.items():
        assert torch.equal(v, lm.model.state_dict()[k])


def test_artifacts_roundtrip(tmp_path):
    import json

    from safetensors.torch import load_file

    lm = tiny_model()
    report, vectors = run_steering(
        lm, Settings("tiny", tmp_path), layers=["model.layers.0.mlp.down_proj"], strengths=[0.0]
    )
    folder = save_steering(tmp_path, report, vectors)
    assert json.loads((folder / "report.json").read_text())["selected"] == report["selected"]
    assert set(load_file(folder / "vectors.safetensors")) == set(vectors)
    assert "Pair correct" in (folder / "summary.md").read_text()
