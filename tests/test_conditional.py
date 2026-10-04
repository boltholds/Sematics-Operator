import torch
from test_model import tiny_model

from semantics_operator.conditional import fit_map, predict, run_conditional
from semantics_operator.config import Settings


def test_conditional_map_learns_variable_transition_constant_cannot():
    x = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    delta = torch.stack([x[:, 0], -x[:, 0]], dim=1)
    model = fit_map(x, delta, ridge=0.001)
    unseen = torch.tensor([[0.5, 0.0], [-0.5, 0.0]])
    expected = torch.stack([unseen[:, 0], -unseen[:, 0]], dim=1)
    assert (predict(model, unseen) - expected).square().mean() < 0.001
    assert (model["mean_delta"] - expected).square().mean() > 0.1
    assert torch.isfinite(predict(fit_map(torch.ones_like(x), delta), unseen)).all()


def test_conditional_experiment_preserves_weights_and_has_equal_selection_budget(tmp_path):
    torch.set_num_threads(1)
    lm = tiny_model()
    before = {k: v.clone() for k, v in lm.model.state_dict().items()}
    report, tensors = run_conditional(lm, Settings("tiny", tmp_path), strengths=[0.0, 1.0])
    assert report["rollback"]["max_score_difference"] == 0
    for methods in report["validation_trials"].values():
        assert len({len(trials) for trials in methods.values()}) == 1
        assert set(methods) == {"constant", "conditional", "shuffled"}
    assert tensors
    for s in report["scenarios"].values():
        assert {"base", "constant", "conditional", "shuffled", "explicit_prompt"} == set(s)
    for k, v in before.items():
        assert torch.equal(v, lm.model.state_dict()[k])


def test_pca_prediction_is_train_basis_projection_of_full_map():
    from semantics_operator.conditional import fit_pca_map

    generator = torch.Generator().manual_seed(10)
    x = torch.randn(12, 6, generator=generator)
    y = torch.randn(12, 6, generator=generator)
    query = torch.randn(3, 6, generator=generator)
    full = fit_map(x, y)
    pca = fit_pca_map(x, y, components=2)
    basis = pca["basis"].float()
    expected = (predict(full, query) - full["mean_delta"].float()) @ basis.T @ basis + full[
        "mean_delta"
    ].float()
    assert torch.allclose(predict(pca, query), expected, atol=1e-6)
    assert pca["coefficients"].shape == (12, 2)
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-6)
    zero = fit_pca_map(x, torch.ones_like(y), components=8)
    assert zero["basis"].shape == (0, 6)
    assert torch.equal(predict(zero, query), torch.ones(3, 6))


def test_pca_selection_and_privileged_diagnostic_are_separate(tmp_path):
    lm = tiny_model()
    report, tensors = run_conditional(
        lm, Settings("tiny", tmp_path), strengths=[0.0], pca_components=[2, 4]
    )
    for choices in report["selected"].values():
        assert choices["pca_selected"]["variant"] == "pca_2"
    assert report["oracle_diagnostics"]
    for trials in report["validation_trials"].values():
        assert "oracle" not in trials
    assert any(k.endswith(".basis") for k in tensors)
    for methods in report["oracle_diagnostics"].values():
        assert (
            methods["rollback"]["overall"]["accuracy"]
            == report["scenarios"]["rollback"]["base"]["overall"]["accuracy"]
        )


def test_exact_donor_replacement_preserves_identical_prompt_scores():
    from semantics_operator.conditional import row_scores
    from semantics_operator.steering import evaluate
    from semantics_operator.world import questions

    lm = tiny_model()
    samples = questions("test")[:3]
    layer = lm.choose_target("")
    donor = lm.representations([q.prompt() for q in samples], layer)
    original = evaluate(lm, samples, {})
    actual = row_scores(lm, samples, {layer: donor}, replace=True)
    assert torch.allclose(actual, original, atol=1e-6)
    assert not lm.model.get_submodule(layer)._forward_hooks


def test_protected_damage_penalizes_only_new_errors_on_protected_nodes():
    from semantics_operator.conditional import penalized_selection, protected_damage
    from semantics_operator.world import OPERATORS, questions

    samples = questions("test")
    baseline = torch.tensor([[1 - q.answer(), q.answer()] for q in samples]).float()
    damaged = baseline.clone()
    damaged[0] = damaged[0].flip(0)  # source
    damaged[2] = damaged[2].flip(0)  # relay: not a protected node
    metric = protected_damage(damaged, samples, baseline)
    assert metric["damaged"] == 1
    assert metric["eligible"] == 24
    assert metric["by_node"]["source"]["damaged"] == 1
    plain = penalized_selection(damaged, samples, OPERATORS[0], baseline, 0)
    penalized = penalized_selection(damaged, samples, OPERATORS[0], baseline, 2)
    assert abs(plain[0] - penalized[0] - 2 / 24) < 1e-7
    assert protected_damage(damaged, samples, damaged)["damaged"] == 0


def test_block_decision_comparison_zero_and_terminal_oracle(tmp_path):
    lm = tiny_model()
    original = lm.tokenizer

    class Tokenizer:
        def __getattr__(self, name):
            return getattr(original, name)

        def encode(self, text, **kwargs):
            if text in (" 0", " 1"):
                return [4, 2 if text == " 0" else 3]
            return original.encode(text, **kwargs)

    lm.tokenizer = Tokenizer()
    before = {k: v.clone() for k, v in lm.model.state_dict().items()}
    report, _ = run_conditional(
        lm,
        Settings("tiny", tmp_path),
        layers=["1"],
        site_kind="block",
        boundary="decision",
        preservation_weight=1,
        strengths=[0],
        pca_components=[2],
    )
    assert report["site"]["common_candidate_prefix"] == [4]
    assert report["layers"] == ["model.layers.1"]
    for name, methods in report["scenarios"].items():
        for method in ("constant", "conditional", "pca_2"):
            assert methods[method]["records"] == methods["base"]["records"]
            assert methods[method]["protected_damage"]["damaged"] == 0
        donor = report["oracle_diagnostics"]["model.layers.1"][name]["records"]
        explicit = methods["explicit_prompt"]["records"]
        assert max(abs(a["p1"] - b["p1"]) for a, b in zip(donor, explicit)) < 1e-5
    assert report["rollback"]["max_score_difference"] == 0
    for k, v in before.items():
        assert torch.equal(v, lm.model.state_dict()[k])
    assert not lm.model.get_submodule("model.layers.1")._forward_hooks
