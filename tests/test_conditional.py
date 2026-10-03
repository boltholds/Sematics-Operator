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
