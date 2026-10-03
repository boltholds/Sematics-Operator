import torch
from test_model import tiny_model

from semantics_operator.config import Settings
from semantics_operator.experiment import resolve_sequence, run_experiment
from semantics_operator.world import Intervention, Node


def test_revision_replaces_old_assumption_but_composition_retains_other_nodes():
    r0, r1 = Intervention(Node.RELAY, 0), Intervention(Node.RELAY, 1)
    l1 = Intervention(Node.LAMP, 1)
    assert resolve_sequence((r0, l1, r1)) == (r1, l1)


def test_full_real_transformer_experiment_preserves_base_and_holds_out_test(tmp_path):
    torch.set_num_threads(1)
    lm = tiny_model()
    original = {k: v.clone() for k, v in lm.model.state_dict().items()}
    cfg = Settings(
        "tiny-test",
        tmp_path,
        target_module="model.layers.0.mlp.down_proj",
        steps=2,
        rank=2,
        output_dir=tmp_path,
    )
    report, patches, representations = run_experiment(lm, cfg)
    assert report["rollback"]["max_score_difference"] == 0
    assert report["rollback"]["base_weight_unchanged"]
    assert report["split"]["overlap"] == 0
    assert set(patches) == {"relay_0", "relay_1", "lamp_1"}
    assert "composition_reversed" in report["scenarios"]
    assert len(report["scenarios"]["relay_0"]["weight_edit"]["records"]) == 40
    assert representations["baseline"].shape == (8, 16)
    assert any(p.norm() > 0 for p in patches.values())
    assert not torch.equal(patches["relay_1"].b, patches["lamp_1"].b)
    for stats in report["training_sampling"]["operators"].values():
        assert stats["changed_presentations"] == cfg.steps
        assert stats["unchanged_presentations"] == cfg.steps
        assert all(n > 0 for n in stats["distinguishing_presentations"].values())
    for key, value in lm.model.state_dict().items():
        assert torch.equal(value, original[key])
    record = report["scenarios"]["relay_0"]["weight_edit"]["records"][0]
    assert {"expected", "base_expected", "prediction", "changed", "p1", "key"} <= record.keys()
