import torch
from test_model import tiny_model

from semantics_operator.config import Settings
from semantics_operator.localization import (
    capture_tail,
    common_prefix,
    patched_scores,
    run_localization,
)


def test_shared_answer_prefix_and_terminal_block_transfer():
    lm = tiny_model()
    original_tokenizer = lm.tokenizer

    class Tokenizer:
        def __getattr__(self, key):
            return getattr(original_tokenizer, key)

        def encode(self, text, **kwargs):
            if text in (" 0", " 1"):
                return [4, 2 if text == " 0" else 3]
            return original_tokenizer.encode(text, **kwargs)

    lm.tokenizer = Tokenizer()
    prefix = common_prefix(lm)
    assert prefix == [4]
    source = ["source = 1. Answer:"]
    target = ["source = 0. Answer:"]
    site = "model.layers.1"
    donor = capture_tail(lm, source, [site], 2, prefix)
    transferred = patched_scores(lm, target, donor, [site], 2, prefix)
    assert torch.allclose(
        transferred.softmax(-1), lm.scores(source).softmax(-1).detach().cpu(), atol=1e-6
    )
    neutral = capture_tail(lm, target, [site], 2, prefix)
    added = patched_scores(
        lm, target, {site: donor[site] - neutral[site]}, [site], 2, prefix, replace=False
    )
    assert torch.allclose(added, transferred, atol=1e-6)
    assert torch.allclose(
        patched_scores(lm, target, neutral, [site], 2, prefix),
        lm.scores(target).detach().cpu(),
        atol=1e-6,
    )
    assert not lm.model.get_submodule(site)._forward_hooks


def test_localization_selects_groups_on_validation_and_checks_rollback(tmp_path):
    lm = tiny_model()
    report = run_localization(
        lm,
        Settings("tiny", tmp_path),
        layer_sets=[(0,), (1,), (0, 1)],
        windows=[1],
        boundaries=["decision"],
    )
    assert len(report["validation_trials"]["relay_0"]) == 6
    assert report["rollback"]["max_score_difference"] == 0
    assert all(
        "selected" in m and "deepest_only" in m and "self_patch" in m
        for m in report["test"].values()
    )
