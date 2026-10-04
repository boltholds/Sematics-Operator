import pytest
import torch
from test_model import tiny_model

from semantics_operator.reft_localization import locate_block


def test_self_patch_uses_the_same_forward_layout_as_scoring():
    lm = tiny_model()
    block = lm.model.get_submodule("model.layers.0")

    def shape_dependent_rounding(module, args, output):
        # Emulate backend roundoff changing with the forward batch shape on CPU.
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[0] == 2:
            h = h.to(torch.bfloat16).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = block.register_forward_hook(shape_dependent_rounding)
    try:
        site, report = locate_block(lm, [0], preservation_weight=1)
        assert site == "model.layers.0"
        assert report["trials"][0]["self_patch_max_score_difference"] < 1e-5
        assert len(block._forward_hooks) == 1  # Keep the caller's hook only.
    finally:
        handle.remove()


@pytest.mark.parametrize("corruption", [0.01, float("nan"), float("inf")])
def test_self_patch_still_rejects_corruption_with_diagnostics(monkeypatch, corruption):
    import semantics_operator.reft_localization as localization

    original = localization.patched_scores

    def corrupt(*args, **kwargs):
        scores = original(*args, **kwargs)
        scores[:, 0] += corruption
        return scores

    monkeypatch.setattr(localization, "patched_scores", corrupt)
    lm = tiny_model()
    with pytest.raises(RuntimeError, match=r"block 0 .*max_score_difference="):
        locate_block(lm, [0], preservation_weight=1)
    assert not lm.model.get_submodule("model.layers.0")._forward_hooks
