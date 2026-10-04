"""Cross-pair directions: signed comparisons, valid anchors, and undefined zeros."""

import json

import pytest
import torch


def pair_metadata(index=0, *, token=9, position=2):
    return {
        "inputs": {
            "a": {"word": f"a{index}", "ids": [1] * position + [token]},
            "b": {"word": f"b{index}", "ids": [2] * position + [token]},
        },
        "anchor": {"a": position, "b": position},
        "layers": [{"index": 12, "key": "layer_012"}],
        "repeat_a_max_abs": 0.0,
    }


def test_direction_cosines_keep_sign_ignore_amplitude_and_mask_zero():
    from semantics_operator.heatmap_directions import compare_directions

    vectors = [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, 0.0]]
    pairs = [pair_metadata(i) for i in range(len(vectors))]
    report = compare_directions(pairs, [{12: torch.tensor(v)} for v in vectors])
    layer = report["layers"][0]
    assert layer["cosine"][0] == [1.0, 1.0, 0.0, -1.0, None]
    assert layer["cosine"][4] == [None] * 5
    assert layer["direction_status"] == ["ok", "ok", "ok", "ok", "zero_delta"]
    assert layer["delta_l2"] == [1.0, 3.0, 2.0, 1.0, 0.0]
    json.dumps(report, allow_nan=False)


def test_anchor_mismatch_and_missing_suffix_are_not_comparable():
    from semantics_operator.heatmap_directions import compare_directions

    pairs = [pair_metadata(), pair_metadata(1, token=8), pair_metadata(2)]
    pairs[-1]["anchor"] = None
    report = compare_directions(pairs, [{12: torch.ones(2)}, {12: torch.ones(2)}, {}])
    assert report["layers"][0]["cosine"][0][0] == pytest.approx(1.0)
    assert report["layers"][0]["cosine"][0][1:] == [None, None]
    assert report["comparisons"][0]["status"] == "different_anchor_token"
    assert report["comparisons"][1]["status"] == "no_shared_suffix"


def test_position_shifts_are_disclosed_and_small_noisy_directions_excluded():
    from semantics_operator.heatmap_directions import compare_directions

    pairs = [pair_metadata(), pair_metadata(1, position=4)]
    report = compare_directions(pairs, [{12: torch.ones(2)}] * 2)
    assert report["layers"][0]["cosine"][0][1] == pytest.approx(1.0)
    assert report["comparisons"][0]["same_anchor_positions"] is False
    assert report["warnings"]
    pairs[1]["repeat_a_max_abs"] = 0.1
    report = compare_directions(pairs, [{12: torch.ones(2)}, {12: torch.tensor([0.01, 0.0])}])
    assert report["layers"][0]["cosine"][0][1] is None
    assert report["layers"][0]["direction_status"][1] == "below_repeat_noise"


def test_nonfinite_direction_cannot_appear_as_agreement():
    from semantics_operator.heatmap_directions import compare_directions

    with pytest.raises(FloatingPointError, match="finite"):
        compare_directions([pair_metadata()], [{12: torch.tensor([float("nan"), 0.0])}])


def test_multilingual_gallery_adds_comparison_for_every_block(tmp_path):
    from gguf_factory import write_lfm2_gguf

    from semantics_operator.config import Settings
    from semantics_operator.heatmap_reporting import run_heatmaps
    from semantics_operator.model import LocalLanguageModel

    path = tmp_path / "tiny.gguf"
    write_lfm2_gguf(path)
    cfg = Settings("tiny-gguf", path, device="cpu", output_dir=tmp_path / "runs")
    with pytest.warns(UserWarning, match="dequantized"):
        lm = LocalLanguageModel.load(cfg)
    folder = run_heatmaps(lm, cfg, pairs=[("холодно", "жарко"), ("cold", "hot")])
    report = json.loads((folder / "report.json").read_text())
    cross = report["cross_pair_directions"]
    assert [l["index"] for l in cross["layers"]] == [0, 1]
    assert (folder / cross["image"]).is_file()
    assert (folder / cross["tensor_file"]).is_file()
    # Numeric direction must be recomputable from full raw captures.
    from safetensors.torch import load_file

    vectors = []
    for pair in report["pairs"]:
        ts = load_file(folder / pair["tensor_file"])
        vectors.append(
            (
                ts["layer_001.b"][pair["anchor"]["b"]] - ts["layer_001.a"][pair["anchor"]["a"]]
            ).double()
        )
    expected = float(torch.nn.functional.cosine_similarity(*vectors, dim=0))
    assert cross["layers"][1]["cosine"][0][1] == pytest.approx(expected)
