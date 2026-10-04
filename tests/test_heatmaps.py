import json

import pytest
import torch
from safetensors.torch import load_file
from test_model import tiny_model


def test_alignment_masks_unequal_replacements_and_preserves_suffix():
    from semantics_operator.heatmaps import align_tokens, aligned_matrices

    rows = align_tokens([1, 2, 3, 9], [1, 7, 8, 3, 9])
    assert [(r["a"], r["b"], r["kind"]) for r in rows] == [
        (0, 0, "prefix"),
        (1, None, "unmatched_a"),
        (None, 1, "unmatched_b"),
        (None, 2, "unmatched_b"),
        (2, 3, "suffix"),
        (3, 4, "suffix"),
    ]
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    b = torch.tensor([[1.0, 2.0], [4.0, 5.0], [6.0, 7.0], [10.0, 12.0], [9.0, 6.0]])
    _, _, delta = aligned_matrices(a, b, rows)
    assert torch.isnan(delta[1:4]).all()
    torch.testing.assert_close(
        delta[[0, 4, 5]], torch.tensor([[0.0, 0.0], [5.0, 6.0], [2.0, -2.0]])
    )
    assert align_tokens([1, 2, 9], [1, 8, 9])[1]["kind"] == "replacement"


def test_capture_all_blocks_has_causal_prefix_and_identity_control():
    from semantics_operator.heatmaps import compare_pair

    lm = tiny_model()
    before = {n: p.clone() for n, p in lm.model.named_parameters()}
    report, tensors = compare_pair(lm, ("0", "1"), template="source = {word}. Answer:")
    assert [s["module"] for s in report["layers"]] == ["model.layers.0", "model.layers.1"]
    assert report["repeat_a_max_abs"] == 0
    assert report["anchor"]["a"] == report["anchor"]["b"] == 5
    # Prefix source/= cannot depend on a future value in a causal model.
    for index in range(2):
        delta = tensors[f"layer_{index:03d}.delta"]
        assert torch.count_nonzero(delta[:2]) == 0
        assert torch.count_nonzero(delta[2:]) > 0
        torch.testing.assert_close(
            delta, tensors[f"layer_{index:03d}.b"] - tensors[f"layer_{index:03d}.a"]
        )
    for name, p in lm.model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in lm.model.modules())
    same, zero = compare_pair(lm, ("0", "0"), template="source = {word}. Answer:")
    assert same["identical_token_ids"]
    assert all(torch.count_nonzero(t) == 0 for k, t in zero.items() if k.endswith(".delta"))


def test_chat_format_and_no_shared_suffix_are_explicit():
    from semantics_operator.heatmaps import compare_pair

    lm = tiny_model()
    raw, _ = compare_pair(lm, ("0", "1"), template="{word}")
    assert raw["anchor"] is None
    lm.tokenizer.chat_template = 'source: {{ messages[0]["content"] }} Answer:'
    chat, _ = compare_pair(lm, ("0", "1"), template="{word}", prompt_format="chat")
    assert chat["inputs"]["a"]["ids"] == lm._prompt_ids("0")
    assert chat["anchor"]["a"] == len(lm._prompt_ids("0")) - 1


@pytest.mark.parametrize("template", ["no placeholder", "{word} {word}", "{word!r}", "{other}"])
def test_template_validation(template):
    from semantics_operator.heatmaps import compare_pair

    with pytest.raises(ValueError, match="template"):
        compare_pair(tiny_model(), ("0", "1"), template=template)


def test_capture_cleanup_after_error_and_nonfinite_rejection():
    from semantics_operator.heatmaps import compare_pair

    lm = tiny_model()
    lm.max_length = 1
    with pytest.raises(ValueError, match="max_length"):
        compare_pair(lm, ("0", "1"))
    assert all(not m._forward_hooks for m in lm.model.modules())
    lm.max_length = 512
    with torch.no_grad():
        lm.model.model.embed_tokens.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="finite"):
        compare_pair(lm, ("0", "1"), template="{word}")
    assert all(not m._forward_hooks for m in lm.model.modules())


def test_gguf_hybrid_blocks_with_unequal_token_counts(tmp_path):
    from gguf_factory import write_lfm2_gguf

    from semantics_operator.config import Settings
    from semantics_operator.heatmaps import compare_pair
    from semantics_operator.model import LocalLanguageModel

    path = tmp_path / "tiny.gguf"
    write_lfm2_gguf(path)
    with pytest.warns(UserWarning, match="dequantized"):
        lm = LocalLanguageModel.load(Settings("gguf", path, device="cpu"))
    report, tensors = compare_pair(lm, ("a", "xyz"), template="Value: {word}. Result:")
    assert len(report["layers"]) == 2  # convolution block AND attention block
    assert len(report["inputs"]["a"]["ids"]) != len(report["inputs"]["b"]["ids"])
    assert any("positional" in w for w in report["warnings"])
    for layer in report["layers"]:
        d = tensors[layer["key"] + ".delta"]
        assert torch.isnan(d).any()
    # The local conv cannot see past the long common suffix; attention can.
    assert report["layers"][0]["anchor_metrics"]["delta_rms"] == 0
    assert report["layers"][1]["anchor_metrics"]["delta_rms"] > 0
    assert report["repeat_a_max_abs"] == 0


def test_cli_saves_all_layers_raw_data_and_offline_gallery(tmp_path, monkeypatch, capsys):
    from PIL import Image

    from semantics_operator.cli import main
    from semantics_operator.config import Settings
    from semantics_operator.model import LocalLanguageModel

    lm = tiny_model()
    monkeypatch.setattr(LocalLanguageModel, "load", lambda _: lm)
    monkeypatch.setattr(
        "semantics_operator.cli.load_settings",
        lambda *a: Settings("tiny", tmp_path, output_dir=tmp_path / "runs"),
    )
    assert (
        main(
            [
                "heatmap",
                "--pair",
                "0",
                "1",
                "--pair",
                "0",
                "0",
                "--template",
                "source = {word}. Answer:",
            ]
        )
        == 0
    )
    folder = next((tmp_path / "runs").glob("*-heatmap-*"))
    report = json.loads((folder / "report.json").read_text())
    assert len(report["pairs"]) == 2
    first, second = report["pairs"]
    assert len(first["layers"]) == 2
    for pair in (first, second):
        data = load_file(str(folder / pair["tensor_file"]))
        assert len(data) == 6
        for layer in pair["layers"]:
            for path in layer["images"]:
                with Image.open(folder / path) as im:
                    assert im.width >= 1000 and im.height >= 500
        assert (folder / pair["overview_image"]).is_file()
        assert (folder / pair["metrics_image"]).is_file()
    assert second["scales"]["delta_max_abs"] == 0
    assert (folder / "index.html").is_file()
    assert "index.html" in capsys.readouterr().out


def test_png_keeps_one_pixel_per_hidden_coordinate_after_labels(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from semantics_operator.heatmap_reporting import _triptych

    observed = []
    save = Figure.savefig

    def measure(figure, *args, **kwargs):
        figure.canvas.draw()
        observed.extend(ax.get_window_extent().width for ax in figure.axes[:3])
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", measure)
    a = torch.arange(2048).float().expand(2, -1)
    _triptych(
        tmp_path / "wide.png",
        (a, a, a * 0),
        ["A:12 source | B:12 source [replacement]"] * 2,
        "Wide hidden state",
        {"activation_max_abs": 2047, "delta_max_abs": 0},
    )
    assert min(observed) >= 2048


def test_progress_can_be_redirected_to_windows_ansi_log(tmp_path):
    import io

    from semantics_operator.config import Settings
    from semantics_operator.heatmap_reporting import run_heatmaps

    buffer = io.BytesIO()
    log = io.TextIOWrapper(buffer, encoding="cp1251")
    folder = run_heatmaps(
        tiny_model(),
        Settings("tiny", tmp_path, output_dir=tmp_path / "runs"),
        pairs=[("source", "switch")],
        template="{word}",
        layers=[0],
        progress=lambda message: print(message, file=log),
    )
    log.flush()
    assert buffer.getvalue()
    assert (folder / "report.json").is_file()
