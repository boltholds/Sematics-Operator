import hashlib

import pytest
import torch
from gguf_factory import write_lfm2_gguf

from semantics_operator.config import Settings
from semantics_operator.model import LocalLanguageModel
from semantics_operator.weights import WeightSession


@pytest.mark.parametrize("quantized", [True, False])
def test_standalone_lfm2_gguf_roundtrip_gradients_and_exact_rollback(tmp_path, quantized):
    torch.set_num_threads(1)
    path = tmp_path / "tiny.gguf"
    reference, expected = write_lfm2_gguf(path, quantized=quantized)
    original_file = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.warns(UserWarning, match="dequantized"):
        lm = LocalLanguageModel.load(Settings("gguf", path, device="cpu"))
    assert lm.model.config.model_type == "lfm2"
    assert lm.model.config.layer_types == ["conv", "full_attention"]
    assert lm.model.config.norm_eps == pytest.approx(0.003)
    assert lm.model.config.eos_token_id == 2
    for name, tensor in lm.model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    target = lm.choose_target("")
    assert target.endswith("feed_forward.w2")
    assert lm.tokenizer.encode("ab", add_special_tokens=True)[0] == 1
    assert lm.tokenizer.eos_token_id == 2
    assert lm.tokenizer.pad_token_id == 0
    assert lm.tokenizer.encode("<start>", add_special_tokens=False) == [4]
    assert lm.tokenizer.encode("ab", add_special_tokens=False) == [262]
    assert "<start>assistant" in lm.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    reference_lm = LocalLanguageModel(reference, lm.tokenizer, 512)
    prompts = ["source=1. Answer:", "source=0. Answer:"]
    before = lm.scores(prompts).detach()
    torch.testing.assert_close(before, reference_lm.scores(prompts), rtol=1e-5, atol=1e-5)
    with WeightSession(lm.model, target, rank=2) as session:
        optimizer = torch.optim.Adam(session.parameters(), lr=0.01)
        loss = -lm.scores(prompts)[:, 1].sum()
        loss.backward()
        assert session.layer.b.grad is not None
        assert session.layer.b.grad.abs().sum() > 0
        optimizer.step()
        # Small but real updates must not be hidden by allclose relative tolerance.
        assert (lm.scores(prompts) - before).abs().max() > 1e-6
    torch.testing.assert_close(lm.scores(prompts), before, rtol=0, atol=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_file


def test_missing_gguf_weight_is_rejected_not_randomly_initialized(tmp_path):
    path = tmp_path / "broken.gguf"
    write_lfm2_gguf(path, omit="blk.1.ffn_down.weight")
    with (
        pytest.warns(UserWarning, match="dequantized"),
        pytest.raises(ValueError, match="incomplete GGUF weights"),
    ):
        LocalLanguageModel.load(Settings("gguf", path, device="cpu"))


def test_unknown_architecture_and_missing_file_are_actionable(tmp_path):
    import gguf

    path = tmp_path / "unsupported.gguf"
    writer = gguf.GGUFWriter(path, "unsupported")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="Unsupported GGUF architecture"):
        LocalLanguageModel.load(Settings("gguf", path, device="cpu"))
    with pytest.raises(ValueError, match="GGUF file"):
        LocalLanguageModel.load(Settings("gguf", tmp_path / "absent.gguf", device="cpu"))


def test_gguf_cli_full_experiment(tmp_path):
    import json

    from semantics_operator.cli import main

    torch.set_num_threads(1)
    checkpoint = tmp_path / "tiny.gguf"
    write_lfm2_gguf(checkpoint)
    env = tmp_path / ".env"
    env.write_text(f'SO_MODELS_DIR="{tmp_path.as_posix()}"\nSO_MODEL=gguf\n')
    config = tmp_path / "test.toml"
    config.write_text(
        '[models.gguf]\npath="tiny.gguf"\ndevice="cpu"\n'
        '[experiment]\nsteps=1\nrank=2\noutput_dir="runs"\n'
    )
    with pytest.warns(UserWarning, match="dequantized"):
        assert main(["run", "--config", str(config), "--env-file", str(env)]) == 0
    [report_path] = (tmp_path / "runs").glob("*/report.json")
    report = json.loads(report_path.read_text())
    assert report["model"]["checkpoint_format"] == "gguf"
    assert report["model"]["dequantized"] is True
    assert report["rollback"]["max_score_difference"] == 0
    assert set(report["scenarios"]) >= {"composition", "revision", "rollback"}
