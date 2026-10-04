import json

import pytest
import torch
from test_model import tiny_model


def test_edit_stays_at_original_anchor_during_continuation_and_cleans_up():
    from semantics_operator.word_transfer_runtime import edit_at

    lm = tiny_model()
    site = "model.layers.0"
    prefix = lm.tokenizer.encode("source = 0. Answer:")
    position = len(prefix) - 1
    vector = torch.arange(16).float() / 10
    block = lm.model.get_submodule(site)
    for continuation in ([], [2, 3]):
        captured = []
        handle = block.register_forward_hook(lambda m, a, o, c=captured: c.append(o.clone()))
        ids, mask = lm._batch([prefix + continuation])
        with edit_at(lm, site, position, vector):
            after = []
            observer = block.register_forward_hook(lambda m, a, o, c=after: c.append(o.clone()))
            lm.model(input_ids=ids, attention_mask=mask, use_cache=False)
            observer.remove()
        handle.remove()
        difference = after[0] - captured[0]
        torch.testing.assert_close(difference[0, position], vector)
        difference[0, position] = 0
        assert torch.count_nonzero(difference) == 0
    with pytest.raises(RuntimeError, match="deliberate"), edit_at(lm, site, position, vector):
        raise RuntimeError("deliberate")
    assert not block._forward_hooks


def test_raw_scoring_matches_manual_multitoken_likelihood_despite_chat_template():
    from semantics_operator.word_transfer_runtime import measure

    lm = tiny_model()
    lm.tokenizer.chat_template = 'unrelated {{ messages[0]["content"] }} source'
    prefix = lm.tokenizer.encode("source = 0. Answer:")
    candidates = ("0 1", "1")
    before = {k: v.clone() for k, v in lm.model.state_dict().items()}
    result, logp = measure(
        lm,
        prefix,
        "model.layers.0",
        len(prefix) - 1,
        None,
        candidates=candidates,
        max_new_tokens=2,
    )
    for i, candidate in enumerate(candidates):
        suffix = lm.tokenizer.encode(candidate, add_special_tokens=False)
        logits = lm.model(torch.tensor([prefix + suffix]), use_cache=False).logits[0].detach()
        expected = sum(
            float(logits[len(prefix) + j - 1].log_softmax(-1)[token])
            for j, token in enumerate(suffix)
        )
        assert result["candidate_logp"][i] == pytest.approx(expected, abs=1e-5)
    assert logp.exp().sum().item() == pytest.approx(1)
    zero, _ = measure(
        lm,
        prefix,
        "model.layers.0",
        len(prefix) - 1,
        torch.zeros(16),
        candidates=candidates,
        max_new_tokens=2,
        baseline_logp=logp,
    )
    assert zero["generation"] == result["generation"]
    assert zero["first_token_kl"] == pytest.approx(0, abs=1e-6)
    for k, v in lm.model.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)


def test_controls_are_reproducible_norm_matched_and_do_not_change_global_rng():
    from semantics_operator.word_transfer import make_directions

    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([4.0, 3.0, 2.0, 1.0])
    rng = torch.random.get_rng_state()
    directions = make_directions(a, b, seed=42)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    again = make_directions(a, b, seed=42)
    assert len([k for k in directions if k.startswith("random_")]) == 3
    for key in directions:
        torch.testing.assert_close(directions[key], again[key])
        if key.startswith("random_"):
            assert directions[key].norm() == pytest.approx(a.norm())
    with pytest.raises(ValueError, match="zero"):
        make_directions(torch.zeros(4), b, seed=42)


def test_metrics_require_both_directions_and_only_count_baseline_correct_damage():
    from semantics_operator.word_transfer import summarize_cases

    def result(prediction, complete=True):
        return {
            "prediction": prediction,
            "generation": {"answer": prediction, "complete": complete},
            "first_token_kl": 0.1,
        }

    cases = [
        {"id": "x_cold", "context": "x", "state": 0},
        {"id": "x_hot", "context": "x", "state": 1},
    ]
    base = {
        "x_cold": {"temperature": result(0), "color": result(0)},
        "x_hot": {"temperature": result(1), "color": result(1)},
    }
    trial = {
        "x_cold": {"temperature": result(1), "color": result(1)},
        "x_hot": {"temperature": result(1), "color": result(0)},
    }
    expected = {"x_cold": {"temperature": 0, "color": 0}, "x_hot": {"temperature": 1, "color": 0}}
    metrics = summarize_cases(cases, trial, base, expected)
    for mode in ("candidate", "greedy"):
        m = metrics[mode]
        assert m["temperature"]["correct"] == 1
        assert m["protected_damage"] == {"damaged": 1, "eligible": 1}
        assert m["paired_joint"] == {"correct": 0, "count": 1}
    trial["x_hot"]["temperature"] = result(0, complete=False)
    metrics = summarize_cases(cases, trial, base, expected)
    assert metrics["candidate"]["temperature"]["correct"] == 2
    assert metrics["greedy"]["temperature"]["correct"] == 1


def test_word_transfer_saves_full_evidence_with_raw_anchor_before_questions(tmp_path):
    from semantics_operator.config import Settings
    from semantics_operator.word_transfer import run_word_transfer
    from semantics_operator.word_transfer_reporting import save_word_transfer

    lm = tiny_model()
    # Use a real tiny transformer with a vocabulary covering both lexical contrasts.
    lm.tokenizer.add_tokens(
        ["холодно", "жарко", "cold", "hot", "blue", "red", "cup", "bowl", "two", "three"]
    )
    lm.model.resize_token_embeddings(len(lm.tokenizer), mean_resizing=False)
    cfg = Settings("tiny", tmp_path, seed=42)
    before = {k: v.clone() for k, v in lm.model.state_dict().items()}
    report, tensors = run_word_transfer(
        lm, cfg, layer=0, strengths=[0, 1.0000001], max_new_tokens=1
    )
    assert report["site"]["layer"] == 0
    assert report["primary_condition"] == "ru_delta@1"
    assert "ru_delta@1" in report["conditions"]
    assert report["conditions"]["ru_delta@1"]["alpha"] == 1.0
    assert any(c["alpha"] == 1.0000001 for c in report["conditions"].values())
    assert report["rollback_max_logp_difference"] < 1e-5
    assert len(report["cases"]) == 4
    for case in report["cases"]:
        for q in case["questions"].values():
            assert q["ids"][: len(case["prefix_ids"])] == case["prefix_ids"]
            assert case["anchor_position"] < len(q["ids"]) - 1
    folder = save_word_transfer(tmp_path, report, tensors)
    saved = json.loads((folder / "report.json").read_text())
    assert saved["conditions"]["ru_delta@1"]["metrics"]
    assert (folder / "index.html").is_file()
    assert (folder / "directions.safetensors").is_file()
    for k, v in lm.model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in lm.model.modules())


def test_word_transfer_cli_runs_and_rejects_silent_protocol_overrides(tmp_path, capsys):
    from semantics_operator.cli import main

    lm = tiny_model()
    lm.tokenizer.add_tokens(
        ["холодно", "жарко", "cold", "hot", "blue", "red", "cup", "bowl", "two", "three"]
    )
    lm.model.resize_token_embeddings(len(lm.tokenizer), mean_resizing=False)
    lm.model.save_pretrained(tmp_path / "weights")
    lm.tokenizer.save_pretrained(tmp_path / "weights")
    config = tmp_path / "config.toml"
    config.write_text(
        '[models.test]\npath="weights"\ndevice="cpu"\n[experiment]\noutput_dir="runs"\n'
    )
    env = tmp_path / ".env"
    env.write_text(f"SO_MODELS_DIR={tmp_path.as_posix()}\nSO_MODEL=test\n")
    args = ["word-transfer", "--config", str(config), "--env-file", str(env)]
    assert main([*args, "--layers", "0", "1"]) == 2
    assert "one predeclared block" in capsys.readouterr().err
    assert main([*args, "--prompt-format", "chat"]) == 2
    assert "raw heatmap template" in capsys.readouterr().err
    assert main([*args, "--layers", "0", "--max-new-tokens", "1"]) == 0
    path = next((tmp_path / "runs").glob("*-word-transfer-*/report.json"))
    report = json.loads(path.read_text())
    assert report["site"]["layer"] == 0
    assert report["primary_condition"] == "ru_delta@1"


def test_word_transfer_rejects_eos_in_place_of_the_colon_anchor(tmp_path):
    from tokenizers.processors import TemplateProcessing

    from semantics_operator.config import Settings
    from semantics_operator.word_transfer import run_word_transfer

    lm = tiny_model()
    lm.tokenizer.add_tokens(
        ["холодно", "жарко", "cold", "hot", "blue", "red", "cup", "bowl", "two", "three"]
    )
    lm.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
    lm.model.resize_token_embeddings(len(lm.tokenizer), mean_resizing=False)
    lm.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="$A [EOS]", special_tokens=[("[EOS]", lm.tokenizer.eos_token_id)]
    )
    with pytest.raises(ValueError, match="colon"):
        run_word_transfer(lm, Settings("tiny", tmp_path), layer=0, max_new_tokens=1)
    assert all(not m._forward_hooks for m in lm.model.modules())
