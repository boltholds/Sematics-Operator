from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from semantics_operator.model import LocalLanguageModel


def tiny_model():
    torch.manual_seed(5)
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "0": 2,
        "1": 3,
        "Answer": 4,
        ":": 5,
        "source": 6,
        "switch": 7,
        "relay": 8,
        "lamp": 9,
        "flag": 10,
        "=": 11,
        ";": 12,
        ".": 13,
        "AND": 14,
        "What": 15,
        "is": 16,
        "?": 17,
        "Circuit": 18,
        "Rules": 19,
        "The": 20,
        "independent": 21,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=0,
    )
    return LocalLanguageModel(LlamaForCausalLM(config).eval(), tokenizer, max_length=512)


def test_candidate_likelihood_matches_manual_forward_and_has_gradients():
    lm = tiny_model()
    scores = lm.scores(["source = 1. Answer:", "Answer:"])
    ids = lm.tokenizer.encode("source = 1. Answer:", add_special_tokens=True)
    for value in (0, 1):
        answer = lm.tokenizer.encode(f" {value}", add_special_tokens=False)
        sequence = torch.tensor([ids + answer])
        logits = lm.model(sequence, use_cache=False).logits[0]
        manual = sum(
            logits[len(ids) - 1 + i].log_softmax(-1)[token] for i, token in enumerate(answer)
        )
        torch.testing.assert_close(scores[0, value], manual)
    (-scores[:, 1].sum()).backward()
    assert lm.model.model.layers[0].mlp.down_proj.weight.grad is not None


def test_no_silent_truncation_and_local_roundtrip(tmp_path: Path):
    lm = tiny_model()
    lm.max_length = 2
    with pytest.raises(ValueError, match="max_length"):
        lm.scores(["source = 1. Answer:"])
    lm.model.save_pretrained(tmp_path)
    lm.tokenizer.save_pretrained(tmp_path)
    from semantics_operator.config import Settings

    loaded = LocalLanguageModel.load(Settings("test", tmp_path, device="cpu"))
    assert loaded.scores(["Answer:"]).shape == (1, 2)
    assert "model.layers.0.mlp.down_proj" in loaded.linear_modules()


def test_representation_capture_removes_hooks():
    lm = tiny_model()
    module = lm.model.model.layers[0].mlp.down_proj
    before = len(module._forward_hooks)
    vectors = lm.representations(["Answer:", "source = 1. Answer:"], "model.layers.0.mlp.down_proj")
    assert vectors.shape == (2, 16)
    assert len(module._forward_hooks) == before


def test_multitoken_continuations_include_each_token_and_ignore_padding():
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    lm = tiny_model()
    vocab = {"[UNK]": 0, "[PAD]": 1, "Ġ": 2, "0": 3, "1": 4, "A": 5, ":": 6}
    backend = Tokenizer(BPE(vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    lm.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    suffix = lm.tokenizer.encode(" 1", add_special_tokens=False)
    assert len(suffix) == 2
    prompts = ["A:", "A A A:"]
    batched = lm.scores(prompts)
    for row, prompt in enumerate(prompts):
        prefix = lm.tokenizer.encode(prompt, add_special_tokens=True)
        for value in (0, 1):
            answer = lm.tokenizer.encode(f" {value}", add_special_tokens=False)
            sequence = torch.tensor([prefix + answer])
            logits = lm.model(sequence, use_cache=False).logits[0]
            expected = sum(
                logits[len(prefix) - 1 + i].log_softmax(-1)[token] for i, token in enumerate(answer)
            )
            torch.testing.assert_close(batched[row, value], expected)


def test_bare_candidate_scoring_and_unforced_greedy_generation(monkeypatch):
    from types import SimpleNamespace

    lm = tiny_model()
    prompt = "source = 0. Answer:"
    ids = lm._prompt_ids(prompt)
    bare = lm.scores([prompt], candidates=("0", "1"))
    torch.testing.assert_close(bare, lm.scores([prompt]))  # WordLevel ignores spaces.
    seen = []
    generated = [3, 0]  # 1 then EOS; no common answer prefix is supplied.

    def forward(input_ids, attention_mask, use_cache):
        seen.append(input_ids[0].tolist())
        logits = torch.zeros(1, input_ids.shape[1], 22)
        logits[0, -1, generated[len(seen) - 1]] = 10
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(lm.model, "forward", forward)
    result = lm.generate_greedy([prompt], max_new_tokens=4)[0]
    assert seen == [ids, ids + [3]]
    assert result == {"text": "1", "token_ids": [3, 0], "stop_reason": "eos"}
    lm.max_length = len(ids) + 1
    with pytest.raises(ValueError, match="max_length"):
        lm.generate_greedy([prompt], max_new_tokens=4)
