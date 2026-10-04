import pytest
import torch
from test_model import tiny_model


def test_full_vocabulary_loss_penalizes_other_words_and_includes_eos():
    from semantics_operator.full_vocab import full_vocab_loss

    # Binary 0/1 preference is unchanged, but another word steals all probability.
    good = torch.tensor([[5.0, 0.0, -5.0, -5.0], [-5.0, -5.0, 5.0, -5.0]])
    bad = good.clone()
    bad[:, 3] = 10
    targets = [[0, 2]]
    a, _, _ = full_vocab_loss([good], [], targets, [True], 0)
    b, _, _ = full_vocab_loss([bad], [], targets, [True], 0)
    assert b > a + 4
    wrong_eos = good.clone()
    wrong_eos[1] = torch.tensor([5.0, -5.0, -5.0, -5.0])
    c, _, _ = full_vocab_loss([wrong_eos], [], targets, [True], 0)
    assert c > a + 4
    _, parts, _ = full_vocab_loss([bad], [good.log_softmax(-1)], targets, [False], 1)
    assert parts["locality_kl"] > 4


def test_full_vocab_microbatch_gradients_equal_full_state_mean():
    from semantics_operator.full_vocab import full_vocab_loss

    torch.manual_seed(7)
    full = [torch.randn(2, 7, requires_grad=True) for _ in range(5)]
    micro = [x.detach().clone().requires_grad_() for x in full]
    reference = [torch.randn(2, 7).log_softmax(-1) for _ in full]
    targets, affected = [[2, 0]] * 5, [False, False, True, True, False]
    expected, _, _ = full_vocab_loss(full, reference, targets, affected, 1.3)
    expected.backward()
    total = 0
    for start in range(0, 5, 2):
        sl = slice(start, start + 2)
        loss, _, _ = full_vocab_loss(
            micro[sl], reference[sl], targets[sl], affected[sl], 1.3, normalization_counts=(2, 3)
        )
        total += float(loss.detach())
        loss.backward()
    assert total == pytest.approx(float(expected.detach()), abs=1e-6)
    for a, b in zip(full, micro, strict=True):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("multitoken", [False, True])
def test_teacher_forcing_matches_manual_logits_and_keeps_original_intervention_position(multitoken):
    from semantics_operator.full_vocab import answer_sequences, teacher_forced_logits
    from semantics_operator.localization import capture_tail
    from semantics_operator.reft import LoReFT, frozen_model

    lm = tiny_model()
    if multitoken:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(
            BPE(
                {
                    "[UNK]": 0,
                    "[PAD]": 1,
                    "Ġ": 2,
                    "0": 3,
                    "1": 4,
                    "A": 5,
                    ":": 6,
                    "[EOS]": 7,
                    "[EOS2]": 8,
                    "Ġ1": 9,
                },
                merges=[("Ġ", "1")],
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = ByteLevel(add_prefix_space=True)
        lm.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
            add_prefix_space=True,
        )
        lm.model.generation_config.eos_token_id = [7, 8]
    prompts = ["source = 0. Answer:", "Answer:"]
    targets = answer_sequences(lm, [0, 1])
    assert targets == ([[2, 3, 7], [9, 7]] if multitoken else [[2, 0], [3, 0]])
    seen = []
    operator = LoReFT(16, 2)

    def edit(h):
        seen.append(h.detach().clone())
        return operator(h)

    with frozen_model(lm):
        actual = teacher_forced_logits(lm, prompts, targets, site="model.layers.0", transform=edit)
        assert seen[0].shape == (2, 16)
        original = capture_tail(lm, prompts, ["model.layers.0"], 1, [])
        torch.testing.assert_close(seen[0], original["model.layers.0"][:, 0])
        for i, prompt in enumerate(prompts):
            prefix = lm._prompt_ids(prompt)
            ids, mask = lm._batch([prefix + targets[i][:-1]])
            with torch.no_grad():
                expected = lm.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0]
            torch.testing.assert_close(actual[i], expected[len(prefix) - 1 :])
        sum(
            torch.nn.functional.cross_entropy(logits, torch.tensor(t))
            for logits, t in zip(actual, targets, strict=True)
        ).backward()
        assert operator.bias.grad.abs().sum() > 0
        assert all(p.grad is None for p in lm.model.parameters())
    assert not lm.model.get_submodule("model.layers.0")._forward_hooks
    lm.model.generation_config.eos_token_id = []
    with pytest.raises(ValueError, match="EOS"):
        answer_sequences(lm, [0])


@pytest.mark.parametrize("mode", ["full_vocab", "binary"])
def test_training_loss_modes_update_only_the_operator(tmp_path, mode):
    from semantics_operator.causal_tasks import training_questions
    from semantics_operator.config import Settings
    from semantics_operator.full_vocab import LossMode, reference_distributions
    from semantics_operator.reft import frozen_model
    from semantics_operator.reft_experiment import evaluate, train_loreft
    from semantics_operator.world import OPERATORS

    lm = tiny_model()
    samples = training_questions("train")
    before = {key: value.clone() for key, value in lm.model.state_dict().items()}
    with frozen_model(lm):
        base = evaluate(lm, samples, "model.layers.0", [])
        reference = reference_distributions(lm, samples) if mode == "full_vocab" else None
        operator, log = train_loreft(
            lm,
            Settings("tiny", tmp_path, steps=2, rank=2),
            samples,
            base,
            "model.layers.0",
            [],
            OPERATORS[1],
            16,
            1,
            lambda _: None,
            loss_mode=LossMode(mode),
            full_reference=reference,
        )
        assert log["loss_mode"] == mode
        assert log["schemes"] == ["and_copy", "and_inverted"]
        assert operator.bias.detach().abs().sum() > 0
        assert all(p.grad is None for p in lm.model.parameters())
    assert all(torch.equal(before[k], v) for k, v in lm.model.state_dict().items())
    assert not lm.model.get_submodule("model.layers.0")._forward_hooks
