import pytest
import torch
from test_model import tiny_model


def test_position_cache_does_not_keep_an_unloaded_model_alive():
    import gc
    import weakref

    from semantics_operator.positions import intervention_positions

    lm = tiny_model()
    intervention_positions(lm, ["Answer:"])
    reference = weakref.ref(lm)
    del lm
    gc.collect()
    assert reference() is None


def test_state_position_excludes_bpe_token_crossing_the_description_boundary():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    from semantics_operator.positions import ReftPosition, intervention_positions

    lm = tiny_model()
    backend = Tokenizer(
        BPE(
            {"[UNK]": 0, "[PAD]": 1, ".": 2, "Ċ": 3, ".Ċ": 4},
            merges=[(".", "Ċ")],
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    lm.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    prompt = "source=0.\nState recorded.\nWhat is lamp?"
    position = intervention_positions(lm, [prompt], ReftPosition.STATE)[0]
    ids = lm._prompt_ids(prompt)
    # The period/newline token crosses the chosen text boundary; use its predecessor.
    assert ids[position + 1] == 4
    assert ids[position] != 4


@pytest.mark.parametrize("chat", [False, True])
def test_state_tokens_and_hidden_vectors_are_independent_of_the_question(chat):
    from semantics_operator.causal_tasks import PromptLayout, PromptStyle, circuit_questions
    from semantics_operator.localization import capture_tail
    from semantics_operator.positions import (
        ReftPosition,
        intervention_positions,
        validate_state_prefixes,
    )

    lm = tiny_model()
    if chat:
        lm.tokenizer.chat_template = "é user: {{ messages[0]['content'] }}\nassistant:"
    samples = circuit_questions("train", styles=tuple(PromptStyle), layout=PromptLayout.STATE_FIRST)
    check = validate_state_prefixes(lm, samples)
    assert check["states"] == 24
    assert check["questions"] == 120
    prompts = [q.prompt() for q in samples[:5]]
    positions = intervention_positions(lm, prompts, ReftPosition.STATE)
    prefixes = [lm._prompt_ids(p)[: pos + 1] for p, pos in zip(prompts, positions)]
    assert all(p == prefixes[0] for p in prefixes)
    h = capture_tail(lm, prompts, ["model.layers.0"], 1, [], positions=positions)["model.layers.0"]
    torch.testing.assert_close(h, h[:1].expand_as(h))
    assert len(set(prompts)) == 5
    assert all(pos < len(lm._prompt_ids(p)) - 5 for p, pos in zip(prompts, positions))


def test_state_edit_reaches_answers_only_through_later_blocks_and_all_paths_use_it():
    from semantics_operator.answer_protocol import BARE_CANDIDATES
    from semantics_operator.causal_tasks import PromptLayout, circuit_questions
    from semantics_operator.full_vocab import answer_sequences, teacher_forced_logits
    from semantics_operator.positions import ReftPosition
    from semantics_operator.reft import intervention_generate, intervention_scores

    lm = tiny_model()
    prompts = [q.prompt() for q in circuit_questions("train", layout=PromptLayout.STATE_FIRST)[:2]]
    base = lm.scores(prompts, candidates=BARE_CANDIDATES)
    seen = []

    def transform(h):
        seen.append(h.detach().clone())
        return h + torch.linspace(-2, 2, h.shape[-1])

    early = intervention_scores(
        lm, prompts, "model.layers.0", [], transform, position=ReftPosition.STATE
    )
    scoring_state = seen[0][::2]
    assert (early - base).abs().max() > 1e-5
    teacher_forced_logits(
        lm,
        prompts,
        answer_sequences(lm, [0, 1]),
        site="model.layers.0",
        transform=transform,
        position=ReftPosition.STATE,
    )
    torch.testing.assert_close(seen[-1], scoring_state)
    intervention_generate(
        lm, prompts[:1], "model.layers.0", transform, max_new_tokens=1, position=ReftPosition.STATE
    )
    torch.testing.assert_close(seen[-1], scoring_state[:1])
    final = intervention_scores(
        lm, prompts, "model.layers.1", [], transform, position=ReftPosition.STATE
    )
    torch.testing.assert_close(final, base, atol=0, rtol=0)
    assert not lm.model.model.layers[0]._forward_hooks
    assert not lm.model.model.layers[1]._forward_hooks


def test_state_localization_keeps_final_block_as_negative_control():
    from semantics_operator.causal_tasks import PromptLayout, circuit_questions
    from semantics_operator.positions import ReftPosition
    from semantics_operator.reft_localization import locate_block

    lm = tiny_model()
    samples = circuit_questions("validation", layout=PromptLayout.STATE_FIRST)
    site, report = locate_block(lm, [0, 1], 1, samples=samples, position=ReftPosition.STATE)
    assert site == "model.layers.0"
    control = next(t for t in report["trials"] if t["layer"] == 1)
    assert control["negative_control"]
    assert control["donor_max_score_difference"] == 0
    assert all(t["self_patch_max_score_difference"] < 1e-5 for t in report["trials"])
    with pytest.raises(ValueError, match="earlier"):
        locate_block(lm, [1], 1, samples=samples, position=ReftPosition.STATE)
