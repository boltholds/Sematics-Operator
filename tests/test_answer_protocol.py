import pytest
import torch
from test_model import tiny_model


def test_inversion_xor_counterfactuals_and_uniform_answer_instruction():
    from semantics_operator.answer_protocol import ANSWER_INSTRUCTION
    from semantics_operator.causal_tasks import PromptStyle, Scheme, circuit_questions
    from semantics_operator.world import OPERATORS

    for scheme in (Scheme.AND_INVERTED, Scheme.AND_XOR):
        samples = circuit_questions("test", scheme, styles=tuple(PromptStyle))
        for q in samples:
            assert q.prompt().endswith(ANSWER_INSTRUCTION + "\nAnswer:")
            assert q.prompt((OPERATORS[0],)).endswith(ANSWER_INSTRUCTION + "\nAnswer:")
            if q.node.value == "lamp":
                assert q.answer((OPERATORS[0],)) == (
                    1 if scheme == Scheme.AND_INVERTED else q.world.flag
                )
                assert q.answer((OPERATORS[1],)) == (
                    0 if scheme == Scheme.AND_INVERTED else 1 - q.world.flag
                )
                assert q.answer((OPERATORS[0], OPERATORS[2])) == 1
            elif q.node.value in ("source", "switch", "flag"):
                assert all(q.answer((op,)) == q.answer() for op in OPERATORS)
        for split in ("train", "validation"):
            with pytest.raises(ValueError, match="test-only"):
                circuit_questions(split, scheme)
    for split in ("train", "validation", "test"):
        for q in circuit_questions(split, styles=tuple(PromptStyle)):
            assert q.prompt().endswith(ANSWER_INSTRUCTION + "\nAnswer:")


def test_incomplete_digit_is_not_counted_as_a_completed_answer():
    from semantics_operator.answer_protocol import parse_generation

    assert (
        parse_generation({"text": " 1\n", "token_ids": [1, 7], "stop_reason": "eos"})["prediction"]
        == 1
    )
    incomplete = parse_generation({"text": "1", "token_ids": [1], "stop_reason": "max_new_tokens"})
    assert incomplete["prediction"] is None
    assert incomplete["generation_status"] == "incomplete"
    wrong_format = parse_generation({"text": "Answer: 0", "token_ids": [], "stop_reason": "eos"})
    assert wrong_format["prediction"] is None
    assert wrong_format["generation_status"] == "format_error"


def test_scoring_and_generation_intervene_at_the_same_original_prompt_position():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    from semantics_operator.reft import intervention_generate, intervention_scores

    lm = tiny_model()
    backend = Tokenizer(
        BPE(
            {"[UNK]": 0, "[PAD]": 1, "Ġ": 2, "0": 3, "1": 4, "A": 5, ":": 6},
            merges=[],
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    lm.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    lm.model.generation_config.eos_token_id = []  # Force two steps to check the position stays fixed.
    prompt, site = "A:", "model.layers.0"
    seen = []

    def capture(h):
        seen.append(h.detach().clone())
        return h + 0.125

    scored = intervention_scores(lm, [prompt], site, [], capture)
    first = seen.pop()
    torch.testing.assert_close(first[0], first[1])  # Candidate digit cannot reach the hook input.
    outputs = intervention_generate(lm, [prompt], site, capture, max_new_tokens=2)
    assert len(outputs[0]["token_ids"]) == 2
    assert len(seen) == 2
    for h in seen:
        torch.testing.assert_close(h[0], first[0])
    identity = intervention_scores(lm, [prompt], site, [], lambda h: h)
    torch.testing.assert_close(identity, lm.scores([prompt], candidates=("0", "1")))
    assert not torch.allclose(identity, lm.scores([prompt]), atol=1e-5)
    assert scored.shape == (1, 2)
    assert not lm.model.get_submodule(site)._forward_hooks
    with pytest.raises(ValueError, match="prefix"):
        intervention_scores(lm, [prompt], site, [2], capture)
    with pytest.raises(RuntimeError, match="probe"):
        intervention_generate(
            lm, [prompt], site, lambda h: (_ for _ in ()).throw(RuntimeError("probe"))
        )
    assert not lm.model.get_submodule(site)._forward_hooks


def test_generated_metrics_measure_source_damage_and_incomplete_answers():
    from semantics_operator.causal_tasks import circuit_questions
    from semantics_operator.generation_evaluation import generation_metrics
    from semantics_operator.world import OPERATORS

    samples = circuit_questions("test")
    base = [
        {"prediction": q.answer(), "generation_status": "binary", "text": str(q.answer())}
        for q in samples
    ]
    edited = [
        {
            "prediction": 0 if q.node.value == "source" else q.answer((OPERATORS[0],)),
            "generation_status": "binary",
        }
        for q in samples
    ]
    edited[1] = {"prediction": None, "generation_status": "incomplete"}
    result = generation_metrics(edited, samples, (OPERATORS[0],), base)
    assert result["protected_damage"]["by_node"]["source"] == {"eligible": 8, "damaged": 4}
    assert result["protected_damage"]["by_node"]["switch"]["damaged"] == 1
    assert result["overall"]["incomplete"] == 1
    assert result["all_nodes_correct"] == 3 / 8
