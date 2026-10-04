import torch

from semantics_operator.causal_tasks import Scheme, circuit_questions, metrics
from semantics_operator.generation_evaluation import generation_metrics
from semantics_operator.world import OPERATORS


def test_lamp_edit_counts_damage_to_relay_and_bridge_in_both_evaluators():
    samples = circuit_questions("test", Scheme.AND_CHAIN)[:6]
    # Inputs 000: natural relay/bridge/lamp are 0. Only lamp should become 1.
    baseline_values = [0, 0, 0, 0, 0, 0]
    edited_values = [0, 0, 1, 1, 0, 1]
    scores = lambda values: torch.tensor([[1 - v, v] for v in values]).float()
    ranked = metrics(scores(edited_values), samples, (OPERATORS[2],), scores(baseline_values))
    generated = generation_metrics(
        [{"prediction": v} for v in edited_values],
        samples,
        (OPERATORS[2],),
        [{"prediction": v} for v in baseline_values],
    )
    for result in (ranked, generated):
        damage = result["protected_damage"]
        assert damage["eligible"] == 5
        assert damage["damaged"] == 2
        assert damage["by_node"]["relay"] == {"eligible": 1, "damaged": 1}
        assert damage["by_node"]["bridge"] == {"eligible": 1, "damaged": 1}
        assert damage["by_node"]["lamp"]["eligible"] == 0


def test_equations_check_predictions_not_only_correctness_and_apply_do_replacement():
    samples = circuit_questions("test", Scheme.AND_INVERTED_CHAIN)[:6]

    def evaluate(values, sequence):
        return generation_metrics(
            [{"prediction": v} for v in values],
            samples,
            sequence,
            [{"prediction": v} for v in [0, 0, 0, 1, 0, 1]],
        )

    # The observed failure: relay=0, bridge=0, lamp=1.
    result = evaluate([0, 0, 0, 1, 0, 0], (OPERATORS[0],))
    structure = result["equation_consistency"]
    assert structure["all_satisfied"] == 0
    assert structure["by_equation"]["bridge"]["violated"] == 1
    assert structure["by_equation"]["lamp"]["violated"] == 1
    # lamp=0 overrides lamp=bridge, so this is a valid do-state despite disagreement.
    from semantics_operator.world import Intervention, Node

    result = evaluate([0, 0, 0, 0, 0, 1], (Intervention(Node.LAMP, 0),))
    assert result["equation_consistency"]["all_satisfied"] == 1
    # Consistent equations can still describe the wrong input state.
    result = evaluate([1, 1, 1, 0, 0, 0], ())
    assert result["equation_consistency"]["all_satisfied"] == 1
    assert result["all_nodes_correct"] == 0
    # An invalid parent's answer cannot make a dependent equation pass.
    result = evaluate([0, 0, 0, 1, 0, None], (OPERATORS[0],))
    assert result["equation_consistency"]["by_equation"]["lamp"]["invalid"] == 1


def test_structurally_affected_but_unchanged_answers_are_not_protected():
    samples = circuit_questions("test")[:5]
    base = torch.tensor([[1.0, 0.0]] * 5)
    result = metrics(base, samples, (OPERATORS[0],), base)
    assert result["protected_damage"]["eligible"] == 3
    assert result["protected_damage"]["by_node"]["relay"]["eligible"] == 0
    assert result["protected_damage"]["by_node"]["lamp"]["eligible"] == 0
