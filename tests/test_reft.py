import pytest
import torch
from test_model import tiny_model


def test_graph_shifts_have_different_counterfactuals_and_complete_questions():
    from semantics_operator.causal_tasks import Scheme, circuit_questions
    from semantics_operator.world import OPERATORS

    sets = {s: circuit_questions("test", s) for s in Scheme}
    assert len(sets[Scheme.AND_COPY]) == 40
    assert len(sets[Scheme.AND_CHAIN]) == 48
    gate = next(
        q
        for q in sets[Scheme.AND_GATED]
        if q.world.source == 1
        and q.world.switch == 1
        and q.world.flag == 0
        and q.node.value == "lamp"
    )
    assert gate.answer((OPERATORS[1],)) == 0
    assert gate.answer((OPERATORS[2],)) == 1
    chain = next(q for q in sets[Scheme.AND_CHAIN] if q.node.value == "bridge")
    assert chain.answer((OPERATORS[0],)) == 0
    assert chain.answer((OPERATORS[1],)) == 1
    assert chain.answer((OPERATORS[0], OPERATORS[2])) == 0
    assert sets[Scheme.AND_COPY][0].world.values() != sets[Scheme.AND_CHAIN][0].world.values()
    assert all("Override" not in q.prompt() for qs in sets.values() for q in qs)
    for scheme, qs in sets.items():
        for q in qs:
            if q.node.value == "relay":
                expected = (
                    q.world.source | q.world.switch
                    if scheme == Scheme.OR_COPY
                    else q.world.source & q.world.switch
                )
                assert q.answer() == expected
    with pytest.raises(ValueError, match="test-only"):
        circuit_questions("train", Scheme.OR_COPY)
    from semantics_operator.causal_tasks import metrics

    samples = sets[Scheme.AND_CHAIN]
    scores = torch.tensor([[1 - q.answer(), q.answer()] for q in samples]).float()
    assert metrics(scores, samples, (), scores)["by_node"]["bridge"] == {
        "count": 8,
        "accuracy": 1.0,
    }


def test_loreft_and_das_match_formulas_and_preserve_complement():
    from semantics_operator.reft import DAS, LoReFT

    model = LoReFT(6, 2, seed=7)
    h, donor = torch.randn(3, 6), torch.randn(3, 6)
    assert torch.allclose(model(h), h, atol=1e-6)
    with torch.no_grad():
        model.bias.add_(1)
    q = model.basis()
    assert torch.allclose(q.T @ q, torch.eye(2), atol=1e-6)
    expected = h + (h @ model.weight.T + model.bias - h @ q) @ q.T
    assert torch.allclose(model(h), expected, atol=1e-6)
    das = DAS(6, 2, seed=8)
    q = das.basis()
    changed = das(h, donor)
    assert torch.allclose(changed @ q, donor @ q, atol=1e-6)
    assert torch.allclose((changed - h) @ (torch.eye(6) - q @ q.T), torch.zeros_like(h), atol=1e-6)
    with pytest.raises(ValueError):
        LoReFT(6, 7)


def test_locality_loss_backpropagates_through_frozen_block_and_removes_hook():
    from semantics_operator.reft import (
        LoReFT,
        frozen_model,
        intervention_scores,
        task_locality_loss,
    )

    lm = tiny_model()
    site = "model.layers.0"
    prompts = ["source = 1. Answer:", "source = 0. Answer:"]
    flags = [p.requires_grad for p in lm.model.parameters()]
    before = {k: v.clone() for k, v in lm.model.state_dict().items()}
    operator = LoReFT(16, 2)
    with frozen_model(lm):
        with torch.no_grad():
            baseline = lm.scores(prompts)
        edited = intervention_scores(lm, prompts, site, [], operator)
        loss, parts = task_locality_loss(
            edited, baseline, torch.tensor([0, 1]), torch.tensor([True, False]), 1.0
        )
        loss.backward()
        assert operator.bias.grad.abs().sum() > 0
        assert all(p.grad is None for p in lm.model.parameters())
        assert abs(parts["locality_kl"]) < 1e-6
        shifted = baseline.clone()
        shifted[1, 0] += 2
        _, parts = task_locality_loss(
            shifted, baseline, torch.tensor([0, 1]), torch.tensor([True, False]), 1.0
        )
        assert parts["locality_kl"] > 0
    assert flags == [p.requires_grad for p in lm.model.parameters()]
    assert all(torch.equal(v, lm.model.state_dict()[k]) for k, v in before.items())
    assert not lm.model.get_submodule(site)._forward_hooks
    with pytest.raises(RuntimeError, match="probe"):
        intervention_scores(
            lm, prompts, site, [], lambda h: (_ for _ in ()).throw(RuntimeError("probe"))
        )
    assert not lm.model.get_submodule(site)._forward_hooks
