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


def test_full_state_batches_cover_roles_and_train_styles_without_changing_targets():
    from semantics_operator.causal_tasks import PromptStyle, circuit_questions, metrics
    from semantics_operator.reft_experiment import train_batches
    from semantics_operator.world import OPERATORS

    samples = circuit_questions("train", styles=tuple(PromptStyle))
    assert len(samples) == len({q.key for q in samples}) == 120
    assert len({q.prompt() for q in samples}) == 120
    for op in OPERATORS:
        batches = train_batches(samples, op, 80, 42)
        assert batches == train_batches(samples, op, 80, 42)
        assert {i for batch in batches for i in batch} == set(range(120))
        for step, indices in enumerate(batches):
            batch = [samples[i] for i in indices]
            assert len(batch) == 5
            assert len({(q.world.name, q.style) for q in batch}) == 1
            assert {q.node.value for q in batch} == {"source", "switch", "relay", "lamp", "flag"}
            assert any(q.answer((op,)) != q.answer() for q in batch) == (step % 2 == 0)
    scores = torch.tensor([[1 - q.answer(), q.answer()] for q in samples]).float()
    report = metrics(scores, samples, (), scores)
    assert report["all_nodes_correct"] == 1
    assert set(report["by_style"]) == {s.value for s in PromptStyle}
    assert report["by_label"]["0"]["accuracy"] == 1
    # One failed style/world must not mark all three descriptions as failed.
    scores[0] = scores[0].flip(0)
    assert metrics(scores, samples, (), scores)["all_nodes_correct"] == 23 / 24


def test_microbatch_locality_gradient_matches_full_state_objective():
    from semantics_operator.reft import task_locality_loss

    baseline = torch.randn(5, 2)
    labels = torch.tensor([0, 1, 0, 0, 1])
    affected = torch.tensor([False, False, True, True, False])
    full = torch.randn(5, 2, requires_grad=True)
    micro = full.detach().clone().requires_grad_()
    expected, _ = task_locality_loss(full, baseline, labels, affected, 1.7)
    expected.backward()
    total = 0
    for start in range(0, 5, 2):
        sl = slice(start, start + 2)
        loss, _ = task_locality_loss(
            micro[sl],
            baseline[sl],
            labels[sl],
            affected[sl],
            1.7,
            normalization_counts=(2, 3),
        )
        total += float(loss.detach())
        loss.backward()
    assert total == pytest.approx(float(expected.detach()), abs=1e-6)
    torch.testing.assert_close(micro.grad, full.grad)


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
