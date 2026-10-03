import pytest
import torch
from torch import nn

from semantics_operator.weights import WeightSession


def test_edits_are_real_weight_updates_and_exception_restores_everything():
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(3, 2))
    x = torch.randn(4, 3)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    baseline = model(x).detach()
    with (
        pytest.raises(RuntimeError, match="test failure"),
        WeightSession(model, "0", rank=1) as session,
    ):
        with torch.no_grad():
            session.layer.a.fill_(0.4)
            session.layer.b.fill_(0.7)
        expected = nn.functional.linear(
            x, original["0.weight"] + session.layer.delta(), original["0.bias"]
        )
        torch.testing.assert_close(model(x), expected)
        assert not torch.equal(model(x), baseline)
        raise RuntimeError("test failure")
    assert isinstance(model[0], nn.Linear)
    assert model[0].weight.requires_grad
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_snapshots_are_independent_and_composition_uses_no_cross_terms():
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    x = torch.randn(4, 3)
    baseline = model(x).detach()
    with WeightSession(model, "0", rank=2) as session:
        with torch.no_grad():
            session.layer.b.fill_(0.1)
        first = session.snapshot()
        d1 = session.layer.delta().detach().clone()
        with torch.no_grad():
            session.layer.a.add_(0.2)
            session.layer.b.add_(0.2)
        second = session.snapshot()
        d2 = session.layer.delta().detach().clone()
        with session.branch((first, second)):
            torch.testing.assert_close(model(x), baseline + nn.functional.linear(x, d1 + d2))
            with session.branch(()):
                torch.testing.assert_close(model(x), baseline)
            torch.testing.assert_close(model(x), baseline + nn.functional.linear(x, d1 + d2))
        torch.testing.assert_close(session.layer.delta(), d2)
        assert not torch.equal(first.a, second.a)


def test_gradient_changes_only_adapter_and_reset_starts_at_baseline():
    model = nn.Sequential(nn.Linear(3, 2))
    base = model[0].weight.detach().clone()
    with WeightSession(model, "0", rank=2) as session:
        optimizer = torch.optim.Adam(session.parameters(), lr=0.1)
        loss = model(torch.ones(2, 3)).square().sum()
        loss.backward()
        optimizer.step()
        assert session.layer.b.abs().sum() > 0
        assert session.layer.base.weight.grad is None
        assert torch.equal(session.layer.base.weight, base)
        session.reset(seed=8)
        assert session.layer.delta().count_nonzero() == 0


def test_norm_matched_random_control_and_invalid_target():
    from semantics_operator.weights import randomized

    model = nn.Sequential(nn.Linear(3, 2))
    with WeightSession(model, "0", rank=2) as session:
        with torch.no_grad():
            session.layer.b.fill_(0.2)
        patch = session.snapshot()
        control = randomized(patch, seed=3)
        torch.testing.assert_close((control.b @ control.a).norm(), (patch.b @ patch.a).norm())
    with (
        pytest.raises(ValueError, match="Linear"),
        WeightSession(nn.Sequential(nn.ReLU()), "0", rank=2),
    ):
        pass
