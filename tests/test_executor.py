import json
from dataclasses import replace

import pytest
import torch


def chain():
    from semantics_operator.executor_tasks import Circuit, Rule

    return Circuit(
        (0, 0, 1, 2, 4), (Rule.ROOT, Rule.COPY, Rule.NOT, Rule.COPY, Rule.ROOT), (0, 0, 0, 0, 1)
    )


def test_oracle_propagates_one_edge_and_preserves_independent_root():
    from semantics_operator.executor_tasks import Episode, oracle_trace

    episode = Episode("chain", chain(), ((1, 1),))
    assert oracle_trace(episode, 3) == [
        (0, 1, 1, 1, 1),
        (0, 1, 0, 1, 1),
        (0, 1, 0, 0, 1),
        (0, 1, 0, 0, 1),
    ]
    double = replace(episode, interventions=((1, 1), (2, 1)))
    assert oracle_trace(double, 3)[-1] == (0, 1, 1, 1, 1)
    order = (3, 1, 4, 0, 2)
    permuted = episode.permute(order)
    assert oracle_trace(permuted, 3) == [
        tuple(s[i] for i in order) for s in oracle_trace(episode, 3)
    ]
    assert episode.graph.topology_key() == permuted.graph.topology_key()


def test_cycles_and_invalid_interventions_are_rejected():
    from semantics_operator.executor_tasks import Circuit, Episode, Rule

    with pytest.raises(ValueError, match="cycle"):
        Circuit((1, 0), (Rule.COPY, Rule.NOT), (0, 0))
    with pytest.raises(ValueError):
        Episode("bad", chain(), ((1, 0), (1, 1)))
    with pytest.raises(ValueError):
        Episode("bad", chain(), ((6, 0),))


def test_dataset_topologies_are_disjoint_and_seeds_reproduce():
    from semantics_operator.executor_tasks import make_dataset

    data = make_dataset(seed=7, train_graphs=8, eval_graphs=4)
    again = make_dataset(seed=7, train_graphs=8, eval_graphs=4)
    assert data == again
    keys = {s: {g.topology_key() for g in graphs} for s, graphs in data.items()}
    assert len(set.union(*keys.values())) == sum(map(len, keys.values()))
    assert all(6 <= len(g.parents) <= 9 and 2 <= g.depth() <= 3 for g in data["train"])
    assert all(11 <= len(g.parents) <= 14 and 6 <= g.depth() <= 8 for g in data["long"])


def test_executor_has_one_hop_state_dependence_and_gradients():
    from semantics_operator.executor_model import LocalExecutor, pack, rollout
    from semantics_operator.executor_tasks import Episode

    torch.manual_seed(12)
    model = LocalExecutor("transformer", width=16, blocks=2, heads=2)
    inputs, gold, _protected = pack([Episode("a", chain(), ((1, 1),))], steps=3)
    p = inputs.initial.clone().requires_grad_()
    output = model(p, inputs.parents, inputs.rules)
    gradient = torch.autograd.grad(output[0, 3], p)[0]
    assert gradient[0, 0] == 0 and gradient[0, 1] == 0 and gradient[0, 4] == 0
    assert gradient[0, 2].abs() > 0
    states = rollout(model, inputs, 3)
    assert states.shape == gold.shape == (1, 4, 5)
    assert torch.all(states[:, :, 1] == 1)
    assert torch.all(states[:, :, 4] == 1)
    assert torch.all(states[:, :, 0] == 0)
    assert not inputs.free[0, 0] and not inputs.free[0, 1] and not inputs.free[0, 4]
    states[:, -1, 3].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_pulse_is_transient_and_metrics_expose_errors():
    from semantics_operator.executor_experiment import trajectory_metrics
    from semantics_operator.executor_tasks import Episode, oracle_step, oracle_trace

    e = Episode("chain", chain(), ((1, 1),))
    gold = oracle_trace(e, 3)
    pulse = list(gold[-1])
    pulse[2] = 1
    assert oracle_step(e, tuple(pulse)) == (0, 1, 0, 1, 1)
    assert oracle_step(e, oracle_step(e, tuple(pulse))) == (0, 1, 0, 0, 1)
    wrong = torch.tensor([[gold[0]] * 4], dtype=torch.float32)
    m = trajectory_metrics([e], wrong, torch.tensor([gold], dtype=torch.float32))
    assert m["all_nodes_final"] == {"correct": 0, "count": 1}
    assert m["changed_free_nodes"] == {"correct": 0, "count": 2}
    assert m["preservation_damage"]["damaged"] == 0
    noop = replace(e, interventions=((1, 0),))
    no_trace = torch.tensor([oracle_trace(noop, 3)], dtype=torch.float32)
    m = trajectory_metrics([noop], no_trace, no_trace)
    assert m["changed_free_nodes"] == {"correct": 0, "count": 0}


def test_executor_cli_runs_without_llm_and_saves_reloadable_weights(tmp_path):
    from safetensors.torch import load_file

    from semantics_operator.cli import main
    from semantics_operator.executor_model import LocalExecutor

    config = tmp_path / "executor.toml"
    config.write_text(
        '[executor]\nsteps=2\nwidth=16\nheads=2\nblocks=2\nbatch_size=4\ntrain_graphs=8\neval_graphs=2\nseeds=[9]\noutput_dir="runs"\n'
    )
    assert (
        main(
            [
                "executor",
                "--executor-config",
                str(config),
                "--env-file",
                str(tmp_path / "absent.env"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report_file = next((tmp_path / "runs").glob("*-executor-*/report.json"))
    report = json.loads(report_file.read_text())
    assert report["experiment"] == "inspectable_causal_executor_v1"
    assert set(report["runs"]) == {"transformer/9", "mlp/9"}
    assert (report_file.parent / "index.html").is_file()
    model = LocalExecutor("transformer", width=16, blocks=2, heads=2)
    model.load_state_dict(load_file(report_file.parent / "transformer-9.safetensors"), strict=True)
    assert report["runs"]["transformer/9"]["selected_step"] in (1, 2)
    assert "pulse_audit" in report["runs"]["transformer/9"]


def test_pulse_flips_actual_memory_even_when_pre_state_is_wrong():
    from semantics_operator.executor_experiment import pulse_audit
    from semantics_operator.executor_tasks import Episode

    class AlwaysOne(torch.nn.Module):
        def forward(self, state, parents, rules):
            return torch.ones_like(state)

    # At step 2, gold n2=0 but this model stores n2=1. Pulse must flip actual 1 to 0.
    result = pulse_audit(AlwaysOne(), [Episode("wrong/0/1", chain(), ((1, 1),))], "cpu")
    assert result["eligible"] == 0
    assert result["all_steps_exact"] == {"correct": 0, "count": 0}
    record = result["trajectories"][0]
    assert record["node"] == 2
    assert record["predicted"][0][2] == 0
    assert record["gold"][0][2] == 0
