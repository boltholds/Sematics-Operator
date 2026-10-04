"""One-hop neural updates with a directly readable scalar state per node."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .executor_tasks import oracle_trace


@dataclass
class Inputs:
    parents: torch.Tensor
    rules: torch.Tensor
    initial: torch.Tensor
    fixed_values: torch.Tensor
    free: torch.Tensor
    valid: torch.Tensor


def pack(episodes, steps, device="cpu"):
    if not episodes:
        raise ValueError("Empty episode batch")
    b, n = len(episodes), max(len(e.graph.parents) for e in episodes)
    parents = torch.zeros((b, n), dtype=torch.long, device=device)
    rules = torch.zeros_like(parents)
    initial = torch.zeros((b, n), device=device)
    free = torch.zeros((b, n), dtype=torch.bool, device=device)
    valid, protected = torch.zeros_like(free), torch.zeros_like(free)
    gold = torch.zeros((b, steps + 1, n), device=device)
    for j, e in enumerate(episodes):
        size = len(e.graph.parents)
        parents[j, :size] = torch.tensor(e.graph.parents, device=device)
        rules[j, :size] = torch.tensor(e.graph.rules, device=device)
        initial[j, :size] = torch.tensor(e.initial(), device=device)
        free[j, :size] = torch.tensor(e.free(), device=device)
        protected[j, :size] = torch.tensor(e.protected(), device=device)
        valid[j, :size] = True
        gold[j, :, :size] = torch.tensor(oracle_trace(e, steps), device=device)
    return Inputs(parents, rules, initial, initial.clone(), free, valid), gold, protected


class LocalExecutor(nn.Module):
    def __init__(self, kind="transformer", width=64, blocks=2, heads=4):
        super().__init__()
        self.kind = kind
        if kind == "transformer":
            self.value = nn.Linear(1, width)
            self.rule = nn.Embedding(3, width)
            self.role = nn.Embedding(2, width)
            layer = nn.TransformerEncoderLayer(
                width,
                heads,
                width * 2,
                dropout=0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.core = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            # TransformerEncoder clones initialization; initialize blocks independently.
            for layer in self.core.layers:
                for name, param in layer.named_parameters():
                    if param.ndim > 1:
                        nn.init.xavier_uniform_(param)
                    elif "norm" in name and "weight" in name:
                        nn.init.ones_(param)
                    else:
                        nn.init.zeros_(param)
            self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        elif kind == "mlp":
            self.core = nn.Sequential(
                nn.Linear(5, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
        else:
            raise ValueError(f"Unknown executor: {kind}")

    def forward(self, state, parents, rules):
        parent = state.gather(1, parents)
        if self.kind == "mlp":
            features = torch.cat(
                (state[..., None], parent[..., None], F.one_hot(rules, 3).to(state.dtype)), dim=-1
            )
            return self.core(features).squeeze(-1).sigmoid()
        b, n = state.shape
        values = torch.stack((state, parent), dim=-1).reshape(b * n, 2, 1)
        tokens = self.value(values)
        tokens = tokens + self.rule(rules.reshape(-1))[:, None, :]
        tokens = tokens + self.role.weight[None, :, :]
        return self.readout(self.core(tokens)[:, 0]).reshape(b, n).sigmoid()


def rollout(model, inputs, steps, start=None):
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    state = inputs.initial if start is None else start
    state = torch.where(inputs.free, state, inputs.fixed_values)
    trace = [state]
    for _ in range(steps):
        predicted = model(state, inputs.parents, inputs.rules)
        state = torch.where(inputs.free, predicted, inputs.fixed_values)
        trace.append(state)
    return torch.stack(trace, dim=1)
