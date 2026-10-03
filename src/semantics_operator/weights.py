"""Effective W = W_base + sum(B @ A / sqrt(rank)); base tensors stay untouched."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WeightPatch:
    a: Tensor
    b: Tensor

    @property
    def scale(self) -> float:
        return 1 / math.sqrt(self.a.shape[0])

    def norm(self) -> float:
        # ||BA||_F^2 = tr((B^T B)(A A^T)), without constructing the full matrix.
        squared = torch.sum((self.b.T @ self.b) * (self.a @ self.a.T))
        return float(squared.clamp_min(0).sqrt() * self.scale)


def randomized(patch: WeightPatch, seed: int) -> WeightPatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(patch.a.shape, generator=generator)
    b = torch.randn(patch.b.shape, generator=generator)
    result = WeightPatch(a, b)
    b *= patch.norm() / max(result.norm(), 1e-20)
    return result


class EditedLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if type(rank) is not int or rank < 1:
            raise ValueError("rank must be a positive integer")
        self.base = base
        self.a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        self.scale = 1 / math.sqrt(rank)
        self.fitting = True
        self.patches: tuple[WeightPatch, ...] = ()
        self.reset(0)

    def reset(self, seed: int):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.a.copy_(torch.randn(self.a.shape, generator=generator) * 0.02)
            self.b.zero_()
        self.a.grad = self.b.grad = None
        self.fitting = True
        self.patches = ()

    def delta(self) -> Tensor:
        return (self.b @ self.a) * self.scale

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if self.fitting:
            return y + (F.linear(F.linear(x.float(), self.a), self.b) * self.scale).to(y.dtype)
        for patch in self.patches:
            y = y + (F.linear(F.linear(x.float(), patch.a), patch.b) * patch.scale).to(y.dtype)
        return y


class WeightSession:
    """One model per session/thread. Removal and requires_grad flags restore on exceptions."""

    def __init__(self, model: nn.Module, target: str, rank: int):
        if not target:
            raise ValueError("A named Linear module is required")
        base = model.get_submodule(target)
        if type(base) is not nn.Linear:
            raise ValueError(f"{target} must be an unquantized torch.nn.Linear")
        parent_name, _, self.child_name = target.rpartition(".")
        self.parent = model.get_submodule(parent_name) if parent_name else model
        self.base = base
        self.model = model
        self.layer = EditedLinear(base, rank)
        self.active = False

    def __enter__(self):
        if self.active or self.parent.get_submodule(self.child_name) is not self.base:
            raise RuntimeError("WeightSession is already active or target changed")
        self.flags = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in self.flags:
            p.requires_grad_(False)
        setattr(self.parent, self.child_name, self.layer)
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        setattr(self.parent, self.child_name, self.base)
        for parameter, flag in self.flags:
            parameter.requires_grad_(flag)
        self.active = False

    def parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.layer.a, self.layer.b

    def reset(self, seed: int):
        self.layer.reset(seed)

    def snapshot(self) -> WeightPatch:
        return WeightPatch(self.layer.a.detach().cpu().clone(), self.layer.b.detach().cpu().clone())

    @contextmanager
    def branch(self, patches: tuple[WeightPatch, ...]):
        if not self.active:
            raise RuntimeError("Enter the WeightSession before selecting a branch")
        old_fitting, old_patches = self.layer.fitting, self.layer.patches
        device = self.layer.a.device
        self.layer.fitting = False
        self.layer.patches = tuple(WeightPatch(p.a.to(device), p.b.to(device)) for p in patches)
        try:
            yield
        finally:
            self.layer.fitting, self.layer.patches = old_fitting, old_patches
