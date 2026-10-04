"""Native LoReFT and low-rank DAS equations, independent of model architecture.

LoReFT: Wu et al. (2024), arXiv:2404.03592, eq. 2.
DII: h + R.T (R source - R h), eq. 1; Geiger et al. (2024).
QR parameterizes orthonormal columns; identity initialization is our choice.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .localization import hidden


class DAS(nn.Module):
    def __init__(self, hidden_size, rank, seed=42):
        super().__init__()
        if not 1 <= rank <= hidden_size:
            raise ValueError("rank must be between 1 and hidden size")
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(hidden_size, rank, generator=g))
        self.raw_basis = nn.Parameter(q)

    def basis(self):
        q, r = torch.linalg.qr(self.raw_basis.float(), mode="reduced")
        return q * torch.where(r.diag() < 0, -1.0, 1.0)

    def forward(self, h, source):
        q = self.basis()
        h32 = h.float()
        return (h32 + ((source.to(h32) - h32) @ q) @ q.T).to(h.dtype)


class LoReFT(DAS):
    def __init__(self, hidden_size, rank, seed=42):
        super().__init__(hidden_size, rank, seed)
        self.weight = nn.Parameter(self.basis().detach().T.contiguous())
        self.bias = nn.Parameter(torch.zeros(rank))

    def forward(self, h):
        q, h32 = self.basis(), h.float()
        return (h32 + (h32 @ self.weight.T + self.bias - h32 @ q) @ q.T).to(h.dtype)


@contextmanager
def frozen_model(lm):
    params = list(lm.model.parameters())
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, flags, strict=True):
            p.requires_grad_(flag)


def intervention_scores(lm, prompts, site, prefix, transform):
    """Differentiable one-position hook; never consumes the distinguishing digit."""
    ends = [len(lm._prompt_ids(p)) + len(prefix) - 1 for p in prompts for _ in (0, 1)]

    def hook(module, args, output):
        tensor = hidden(output)
        rows = torch.arange(len(ends), device=tensor.device)
        positions = torch.tensor(ends, device=tensor.device)
        changed = transform(tensor[rows, positions])
        result = tensor.clone()
        result[rows, positions] = changed.to(result)
        if isinstance(output, tuple):
            return (result, *output[1:])
        if isinstance(output, list):
            return [result, *output[1:]]
        return result

    handle = lm.model.get_submodule(site).register_forward_hook(hook)
    try:
        return lm.scores(prompts)
    finally:
        handle.remove()


def task_locality_loss(
    scores,
    baseline,
    labels,
    affected,
    locality_weight,
    *,
    normalization_counts=None,
):
    """CE on affected nodes + forward KL on unaffected nodes; binary candidates only."""
    logp = scores.float().log_softmax(-1)
    labels, affected = labels.to(scores.device), affected.to(scores.device)
    task_count, local_count = normalization_counts or (int(affected.sum()), int((~affected).sum()))
    zero = scores.sum() * 0
    task = (
        F.nll_loss(logp[affected], labels[affected], reduction="sum") / task_count
        if affected.any()
        else zero
    )
    locality = (
        F.kl_div(
            logp[~affected],
            baseline.detach().to(scores).softmax(-1)[~affected],
            reduction="sum",
        )
        / local_count
        if (~affected).any()
        else zero
    )
    return task + locality_weight * locality, {
        "task_ce": float(task.detach()),
        "locality_kl": float(locality.detach()),
    }
