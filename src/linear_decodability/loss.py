"""Region-balanced localization objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F

REGION_BUDGETS = (0.50, 0.45, 0.05)


def region_balanced_bce(logits, target, competitor, budgets=REGION_BUDGETS):
    """Coverage-weighted BCE over target, competitor, and background regions.

    ``logits``, ``target`` and ``competitor`` share shape ``[B, ...]``; coverages
    lie in ``[0, 1]`` and act as spatial weights, so every covered site is pushed
    toward probability one (target) or zero (competitor, background). Target
    overlap takes precedence over competitors. Regions with zero area drop out
    and the remaining budgets are renormalized per query.
    """
    z = logits.float().flatten(1)
    t = target.to(device=z.device, dtype=torch.float32).flatten(1)
    o = competitor.to(device=z.device, dtype=torch.float32).flatten(1)
    if z.shape != t.shape or t.shape != o.shape:
        raise ValueError(f"Shape mismatch: logits {tuple(z.shape)}, target {tuple(t.shape)}, competitor {tuple(o.shape)}")
    if (t.sum(-1) <= 0).any():
        raise ValueError("Every query needs a non-empty target region")
    o = o * (t == 0)
    background = ((t + o) == 0).float()
    total = torch.zeros_like(z[:, 0])
    mass = torch.zeros_like(z[:, 0])
    for budget, mask, error in ((budgets[0], t, F.softplus(-z)),
                                (budgets[1], o, F.softplus(z)),
                                (budgets[2], background, F.softplus(z))):
        area = mask.sum(-1)
        present = (area > 0).float() * budget
        total = total + present * (mask * error).sum(-1) / area.clamp_min(1e-12)
        mass = mass + present
    return (total / mass).mean()
