"""Sub-token linear readout heads (training only)."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .loss import region_balanced_bce


class SubTokenReadout(nn.Module):
    """One position-shared ``Linear(dim -> S*S)`` per supervised layer.

    Each head reads a single token and cannot aggregate spatial context, so
    localization must already be linearly decodable from the backbone token.
    Heads run in FP32 with autocast disabled and never feed the policy.
    """

    def __init__(self, dim: int, layers: Sequence[int], tokens: int, sub: int = 4):
        super().__init__()
        self.dim, self.layers, self.tokens, self.sub = dim, tuple(layers), tokens, sub
        self.heads = nn.ModuleDict({str(layer): nn.Linear(dim, sub * sub) for layer in self.layers})

    def logits(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] != self.tokens or hidden.shape[2] != self.dim:
            raise ValueError(f"Layer {layer}: expected [B, {self.tokens}, {self.dim}], got {tuple(hidden.shape)}")
        head = self.heads[str(layer)]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return head(hidden.float())

    def forward(self, hidden: Mapping[int, torch.Tensor], target: torch.Tensor, competitor: torch.Tensor,
                *, detach: bool = False) -> torch.Tensor:
        """Mean region-balanced loss over layers; ``target``/``competitor`` are ``[B, tokens, S*S]``."""
        losses = []
        for layer in self.layers:
            h = hidden[layer].detach() if detach else hidden[layer]
            losses.append(region_balanced_bce(self.logits(layer, h), target, competitor))
        return torch.stack(losses).mean()


def default_layers(depth: int) -> tuple[int, int, int]:
    """Ends of the first, second, and final thirds (``(9, 19, 29)`` for 30 blocks)."""
    if depth < 3:
        raise ValueError("Need at least three blocks")
    return depth // 3 - 1, 2 * depth // 3 - 1, depth - 1
