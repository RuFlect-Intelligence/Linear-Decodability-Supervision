"""Combining the native objective with the localization loss."""
from __future__ import annotations

from typing import Iterable

import torch


def zero_anchor(outputs: Iterable[torch.Tensor | None]) -> torch.Tensor:
    """``0 * sum(outputs)``: keeps every parameter in the autograd graph (ZeRO) with zero gradient."""
    values = [value.float().sum() for value in outputs if value is not None]
    if not values:
        raise ValueError("Need at least one native output to anchor")
    return torch.stack(values).sum() * 0


def split_rank_loss(is_native_rank: bool, native_loss: torch.Tensor | None = None,
                    localization_loss: torch.Tensor | None = None, anchor: torch.Tensor | None = None,
                    weight: float = 0.0) -> torch.Tensor:
    """Half the ranks: ``2 * L_native``; the other half: ``2 * lambda * L_loc + anchor``.

    Averaging gradients over all ranks then yields ``mean L_native + lambda * mean L_loc``.
    """
    if is_native_rank:
        return 2 * native_loss
    return 2 * weight * localization_loss + anchor


def colocated_loss(native_loss: torch.Tensor, localization_loss: torch.Tensor, anchor: torch.Tensor,
                   weight: float) -> torch.Tensor:
    """Every rank computes both terms."""
    return native_loss + weight * localization_loss + anchor
