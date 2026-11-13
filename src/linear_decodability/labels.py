"""Masks -> token-major sub-cell coverages."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Geometry:
    """Head-camera slot of a model input: pixel size, token grid, sub-cells per token side."""
    image_hw: tuple[int, int]
    grid: tuple[int, int]
    sub: int = 4

    @property
    def tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def fine_hw(self) -> tuple[int, int]:
        return self.grid[0] * self.sub, self.grid[1] * self.sub


def token_major(fine: torch.Tensor, grid: tuple[int, int], sub: int) -> torch.Tensor:
    """``[..., rows*sub, cols*sub]`` -> ``[..., rows*cols, sub*sub]`` matching head outputs."""
    rows, cols = grid
    lead = fine.shape[:-2]
    value = fine.reshape(*lead, rows, sub, cols, sub)
    value = value.movedim(-3, -2)
    return value.reshape(*lead, rows * cols, sub * sub)


def coverage(mask: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """Mask ``[H, W]`` in ``[0, 1]``, already in the model's image orientation -> ``[tokens, S*S]``.

    Antialiased bilinear resize into the model image, then area pooling to the
    sub-cell grid.
    """
    value = torch.as_tensor(mask, dtype=torch.float32)
    if value.ndim != 2:
        raise ValueError("Mask must be [H, W]")
    value = F.interpolate(value[None, None], size=geometry.image_hw, mode="bilinear",
                          align_corners=False, antialias=True)
    value = F.interpolate(value, size=geometry.fine_hw, mode="area")[0, 0].clamp(0, 1)
    return token_major(value, geometry.grid, geometry.sub)


def query_labels(target_mask, competitor_masks, geometry: Geometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Target coverage and the union of competitor coverages; raises if the target vanishes."""
    target = coverage(target_mask, geometry)
    if not torch.isfinite(target).all() or target.sum() <= 0:
        raise ValueError("Target coverage is empty after projection")
    competitor = torch.zeros_like(target)
    for mask in competitor_masks:
        competitor = torch.maximum(competitor, coverage(mask, geometry))
    return target, competitor
