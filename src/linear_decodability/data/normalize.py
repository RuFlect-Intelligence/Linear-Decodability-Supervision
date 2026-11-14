"""Min-max normalization and the 80-D unified action layout."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Released OpenWAM RoboTwin wire layout: 10 slots per arm, starting at 0 and 34.
ACTION_SLOTS = np.r_[0:10, 34:44]


def eef_statistics(path) -> dict[str, np.ndarray]:
    """Read the ``eef`` min/max/mean/std block of a released normalization file."""
    value = np.load(Path(path), allow_pickle=True).item()["eef"]
    for key in ("min", "max"):
        if np.asarray(value[key]).shape != (20,):
            raise ValueError(f"Require 20-D end-effector statistics, got {np.asarray(value[key]).shape}")
    return value


def normalize_scatter(values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Clip to ``[-1, 1]`` with the released min/max, then scatter into 80 dims."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 20:
        raise ValueError(f"Require 20-D absolute end-effector vectors, got {values.shape[-1]}")
    low, high = stats["min"].astype(np.float32), stats["max"].astype(np.float32)
    scaled = np.clip((values - low) / np.maximum(high - low, 1e-6) * 2 - 1, -1, 1)
    out = np.zeros((*scaled.shape[:-1], 80), dtype=np.float32)
    out[..., ACTION_SLOTS] = scaled
    return out
