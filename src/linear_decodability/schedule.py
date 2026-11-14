"""Loss weight and learning-rate schedules."""
from __future__ import annotations

import math

LOCALIZATION_WEIGHT = 0.25
WARMUP_UPDATES = 32
BACKBONE_LR = 1e-5
HEAD_LR = 3e-4
MIN_LR_RATIO = 0.1


def localization_weight(step: int, weight: float = LOCALIZATION_WEIGHT, warmup: int = WARMUP_UPDATES) -> float:
    """``lambda(s) = lambda0 * min(1, (s + 1) / s_w)``."""
    return weight * min(1.0, (step + 1) / warmup)


def lr_scale(step: int, total: int, warmup: int = WARMUP_UPDATES, floor: float = MIN_LR_RATIO) -> float:
    """Linear warm-up, then cosine decay to ``floor`` of the peak.

    ``step``/``total`` may be update counts or elapsed seconds for time-budgeted runs.
    """
    progress = min(1.0, step / total)
    return min(1.0, (step + 1) / warmup) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def budget_lr_scale(step: int, elapsed: float, budget: float, warmup: int = WARMUP_UPDATES,
                    floor: float = MIN_LR_RATIO) -> float:
    """Time-budgeted variant: warm-up by update count, cosine decay by elapsed wall-clock."""
    progress = min(1.0, elapsed / budget)
    return min(1.0, (step + 1) / warmup) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
