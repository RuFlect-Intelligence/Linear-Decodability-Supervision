"""Exported policies contain no readout heads."""
from __future__ import annotations

from typing import Mapping

import torch

HEAD_PREFIXES = ("readout.", "heads.")


def strip_readout(state_dict: Mapping[str, torch.Tensor], prefixes=HEAD_PREFIXES) -> dict[str, torch.Tensor]:
    """Drop readout parameters if they were registered on the policy module."""
    return {key: value for key, value in state_dict.items() if not key.startswith(tuple(prefixes))}
