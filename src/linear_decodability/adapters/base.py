"""What Linear-Decodability Supervision needs from a backbone."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from ..labels import Geometry


@dataclass
class LocalizationOutput:
    """Hidden states of the supervised head-camera tokens, plus native outputs for the zero anchor."""
    hidden: Mapping[int, torch.Tensor]
    native_outputs: Sequence[torch.Tensor]


@runtime_checkable
class Adapter(Protocol):
    """Model binding.

    ``localization_forward`` runs the policy's own training forward with the
    clean current observation, the instruction, and pure noise at the maximal
    diffusion timestep for everything the policy would predict. It returns the
    tapped head-camera tokens ``[B, tokens, dim]`` for each supervised layer.
    """

    geometry: Geometry
    layers: Sequence[int]
    hidden_dim: int

    def native_loss(self, batch: Any) -> torch.Tensor: ...

    def localization_forward(self, query: Any, noise_seed: int) -> LocalizationOutput: ...

    def action_parameter(self, name: str) -> bool: ...
