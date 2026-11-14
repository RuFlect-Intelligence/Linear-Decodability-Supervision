"""What Linear-Decodability Supervision needs from a backbone.

An adapter binds one policy to the trainer. ``module`` is the single
``nn.Module`` handed to DeepSpeed; it must dispatch on ``kind``::

    module("native", sample, seed)       -> ({"loss": scalar, ...}, None, None)
    module("localization", sample, seed) -> (None, {layer: [1, tokens, dim]}, native_outputs)

``native_outputs`` are the policy's own prediction tensors (e.g. video and
action) from the localization forward; the trainer adds ``0 * sum(outputs)``
so every parameter stays in the autograd graph under ZeRO with zero gradient.
The native dict may also carry detached ``loss_video`` / ``loss_action`` for
logging.

Optional hooks, used when present:

- ``bind(bundle)``: called once per rank before caching, e.g. to export the
  bundle into the format the upstream dataset reads;
- ``windows(bundle) -> list[dict]``: the native training windows (default
  ``bundle.windows()``);
- ``schedule(bundle, num_windows, world, step)``: the per-update addressing
  (default ``SplitRankSchedule``), for reproducing a run that sampled differently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch import nn

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
    named: list[tuple[str, nn.Parameter]]
    module: nn.Module

    def __init__(self, checkpoint_dir, weights: str, device, cache_dir) -> None: ...

    def action_parameter(self, name: str) -> bool: ...

    def frozen_to(self, device) -> None: ...

    def encode_window(self, bundle, window: dict) -> dict: ...

    def encode_query(self, query) -> dict: ...

    def sample(self, kind: str, record: dict) -> dict: ...

    def native_loss(self, sample: dict, noise_seed: int) -> torch.Tensor: ...

    def localization_forward(self, sample: dict, noise_seed: int) -> LocalizationOutput: ...

    def save(self, directory, step: int) -> Path: ...
