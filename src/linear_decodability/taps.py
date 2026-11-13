"""Read-only capture of block outputs."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Mapping, Sequence

import torch
from torch import nn


@contextmanager
def capture(blocks: Mapping[int, nn.Module] | Sequence[nn.Module], layers: Sequence[int],
            select: Callable[[torch.Tensor], torch.Tensor] | None = None,
            output: Callable[[object], torch.Tensor] | None = None) -> Iterator[dict[int, torch.Tensor]]:
    """Forward hooks that record ``select(output(block_output))`` for each layer.

    Hooks return ``None`` and therefore never modify native hidden states.
    ``output`` extracts the hidden tensor from a block's return value
    (default: the value itself, or its first element for tuples); ``select``
    restricts it to the supervised head-camera tokens.
    """
    values: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(layer):
        def hook(module, args, result):
            hidden = output(result) if output else (result[0] if isinstance(result, (tuple, list)) else result)
            values[layer] = select(hidden) if select else hidden
        return hook

    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(hook_for(layer)))
        yield values
    finally:
        for handle in handles:
            handle.remove()
