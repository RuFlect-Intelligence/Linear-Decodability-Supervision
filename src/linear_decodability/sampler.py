"""Epoch-wise permutation of localization queries shared across ranks."""
from __future__ import annotations

from typing import Iterator

import torch


def query_stream(num_queries: int, per_update: int, seed: int = 0, start_update: int = 0) -> Iterator[list[int]]:
    """Yield the query indices of each update; every query is visited once per epoch.

    Queries are kept in order inside each epoch so instruction pairs stored
    adjacently (``2k``, ``2k+1``) stay together when ``per_update`` is even.
    """
    if num_queries <= 0 or per_update <= 0:
        raise ValueError("Need positive num_queries and per_update")
    pair = per_update % 2 == 0 and num_queries % 2 == 0
    unit = 2 if pair else 1
    buffer: list[int] = []
    epoch = 0
    produced = 0
    while True:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(num_queries // unit, generator=generator).tolist()
        buffer.extend(index * unit + offset for index in order for offset in range(unit))
        epoch += 1
        while len(buffer) >= per_update:
            batch, buffer = buffer[:per_update], buffer[per_update:]
            if produced >= start_update:
                yield batch
            produced += 1
