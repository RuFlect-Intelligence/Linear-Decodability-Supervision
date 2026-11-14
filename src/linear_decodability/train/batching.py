"""Global per-update addressing for split-rank training (deterministic, resumable)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

NATIVE_SEED = 42
QUERY_SEED = 100_042
QUERY_NOISE_OFFSET = 10_000_000


class PermutationCursor:
    """Endless stream over ``range(size)``; a fresh permutation (``seed + epoch``) per epoch."""

    def __init__(self, size: int, seed: int, consumed: int = 0):
        if size < 1:
            raise ValueError("Empty training source")
        self.size, self.seed, self.consumed = size, seed, consumed
        self._epoch, self._order = None, None

    def take(self, count: int) -> list[int]:
        result = []
        for _ in range(count):
            epoch, offset = divmod(self.consumed, self.size)
            if self._epoch != epoch:
                self._order = np.random.default_rng(self.seed + epoch).permutation(self.size)
                self._epoch = epoch
            result.append(int(self._order[offset]))
            self.consumed += 1
        return result


@dataclass
class SplitRankSchedule:
    """Ranks ``[0, world/2)`` take native windows, the rest take localization queries.

    Queries are drawn as whole instruction pairs (both instructions for the same
    observed view in one update); the two members of a pair share a noise seed.
    """

    num_windows: int
    pairs: Sequence[Sequence[int]]
    world: int
    step: int = 0
    windows: PermutationCursor = field(init=False)
    queries: PermutationCursor = field(init=False)

    def __post_init__(self):
        if self.world < 2 or self.world % 4:
            raise ValueError("Split-rank training needs a world size divisible by 4")
        if any(len(pair) != 2 for pair in self.pairs):
            raise ValueError("Every localization view needs exactly two instructions")
        per = self.world // 2
        self.windows = PermutationCursor(self.num_windows, NATIVE_SEED, self.step * per)
        self.queries = PermutationCursor(len(self.pairs), QUERY_SEED, self.step * per // 2)

    @property
    def native_ranks(self) -> int:
        return self.world // 2

    def is_native(self, rank: int) -> bool:
        return rank < self.native_ranks

    def next(self) -> tuple[list[int], list[int], list[int]]:
        """Window ids, query ids, and one noise seed per rank for the next update."""
        per = self.native_ranks
        windows = self.windows.take(per)
        queries = [index for group in self.queries.take(per // 2) for index in self.pairs[group]]
        base = NATIVE_SEED + self.step * self.world
        noise = [base + rank for rank in range(per)]
        noise += [base + QUERY_NOISE_OFFSET + rank // 2 for rank in range(per)]
        self.step += 1
        return windows, queries, noise

    def item(self, rank: int, windows: list[int], queries: list[int]) -> tuple[str, int]:
        if self.is_native(rank):
            return "native", windows[rank]
        return "localization", queries[rank - self.native_ranks]


class SharedStreamSchedule:
    """Split-rank addressing where windows and pairs come from one shared RNG stream.

    Windows and instruction pairs are each an endless sequence of permutations
    drawn lazily from a single ``numpy.random.default_rng(seed)``, so the two
    streams interleave. Used by adapters whose reference runs sampled this way
    (InternW0-Delta on RoboTwin-IF). Same rank layout and noise seeds as
    ``SplitRankSchedule``.
    """

    def __init__(self, num_windows: int, pairs: Sequence[Sequence[int]], world: int, seed: int, step: int = 0):
        if world < 2 or world % 4:
            raise ValueError("Split-rank training needs a world size divisible by 4")
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("Every localization view needs exactly two instructions")
        self.num_windows, self.pairs, self.world, self.seed = num_windows, [list(p) for p in pairs], world, seed
        rng = np.random.default_rng(seed)

        def endless(size):
            while True:
                yield from rng.permutation(size).tolist()
        self._windows, self._pairs, self._queue = endless(num_windows), endless(len(self.pairs)), []
        self.step = 0
        for _ in range(step):
            self.next()

    @property
    def native_ranks(self) -> int:
        return self.world // 2

    def is_native(self, rank: int) -> bool:
        return rank < self.native_ranks

    def next(self) -> tuple[list[int], list[int], list[int]]:
        per = self.native_ranks
        while len(self._queue) < per:
            self._queue.extend(self.pairs[next(self._pairs)])
        queries, self._queue = self._queue[:per], self._queue[per:]
        windows = [next(self._windows) for _ in range(per)]
        base = self.seed + self.step * self.world
        noise = [base + rank for rank in range(per)] + [base + QUERY_NOISE_OFFSET + rank // 2 for rank in range(per)]
        self.step += 1
        return windows, queries, noise

    def item(self, rank: int, windows: list[int], queries: list[int]) -> tuple[str, int]:
        if self.is_native(rank):
            return "native", windows[rank]
        return "localization", queries[rank - self.native_ranks]
