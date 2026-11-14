import math

import pytest

from linear_decodability.schedule import budget_lr_scale
from linear_decodability.train import PermutationCursor, SharedStreamSchedule, SplitRankSchedule


def test_cursor_covers_each_index_once_per_epoch():
    cursor = PermutationCursor(7, seed=3)
    first, second = cursor.take(7), cursor.take(7)
    assert sorted(first) == sorted(second) == list(range(7))
    assert first != second


def test_split_rank_schedule_pairs_noise_and_resume():
    pairs = [[2 * i, 2 * i + 1] for i in range(5)]
    schedule = SplitRankSchedule(num_windows=11, pairs=pairs, world=8)
    windows, queries, noise = schedule.next()
    assert len(windows) == 4 and len(queries) == 4
    assert queries[0] // 2 == queries[1] // 2 and queries[2] // 2 == queries[3] // 2
    assert noise[:4] == [42, 43, 44, 45] and noise[4] == noise[5] != noise[6] == noise[7]
    assert schedule.item(0, windows, queries) == ("native", windows[0])
    assert schedule.item(5, windows, queries) == ("localization", queries[1])
    for _ in range(9):
        schedule.next()
    resumed = SplitRankSchedule(num_windows=11, pairs=pairs, world=8, step=10)
    assert resumed.next() == schedule.next()


def test_split_rank_schedule_rejects_bad_inputs():
    with pytest.raises(ValueError):
        SplitRankSchedule(num_windows=4, pairs=[[0, 1]], world=6)
    with pytest.raises(ValueError):
        SplitRankSchedule(num_windows=4, pairs=[[0, 1, 2]], world=8)


def test_shared_stream_schedule_matches_reference_and_resumes():
    import numpy as np
    pairs = [[2 * i, 2 * i + 1] for i in range(3)]
    schedule = SharedStreamSchedule(num_windows=5, pairs=pairs, world=8, seed=9)
    rng = np.random.default_rng(9)

    def endless(size):
        while True:
            yield from rng.permutation(size).tolist()
    window_order, pair_order, queue = endless(5), endless(3), []
    for step in range(12):
        while len(queue) < 4:
            queue.extend(pairs[next(pair_order)])
        expected_queries, queue = queue[:4], queue[4:]
        expected_windows = [next(window_order) for _ in range(4)]
        windows, queries, noise = schedule.next()
        assert (windows, queries) == (expected_windows, expected_queries)
        assert noise[:4] == [9 + step * 8 + rank for rank in range(4)]
        assert noise[4] == noise[5] == 9 + step * 8 + 10_000_000
    resumed = SharedStreamSchedule(num_windows=5, pairs=pairs, world=8, seed=9, step=12)
    assert resumed.next() == schedule.next()


def test_adapter_schedule_hook(tmp_path):
    pytest.importorskip("av")
    from test_adapters import tiny_bundle
    from linear_decodability.train.split_rank import adapter_schedule

    class Custom:
        def schedule(self, bundle, num_windows, world, step):
            return SharedStreamSchedule(num_windows, bundle.pairs(), world, seed=1, step=step)

    bundle = tiny_bundle(tmp_path / "bundle")
    assert isinstance(adapter_schedule(Custom(), bundle, 3, 8, 0), SharedStreamSchedule)
    assert isinstance(adapter_schedule(object(), bundle, 3, 8, 0), SplitRankSchedule)


def test_budget_lr_scale():
    assert budget_lr_scale(0, 0.0, 100.0) == pytest.approx(1 / 32)
    assert budget_lr_scale(100, 50.0, 100.0) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 2)))
    assert budget_lr_scale(100, 500.0, 100.0) == pytest.approx(0.1)
