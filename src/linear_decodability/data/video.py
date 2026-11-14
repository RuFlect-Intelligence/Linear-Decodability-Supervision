"""Frame-exact video decoding for the released episode videos (requires ``av``)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def decode_frames(path, indices: Sequence[int]):
    """Return RGB PIL frames at zero-based ``indices`` (order and duplicates preserved).

    Decodes sequentially from the first frame so numbering never depends on
    container seek accuracy.
    """
    if not indices:
        return []
    import av

    wanted = [int(index) for index in indices]
    if any(index < 0 for index in wanted):
        raise ValueError("Frame indices must be nonnegative")
    targets, last = set(wanted), max(wanted)
    found = {}
    with av.open(str(Path(path))) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in targets:
                found[index] = frame.to_image()
            if index >= last:
                break
    missing = targets.difference(found)
    if missing:
        raise RuntimeError(f"Missing frames {sorted(missing)} in {path}")
    return [found[index].copy() for index in wanted]
