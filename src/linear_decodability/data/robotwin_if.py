"""Reader for the released RoboTwin-IF episode and localization-query bundle.

Layout (all paths relative to the bundle root)::

    manifest.json          schema, file hashes, counts
    episodes.jsonl         one row per expert episode
    queries.jsonl          one row per localization query
    episodes/<id>.npz      action/state (T, 20) absolute end-effector,
                           joint_action/joint_state (T, 14) joints
    episodes/<id>/<cam>.mp4  head_camera, left_camera, right_camera
    queries/<id>.npz       head/left/right uint8, observation.state (20,),
                           joint_state (14,), target_mask, other_mask

Instruction pairs share ``scene_spec_id``; the two rows of a pair label the
same observed view.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")
WINDOW_STRIDE = 16
NUM_FRAMES = 9
VIDEO_STRIDE = 4
NUM_ACTIONS = 32


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    instruction: str
    scene_spec_id: str
    training_frames: int
    root: Path

    def arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.root / f"{self.episode_id}.npz", allow_pickle=False) as data:
            return {key: data[key] for key in ("action", "state", "valid_action")}

    def cameras(self) -> dict[str, Path]:
        return {name: self.root / self.episode_id / f"{name}.mp4" for name in CAMERAS}


@dataclass(frozen=True)
class Query:
    id: str
    instruction: str
    scene_spec_id: str
    panel: str
    root: Path

    def arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.root / f"{self.id}.npz", allow_pickle=False) as data:
            values = {name: data[name] for name in CAMERAS}
            values["state"] = data["observation.state"]
            if "joint_state" in data:
                values["joint_state"] = data["joint_state"]
            values["target_mask"] = data["target_mask"]
            values["other_mask"] = data["other_mask"]
        return values


class Bundle:
    """Verified RoboTwin-IF release bundle."""

    def __init__(self, root, *, verify: bool = True):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("schema") != "robotwin_if_release_v1":
            raise ValueError(f"Unknown bundle schema: {manifest.get('schema')}")
        self.manifest = manifest
        if verify:
            for name, digest in manifest["files"].items():
                if sha256(self.root / name) != digest:
                    raise ValueError(f"Bundle file changed: {name}")
        self.episodes = {row["episode_id"]: Episode(row["episode_id"], row["instruction"],
                                                   row["scene_spec_id"], int(row["training_frames"]),
                                                   self.root / "episodes")
                         for row in read_jsonl(self.root / "episodes.jsonl")}
        self.queries = [Query(row["id"], row["instruction"], row["scene_spec_id"], row["panel"],
                              self.root / "queries")
                        for row in read_jsonl(self.root / "queries.jsonl")]

    def windows(self) -> list[dict]:
        """Native training windows: every ``WINDOW_STRIDE`` frames of every episode."""
        rows = []
        for episode in sorted(self.episodes.values(), key=lambda e: e.episode_id):
            for start in range(0, episode.training_frames, WINDOW_STRIDE):
                rows.append(dict(id=f"{episode.episode_id}_f{start:06d}", episode_id=episode.episode_id,
                                 start=start, training_frames=episode.training_frames))
        return rows

    def window_indices(self, start: int, length: int):
        """Action ids, video ids, and their validity masks (native RoboTwin alignment)."""
        if not 0 <= start < length:
            raise ValueError(f"Window start {start} outside episode of {length} frames")
        count = min(NUM_ACTIONS, length - start)
        actions = np.minimum(start + np.arange(NUM_ACTIONS), length - 1)
        video = np.minimum(start + np.arange(0, 33, VIDEO_STRIDE), length)
        return actions, video, np.arange(NUM_ACTIONS) < count, np.arange(0, 33, VIDEO_STRIDE) <= count

    def pairs(self) -> list[list[int]]:
        """Indices of the two original instructions of each observed view."""
        grouped: dict[tuple[str, str], list[int]] = {}
        for index, query in enumerate(self.queries):
            grouped.setdefault((query.panel, query.scene_spec_id), []).append(index)
        bad = [key for key, indices in grouped.items() if len(indices) != 2]
        if bad:
            raise ValueError(f"Every localization view needs exactly two instructions: {bad[:3]}")
        return [grouped[key] for key in sorted(grouped)]
