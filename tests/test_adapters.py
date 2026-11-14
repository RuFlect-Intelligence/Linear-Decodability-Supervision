import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def tiny_bundle(root: Path):
    av = pytest.importorskip("av")
    from linear_decodability.data import Bundle, CAMERAS
    (root / "episodes" / "ep0").mkdir(parents=True)
    (root / "queries").mkdir()
    frames = 40
    pose = np.tile(np.r_[0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.5], 2).astype(np.float32)
    np.savez(root / "episodes" / "ep0.npz", action=np.tile(pose, (frames, 1)), state=np.tile(pose, (frames, 1)),
             valid_action=np.ones(frames, bool))
    for camera in CAMERAS:
        with av.open(str(root / "episodes" / "ep0" / f"{camera}.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=25)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(frames):
                image = np.full((48, 64, 3), index * 5, np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    rows = []
    for index, instruction in enumerate(("pick the red block", "pick the blue block")):
        target = np.zeros((48, 64), np.float32)
        other = np.zeros((48, 64), np.float32)
        target[:, :32] = 1 if index == 0 else 0
        target[:, 32:] = 0 if index == 0 else 1
        other[:] = 1 - target
        query = {camera: np.full((48, 64, 3), 100, np.uint8) for camera in CAMERAS}
        np.savez(root / "queries" / f"q{index}.npz", **query, target_mask=target, other_mask=other,
                 **{"observation.state": pose})
        rows.append(dict(id=f"q{index}", instruction=instruction, scene_spec_id="s0", panel="pick"))
    (root / "episodes.jsonl").write_text(json.dumps(dict(episode_id="ep0", instruction="pick the red block",
                                                         scene_spec_id="s0", training_frames=frames)) + "\n")
    (root / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "manifest.json").write_text(json.dumps(dict(schema="robotwin_if_release_v1", files={})))
    return Bundle(root)


def load_toy():
    spec = importlib.util.spec_from_file_location("toy_adapter", ROOT / "examples" / "toy_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["toy_adapter"] = module
    spec.loader.exec_module(module)
    return module


def test_toy_adapter_passes_contract_checks(tmp_path):
    from linear_decodability.verify import check
    bundle = tiny_bundle(tmp_path / "bundle")
    assert bundle.pairs() == [[0, 1]]
    toy = load_toy()
    adapter = toy.ToyAdapter(tmp_path, "", "cpu", tmp_path / "cache")
    report = check(adapter, bundle, tmp_path / "cache")
    assert report["action_parameters"] == 4 and report["backbone_parameters_with_grad"] > 0
    path = adapter.save(tmp_path, 3)
    assert path.exists() and not any("heads" in key for key in torch.load(path))


def test_contract_check_catches_action_leak(tmp_path):
    from linear_decodability.verify import check
    bundle = tiny_bundle(tmp_path / "bundle")
    toy = load_toy()

    class Leaky(toy.ToyAdapter):
        def localization_forward(self, sample, noise_seed):
            output = super().localization_forward(sample, noise_seed)
            hidden = {k: v + self.policy.action_out.bias.sum() for k, v in output.hidden.items()}
            return type(output)(hidden=hidden, native_outputs=output.native_outputs)

    with pytest.raises(AssertionError, match="action parameter"):
        check(Leaky(tmp_path, "", "cpu", tmp_path / "cache"), bundle, tmp_path / "cache")


def test_lingbot_pose_conversion_is_rotation_preserving():
    pytest.importorskip("scipy")
    from scipy.spatial.transform import Rotation
    from linear_decodability.adapters.lingbot_va import lingbot_pose16
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 1.1])
    matrix = rotation.as_matrix()
    arm = np.r_[0.1, 0.2, 0.3, matrix[:, 0], matrix[:, 1], 0.7]
    pose = lingbot_pose16(np.tile(arm, 2)[None])
    assert pose.shape == (1, 16)
    assert np.allclose(np.abs(pose[0, 3:7] @ rotation.as_quat()), 1)
    assert pose[0, 7] == pytest.approx(0.7)
