"""Smallest complete adapter: a toy world-action model on CPU.

Copy this file when binding a new model. Every method maps to one item of the
adapter contract in ``linear_decodability.adapters.base``; replace the toy
pieces (``ToyPolicy``, the fake encoders) with your model's own code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from linear_decodability.adapters.base import LocalizationOutput
from linear_decodability.labels import Geometry, query_labels
from linear_decodability.taps import capture

GEOMETRY = Geometry(image_hw=(32, 40), grid=(4, 5), sub=2)
LAYERS = (0, 2)
DIM = 16


class ToyPolicy(nn.Module):
    """Patch tokens of the head camera + an action query token through shared blocks."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(3 * 8 * 8, DIM)
        self.text = nn.Embedding(256, DIM)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM), nn.GELU()) for _ in range(3))
        self.video_out = nn.Linear(DIM, 3 * 8 * 8)
        self.action_in = nn.Linear(4, DIM)
        self.action_out = nn.Linear(DIM, 4)

    def forward(self, image, text_ids, noisy_action):
        patches = image.unfold(2, 8, 8).unfold(3, 8, 8).permute(0, 2, 3, 1, 4, 5).flatten(3).flatten(1, 2)
        tokens = self.patch(patches) + self.text(text_ids).mean(1, keepdim=True)
        action = self.action_in(noisy_action)[:, None]
        hidden = torch.cat([tokens, action], 1)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.video_out(hidden[:, :-1]), self.action_out(hidden[:, -1])


def text_ids(instruction: str) -> torch.Tensor:
    return torch.tensor([list(instruction.encode()[:16].ljust(16, b" "))])


def image_tensor(frame) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(frame, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    return nn.functional.interpolate(value, size=GEOMETRY.image_hw, mode="bilinear", align_corners=False)


class ToyForward(nn.Module):
    """The module handed to DeepSpeed: dispatches on ``kind`` (see ``adapters/base.py``)."""

    def __init__(self, policy, geometry, layers):
        super().__init__()
        self.policy, self.geometry, self.layers = policy, geometry, layers

    def forward(self, kind: str, sample: dict, seed: int):
        device = next(self.policy.parameters()).device
        generator = torch.Generator().manual_seed(int(seed))
        image, text = sample["image"][None].to(device), sample["text"][None].to(device)
        if kind == "native":
            action = sample["action"][None].to(device)
            noise = torch.randn(action.shape, generator=generator).to(device)
            video, predicted = self.policy(image, text, action + noise)
            video_loss, action_loss = 0.1 * video.pow(2).mean(), nn.functional.mse_loss(predicted, -noise)
            return dict(loss=video_loss + action_loss, loss_video=video_loss.detach(),
                        loss_action=action_loss.detach()), None, None
        noise = torch.randn((1, 4), generator=generator).to(device)
        with capture(self.policy.blocks, self.layers, select=lambda h: h[:, :self.geometry.tokens]) as hidden:
            video, action = self.policy(image, text, noise)
        return None, dict(hidden), (video, action)


class ToyAdapter:
    geometry = GEOMETRY
    layers = LAYERS
    hidden_dim = DIM

    def __init__(self, checkpoint_dir, weights, device, cache_dir):
        torch.manual_seed(0)
        self.policy = ToyPolicy().to(device)
        if weights:
            self.policy.load_state_dict(torch.load(Path(checkpoint_dir) / weights, map_location=device))
        self.device = torch.device(device)
        self.named = list(self.policy.named_parameters())
        self.module = ToyForward(self.policy, self.geometry, self.layers)

    def action_parameter(self, name: str) -> bool:
        return name.startswith(("action_in.", "action_out."))

    def frozen_to(self, device) -> None:
        pass

    def encode_window(self, bundle, window: dict) -> dict:
        from linear_decodability.data.video import decode_frames
        episode = bundle.episodes[window["episode_id"]]
        arrays = episode.arrays()
        frame = decode_frames(episode.cameras()["head_camera"], [window["start"]])[0]
        return dict(image=image_tensor(frame)[0], text=text_ids(episode.instruction)[0],
                    action=torch.from_numpy(arrays["action"][window["start"], :4]).float())

    def encode_query(self, query) -> dict:
        arrays = query.arrays()
        target, competitor = query_labels(torch.from_numpy(arrays["target_mask"]).float(),
                                          [torch.from_numpy(arrays["other_mask"]).float()], self.geometry)
        return dict(image=image_tensor(arrays["head_camera"])[0], text=text_ids(query.instruction)[0],
                    target=target, competitor=competitor)

    def sample(self, kind: str, record: dict) -> dict:
        return torch.load(record["file"], map_location="cpu", weights_only=True)

    def native_loss(self, sample: dict, noise_seed: int) -> torch.Tensor:
        return self.module("native", sample, noise_seed)[0]["loss"]

    def localization_forward(self, sample: dict, noise_seed: int) -> LocalizationOutput:
        _, hidden, outputs = self.module("localization", sample, noise_seed)
        return LocalizationOutput(hidden=hidden, native_outputs=outputs)

    def save(self, directory, step: int) -> Path:
        path = Path(directory) / f"toy_step_{step}.pt"
        torch.save(self.policy.state_dict(), path)
        return path
