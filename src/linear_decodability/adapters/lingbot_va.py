"""LingBot-VA binding (RoboTwin T-shaped layout, Wan2.2-TI2V-5B transformer).

Each latent frame is a 12x10 token grid: the two wrist cameras (128x160 each)
side by side in the top four rows, the head camera (256x320) below. The head
camera is therefore tokens 40-119 of a frame. Supervised layers are the
transformer blocks 9/19/29.

Requires the LingBot-VA source (``wan_va``, upstream revision
``7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb``, unmodified) on ``sys.path``
together with its own top-level packages: add both ``<lingbot-va>`` and
``<lingbot-va>/wan_va``. The checkpoint directory holds ``transformer/``,
``vae/``, ``text_encoder/`` and ``tokenizer/`` (e.g. the released
``lingbot-va-posttrain-robotwin``); ``--weights`` may point to another
``transformer`` directory with the same config.
"""
from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ..labels import Geometry
from .base import LocalizationOutput

GEOMETRY = Geometry(image_hw=(256, 320), grid=(8, 10))
LAYERS = (9, 19, 29)
HIDDEN_DIM = 3072
FRAME_TOKENS = 120
HEAD_TOKENS = slice(40, 120)
NATIVE_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_ROWS = 33
PROMPT_DROPOUT = 0.1
WINDOW_ACTIONS = 32
UNUSED_PREFIX = "condition_embedder_action.text_embedder."


def config():
    from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
    return deepcopy(va_robotwin_train_cfg)


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    a, b = values[..., :3], values[..., 3:]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def lingbot_pose16(value: np.ndarray) -> np.ndarray:
    """20-D ``[xyz, rot6d, gripper] x 2`` -> LingBot's 16-D ``[xyz, quat, gripper] x 2``.

    The quaternion is scipy's ``as_quat`` (xyzw) of the same rotation, which is
    the convention the released RoboTwin LingBot-VA policy was trained with.
    """
    from scipy.spatial.transform import Rotation
    arms = []
    for offset in (0, 10):
        arms.append(np.concatenate((value[:, offset:offset + 3],
                                    Rotation.from_matrix(rot6d_to_matrix(value[:, offset + 3:offset + 9])).as_quat(),
                                    value[:, offset + 9:offset + 10]), axis=1))
    return np.concatenate(arms, axis=1)


def action_codec(cfg):
    """LingBot's own ``_action_post_process`` on a minimal processor (no LeRobot dataset)."""
    from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset
    processor = LatentLeRobotDataset.__new__(LatentLeRobotDataset)
    processor.config = cfg
    processor.q01 = np.asarray(cfg.norm_stat["q01"], dtype=float)[None]
    processor.q99 = np.asarray(cfg.norm_stat["q99"], dtype=float)[None]

    def encode(absolute: np.ndarray):
        if len(absolute) != ACTION_ROWS:
            raise ValueError(f"Need {ACTION_ROWS} action rows, got {len(absolute)}")
        return processor._action_post_process(0, ACTION_ROWS, np.arange(0, ACTION_ROWS, 4), lingbot_pose16(absolute))
    return encode


def native_observation(images: dict) -> dict:
    from ..data.robotwin_if import CAMERAS
    return {f"observation.images.{native}": np.asarray(images[name], dtype=np.uint8)
            for name, native in zip(CAMERAS, NATIVE_CAMERAS)}


class Encoders:
    """LingBot's own streaming VAE and T5 (``VA_Server`` helpers, no server state)."""

    def __init__(self, checkpoint_dir: Path, device):
        from wan_va.wan_va_server import VA_Server
        self.server = VA_Server.__new__(VA_Server)
        self.server.device, self.server.dtype = torch.device(device), torch.bfloat16
        self.server.env_type, self.server.height, self.server.width = "robotwin_tshape", 256, 320
        self.server.job_config = SimpleNamespace(obs_cam_keys=[f"observation.images.{c}" for c in NATIVE_CAMERAS])
        self.checkpoint_dir = checkpoint_dir

    def load_vae(self):
        from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae
        server = self.server
        if getattr(server, "vae", None) is None:
            server.vae = load_vae(str(self.checkpoint_dir / "vae"), torch_dtype=torch.bfloat16,
                                  torch_device=server.device).eval().requires_grad_(False)
            server.streaming_vae = WanVAEStreamingWrapper(server.vae)
            server.streaming_vae_half = WanVAEStreamingWrapper(server.vae)

    def load_text(self):
        from wan_va.modules.utils import load_text_encoder, load_tokenizer
        server = self.server
        if getattr(server, "text_encoder", None) is None:
            server.text_encoder = load_text_encoder(str(self.checkpoint_dir / "text_encoder"), torch_dtype=torch.bfloat16,
                                                    torch_device=server.device).eval().requires_grad_(False)
            server.tokenizer = load_tokenizer(str(self.checkpoint_dir / "tokenizer"))

    def release(self):
        for name in ("vae", "streaming_vae", "streaming_vae_half", "text_encoder", "tokenizer"):
            setattr(self.server, name, None)

    @torch.no_grad()
    def video(self, frames: list[dict]) -> torch.Tensor:
        """First frame alone, then chunks of four (LingBot's causal streaming encoding)."""
        self.load_vae()
        server = self.server
        observations = [native_observation(frame) for frame in frames]
        server.streaming_vae.clear_cache()
        server.streaming_vae_half.clear_cache()
        chunks = [server._encode_obs({"obs": observations[:1]})]
        for start in range(1, len(observations), 4):
            chunks.append(server._encode_obs({"obs": observations[start:start + 4]}))
        return torch.cat(chunks, dim=2)[0].cpu()

    @torch.no_grad()
    def text(self, prompt: str) -> torch.Tensor:
        self.load_text()
        return self.server._get_t5_prompt_embeds(prompt=[prompt], max_sequence_length=512)[0].cpu()


class NativeLoss:
    """Upstream ``Trainer._prepare_input_dict`` / ``compute_loss`` on a minimal trainer state."""

    def __init__(self, cfg, device):
        from wan_va.train import Trainer
        from wan_va.utils import FlowMatchScheduler
        trainer = Trainer.__new__(Trainer)
        trainer.config, trainer.device, trainer.patch_size = cfg, torch.device(device), cfg.patch_size
        trainer.gradient_accumulation_steps = 1
        trainer.train_scheduler_latent = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0., extra_one_step=True)
        trainer.train_scheduler_latent.set_timesteps(1000, training=True)
        trainer.train_scheduler_action = FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0., extra_one_step=True)
        trainer.train_scheduler_action.set_timesteps(1000, training=True)
        self.trainer = trainer


def block_taps(transformer, layers=LAYERS):
    """Forward hooks on the checkpoint-wrapped blocks: fire once per forward, never on recomputation."""
    from contextlib import contextmanager

    @contextmanager
    def taps():
        values, handles = {}, []
        for layer in layers:
            def hook(module, args, output, layer=layer):
                values[layer] = output
            handles.append(transformer.blocks[layer].register_forward_hook(hook))
        try:
            yield values
        finally:
            for handle in handles:
                handle.remove()
    return taps()


class LingBotForward(nn.Module):
    """Single module handed to DeepSpeed: native window loss or localization forward."""

    def __init__(self, transformer, cfg, device):
        super().__init__()
        self.transformer = transformer
        self.native = NativeLoss(cfg, device)

    def forward(self, kind: str, sample: dict, seed: int):
        from wan_va.train import Trainer
        trainer = self.native.trainer
        device = trainer.device
        if kind == "native":
            batch = {k: v.to(device) for k, v in sample.items()}
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(int(seed))
                inputs = Trainer._prepare_input_dict(trainer, batch)
                output = self.transformer(inputs, train_mode=True)
                video, action = Trainer.compute_loss(trainer, inputs, output)
            return dict(loss=video + action, loss_video=video.detach(), loss_action=action.detach()), None, None
        from wan_va.utils import get_mesh_id
        first = sample["first_frame_latents"].to(device, torch.bfloat16)
        height, width = first.shape[-2:]
        t_video = trainer.train_scheduler_latent.timesteps[:1].to(device)
        t_action = trainer.train_scheduler_action.timesteps[:1].to(device)
        noise = torch.randn(first.shape, device=device, dtype=torch.bfloat16,
                            generator=torch.Generator(device=device).manual_seed(int(seed)))
        text = sample["text_emb"].to(device, torch.bfloat16)
        video_inputs = dict(noisy_latents=noise, latent=first, timesteps=t_video[None],
                            cond_timesteps=torch.zeros_like(t_video)[None],
                            grid_id=get_mesh_id(1, height // 2, width // 2, t=0, f_w=1, f_shift=0,
                                                action=False).to(device)[None], text_emb=text)
        zeros = torch.zeros(1, 30, 1, 4, 1, device=device, dtype=torch.bfloat16)
        action_inputs = dict(noisy_latents=zeros, latent=zeros.clone(), timesteps=t_action[None],
                             cond_timesteps=torch.zeros_like(t_action)[None],
                             actions_mask=torch.zeros_like(zeros, dtype=torch.bool),
                             grid_id=get_mesh_id(1, 4, 1, t=1, f_w=1, f_shift=0, action=True).to(device)[None],
                             text_emb=text)
        inputs = dict(latent_dict=video_inputs, action_dict=action_inputs, chunk_size=1, window_size=4)
        with block_taps(self.transformer) as captured:
            video, action = self.transformer(inputs, train_mode=True)
        tokens = (height // 2) * (width // 2)
        if tokens != FRAME_TOKENS or set(captured) != set(LAYERS):
            raise ValueError("Unexpected LingBot RoboTwin layout")
        hidden = {layer: captured[layer][:, tokens:2 * tokens][:, HEAD_TOKENS] for layer in LAYERS}
        return None, hidden, (video, action)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class LingBotVAAdapter:
    geometry = GEOMETRY
    layers = LAYERS
    hidden_dim = HIDDEN_DIM

    def __init__(self, checkpoint_dir, weights, device, cache_dir):
        from wan_va.distributed.fsdp import apply_ac
        from wan_va.modules.utils import load_transformer
        self.checkpoint_dir = Path(checkpoint_dir)
        transformer_dir = Path(weights) if weights else self.checkpoint_dir / "transformer"
        if not transformer_dir.is_absolute():
            transformer_dir = self.checkpoint_dir / transformer_dir
        self.transformer_dir = transformer_dir
        self.device = torch.device(device)
        self.cfg = config()
        transformer = load_transformer(str(transformer_dir), torch_dtype=torch.bfloat16, torch_device=self.device,
                                       attn_mode="flex")
        transformer.to(torch.bfloat16).train().requires_grad_(True)
        apply_ac(transformer)
        self.transformer = transformer
        for name, parameter in transformer.named_parameters():
            if name.startswith(UNUSED_PREFIX):
                parameter.requires_grad_(False)
        self.named = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
        self.module = LingBotForward(transformer, self.cfg, self.device)
        self.cache = Path(cache_dir)
        self.encoders = Encoders(self.checkpoint_dir, self.device)
        self.encode_actions = action_codec(self.cfg)

    def windows(self, bundle) -> list[dict]:
        """LingBot trains only on complete 32-step action windows."""
        return [w for w in bundle.windows() if w["start"] + WINDOW_ACTIONS <= w["training_frames"]]

    def action_parameter(self, name: str) -> bool:
        return name.split(".")[0] in ("action_embedder", "condition_embedder_action", "action_proj_out")

    def frozen_to(self, device) -> None:
        if torch.device(device).type == "cpu":
            self.encoders.release()

    def _text(self, prompt: str) -> str:
        key = text_key(prompt)
        path = self.cache / "text" / f"{key}.pt"
        if not path.exists():
            atomic_save(path, self.encoders.text(prompt))
        return key

    @torch.no_grad()
    def encode_window(self, bundle, window: dict) -> dict:
        from ..data.video import decode_frames
        episode = bundle.episodes[window["episode_id"]]
        arrays = episode.arrays()
        start = window["start"]
        frame_ids = (start + np.arange(9) * 4).tolist()
        decoded = {name: decode_frames(path, frame_ids) for name, path in episode.cameras().items()}
        frames = [{name: np.asarray(decoded[name][i]) for name in decoded} for i in range(len(frame_ids))]
        latents = self.encoders.video(frames)
        absolute = arrays["action"][np.minimum(start + np.arange(ACTION_ROWS), len(arrays["action"]) - 1)]
        actions, mask = self.encode_actions(absolute)
        if actions.shape[1] != latents.shape[1]:
            raise ValueError("LingBot video/action time dimensions differ")
        self._text("")
        return dict(latents=latents, actions=actions, actions_mask=mask, prompt_key=self._text(episode.instruction))

    @torch.no_grad()
    def encode_query(self, query) -> dict:
        from ..labels import query_labels
        arrays = query.arrays()
        latent = self.encoders.video([{name: arrays[name] for name in ("head_camera", "left_camera", "right_camera")}])
        target, competitor = query_labels(torch.from_numpy(arrays["target_mask"]).float(),
                                          [torch.from_numpy(arrays["other_mask"]).float()], self.geometry)
        return dict(first_frame_latents=latent[None], prompt_key=self._text(query.instruction),
                    target=target, competitor=competitor)

    def _embedding(self, key: str) -> torch.Tensor:
        return torch.load(self.cache / "text" / f"{key}.pt", map_location="cpu", weights_only=True)

    def sample(self, kind: str, record: dict) -> dict:
        value = torch.load(record["file"], map_location="cpu", weights_only=True)
        if kind == "native":
            key = value.pop("prompt_key")
            if torch.rand(1).item() < PROMPT_DROPOUT:
                key = text_key("")
            value["text_emb"] = self._embedding(key)
            return {k: v[None] for k, v in value.items()}
        value["text_emb"] = self._embedding(value.pop("prompt_key"))[None]
        return value

    def localization_forward(self, sample: dict, noise_seed: int) -> LocalizationOutput:
        _, hidden, outputs = self.module("localization", sample, noise_seed)
        return LocalizationOutput(hidden=hidden, native_outputs=outputs)

    def native_loss(self, sample: dict, noise_seed: int) -> torch.Tensor:
        return self.module("native", sample, noise_seed)[0]["loss"]

    def save(self, directory, step: int) -> Path:
        """Diffusers transformer directory loadable by LingBot-VA's own server; no readout heads."""
        import json
        from safetensors.torch import save_file
        target = Path(directory) / "checkpoints" / f"checkpoint_step_{step}" / "transformer"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "diffusion_pytorch_model.safetensors"
        temporary = path.with_suffix(".tmp.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in self.transformer.state_dict().items()}, str(temporary))
        temporary.replace(path)
        source = json.loads((self.transformer_dir / "config.json").read_text())
        (target / "config.json").write_text(json.dumps({k: v for k, v in source.items() if k != "_name_or_path"},
                                                       indent=2))
        return path
