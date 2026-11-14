"""OpenWAM binding (dual_system joint_self_attn, Wan2.2-TI2V-5B video branch).

Head camera occupies the top 256x320 slot of the native 384x320 L-shaped
canvas: the first 80 tokens (8x10) of the first latent frame. Supervised
layers are the video DiT blocks 9/19/29.

Requires an OpenWAM checkout importable as ``openwam`` (pinned upstream
revision ``d8dd33d8576b475f5a5cdc6fb8ca902778a199b3``); no upstream patch is
needed.
"""
from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..labels import Geometry
from .base import LocalizationOutput

GEOMETRY = Geometry(image_hw=(256, 320), grid=(8, 10))
LAYERS = (9, 19, 29)
HIDDEN_DIM = 3072
HEAD_TOKENS = slice(0, 80)
CANVAS_HW = (384, 320)
LOCALIZATION_TIMESTEP = 1000.0
FUTURE_LATENT_SHAPE = (48, 3, 24, 20)
ACTION_SHAPE = (32, 80)
FREEZE = ("text_encoder", "vae", "video_encoder", "image_encoder")


def stable_step(self, layer_id, vstate, astate, *, attn_mask, offload):
    """Checkpointed block step whose closure does not retain its own outputs.

    Same math as ``DualSystemMoTDriver._step_checkpointed``; the closure keeps
    templates without the two overwritten tensor fields, which avoids a
    reference cycle that otherwise holds every layer's activations.
    """
    video_template = copy.copy(vstate)
    video_template.hidden_states = None
    action_template = copy.copy(astate)
    payload_template = copy.copy(astate.payload)
    action_template.payload = payload_template
    payload_template.x_action = None

    def run(vx, ax):
        video = copy.copy(video_template)
        action = copy.copy(action_template)
        payload = copy.copy(payload_template)
        action.payload = payload
        video.hidden_states = vx
        payload.x_action = ax
        self._step_impl(layer_id, video, action, attn_mask=attn_mask, suppress_inner_attn_ckpt=True)
        return video.hidden_states, payload.x_action

    context = torch.autograd.graph.save_on_cpu() if offload else nullcontext()
    with context:
        vx, ax = torch.utils.checkpoint.checkpoint(run, vstate.hidden_states, astate.payload.x_action,
                                                   use_reentrant=False)
    vstate.hidden_states = vx
    astate.payload.x_action = ax
    return vstate, astate


@contextmanager
def video_block_taps(arch, layers=LAYERS, tokens=120):
    """Read-only capture of video block outputs through ``post_attn_at_layer``."""
    backbone = arch.video_backbone
    original = backbone.post_attn_at_layer
    values: dict[int, torch.Tensor] = {}

    def post(layer, state, attention, post_state):
        result = original(layer, state, attention, post_state)
        if layer in layers:
            values[layer] = result.hidden_states[:, :tokens]
        return result

    backbone.post_attn_at_layer = post
    try:
        yield values
    finally:
        backbone.post_attn_at_layer = original


def load_architecture(checkpoint_dir, weights: str, device):
    """Released checkpoint directory -> training-ready architecture."""
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver

    DualSystemMoTDriver._step_checkpointed = stable_step
    _, arch = load_from_checkpoint_dir(str(checkpoint_dir), device=str(device), ckpt_name=weights or None)
    arch.init_training_schedulers(1000)
    arch.set_training_runtime(use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
    return arch


def trainable_parameters(arch) -> list[tuple[str, nn.Parameter]]:
    """Everything except the frozen VAE / text / image encoders."""
    arch.requires_grad_(True)
    for name in FREEZE:
        module = getattr(arch.video_backbone, name, None)
        if isinstance(module, nn.Module):
            module.requires_grad_(False).eval()
    return [(name, p) for name, p in arch.named_parameters() if p.requires_grad]


def prompt(instruction: str) -> str:
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    return format_prompt_for_inference(instruction)


def canvas(images: dict[str, Any]):
    from PIL import Image
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout
    from ..data.robotwin_if import CAMERAS
    frames = {name: images[name] if isinstance(images[name], Image.Image)
              else Image.fromarray(np.asarray(images[name], dtype=np.uint8)) for name in CAMERAS}
    return assemble_multiview_layout(frames, list(CAMERAS), *CANVAS_HW)


class TextCache:
    """Frozen T5 embeddings keyed by prompt."""

    def __init__(self, arch, directory):
        self.arch = arch
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @torch.no_grad()
    def __call__(self, text: str) -> dict[str, torch.Tensor]:
        path = self.directory / f"{self.key(text)}.pt"
        if not path.exists():
            from openwam.model.video_backbone.wan import encode
            backbone = self.arch.video_backbone
            context, lengths = encode.encode_text([text], tokenizer=backbone._tokenizer,
                                                  text_encoder=backbone.text_encoder, device=self.arch.device)
            atomic_save(path, dict(context=context.cpu(), seq_lens=lengths.cpu(), prompt=text))
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["prompt"] != text:
            raise ValueError(f"Text cache collision: {path}")
        return value

    @contextmanager
    def patched(self):
        """Route ``encode.encode_text`` through the cache during ``prepare_inputs``."""
        from openwam.model.video_backbone.wan import encode
        original = encode.encode_text

        def cached(prompts, **kwargs):
            encode.encode_text = original
            try:
                values = [self(p) for p in prompts]
            finally:
                encode.encode_text = cached
            return (torch.cat([v["context"] for v in values]).to(self.arch.device),
                    torch.cat([v["seq_lens"] for v in values]).to(self.arch.device))

        encode.encode_text = cached
        try:
            yield
        finally:
            encode.encode_text = original


def atomic_save(path: Path, value) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


class OpenWAMForward(nn.Module):
    """Single module handed to DeepSpeed: native window loss or localization forward."""

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

    def forward(self, kind: str, sample: dict, seed: int):
        arch = self.arch
        device, dtype = torch.device(arch.device), arch.dtype
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            if kind == "native":
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["inputs"].items()}
                inputs.update(context=sample["context"].to(device), seq_lens=sample["seq_lens"].to(device),
                              use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
                return arch.compute_loss(**inputs, lambda_video=1.0, lambda_action=1.0), None, None
            future = torch.randn((1, *FUTURE_LATENT_SHAPE), device=device, dtype=dtype)
            observed = sample["first_frame_latents"].to(device)
            future[:, :, :1] = observed
            actions = torch.randn((1, *ACTION_SHAPE), device=device, dtype=dtype)
            timestep = torch.full((1,), LOCALIZATION_TIMESTEP, device=device, dtype=dtype)
            inputs = dict(latents=future, first_frame_latents=observed,
                          context=sample["context"].to(device), seq_lens=sample["seq_lens"].to(device),
                          height=CANVAS_HW[0], width=CANVAS_HW[1], num_frames=9, fuse_vae_embedding_in_latents=True,
                          num_clean_prefix_frames=0, use_gradient_checkpointing=True,
                          use_gradient_checkpointing_offload=False, cfg_scale=1.0, cfg_merge=False, tiled=False)
            with video_block_taps(arch) as captured:
                video, action = arch(actions, timestep, timestep=timestep,
                                     proprio=sample["proprio"].to(device=device, dtype=dtype), **inputs)
            hidden = {layer: captured[layer][:, HEAD_TOKENS] for layer in LAYERS}
        return None, hidden, (video, action)


class OpenWAMAdapter:
    geometry = GEOMETRY
    layers = LAYERS
    hidden_dim = HIDDEN_DIM

    def __init__(self, checkpoint_dir, weights: str, device, cache_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.arch = load_architecture(self.checkpoint_dir, weights, device)
        self.named = trainable_parameters(self.arch)
        self.module = OpenWAMForward(self.arch)
        self.cache = Path(cache_dir)
        self.text = TextCache(self.arch, self.cache / "text")
        from ..data.normalize import eef_statistics
        self.stats = eef_statistics(self.checkpoint_dir / "normalization_stats.npy")

    def action_parameter(self, name: str) -> bool:
        return name.startswith("action_backbone.")

    def frozen_to(self, device) -> None:
        self.arch.move_frozen_to_device(torch.device(device), names=("text_encoder", "vae"))

    @torch.no_grad()
    def encode_window(self, bundle, window: dict) -> dict:
        """Native training sample with frozen VAE latents; T5 context stored by prompt key."""
        from ..data.normalize import ACTION_SLOTS, normalize_scatter
        from ..data.video import decode_frames
        episode = bundle.episodes[window["episode_id"]]
        arrays = episode.arrays()
        action_ids, video_ids, action_valid, video_valid = bundle.window_indices(window["start"], window["training_frames"])
        frames = {name: decode_frames(path, video_ids.tolist()) for name, path in episode.cameras().items()}
        video = [canvas({name: frames[name][i] for name in frames}) for i in range(len(video_ids))]
        dims = np.zeros(80, bool)
        dims[ACTION_SLOTS] = True
        text = prompt(episode.instruction)
        sample = dict(video=video, first_frame_image=[video[0]], vace_video=None, prompt=text,
                      action=normalize_scatter(arrays["action"][action_ids], self.stats),
                      proprio=normalize_scatter(arrays["state"][[window["start"]]], self.stats),
                      action_mask=action_valid[:, None] & dims[None], proprio_mask=dims[None], video_mask=video_valid)
        with self.text.patched():
            inputs = self.arch.prepare_inputs([sample])
        inputs.pop("context")
        inputs.pop("seq_lens")
        return dict(inputs=cpu_tree(inputs), prompt_key=self.text.key(text), prompt=text)

    @torch.no_grad()
    def encode_query(self, query) -> dict:
        from openwam.model.video_backbone.wan import encode
        from ..data.normalize import normalize_scatter
        from ..labels import query_labels
        arrays = query.arrays()
        backbone = self.arch.video_backbone
        pixels = encode.preprocess_video([canvas(arrays)], encoder=backbone.video_encoder,
                                         dtype=backbone.dtype, device=backbone.device)
        latent = encode.encode_video(pixels, vae=backbone.vae, encoder=backbone.video_encoder).cpu()
        text = prompt(query.instruction)
        self.text(text)
        target, competitor = query_labels(torch.from_numpy(arrays["target_mask"]).float(),
                                          [torch.from_numpy(arrays["other_mask"]).float()], self.geometry)
        return dict(first_frame_latents=latent, proprio=torch.from_numpy(normalize_scatter(arrays["state"][None], self.stats)),
                    prompt_key=self.text.key(text), target=target, competitor=competitor)

    def sample(self, kind: str, record: dict) -> dict:
        value = torch.load(record["file"], map_location="cpu", weights_only=True)
        text = torch.load(self.cache / "text" / f"{value['prompt_key']}.pt", map_location="cpu", weights_only=True)
        value.update(context=text["context"], seq_lens=text["seq_lens"])
        return value

    def localization_forward(self, sample: dict, noise_seed: int) -> LocalizationOutput:
        _, hidden, outputs = self.module("localization", sample, noise_seed)
        return LocalizationOutput(hidden=hidden, native_outputs=outputs)

    def native_loss(self, sample: dict, noise_seed: int) -> torch.Tensor:
        return self.module("native", sample, noise_seed)[0]["loss"]

    def save(self, directory, step: int) -> Path:
        """Deployable OpenWAM checkpoint directory (same format as the release); no readout heads."""
        import shutil
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", "normalization_stats.npy"):
            if not (directory / name).exists():
                shutil.copyfile(self.checkpoint_dir / name, directory / name)
        if not (directory / "tokenizer").exists():
            shutil.copytree(self.checkpoint_dir / "tokenizer", directory / "tokenizer")
        path = directory / f"checkpoint_step_{step}.safetensors"
        temporary = path.with_suffix(".tmp.safetensors")
        state = {k: v.detach().cpu().contiguous() for k, v in self.arch.state_dict().items()}
        self.arch.save_checkpoint(str(temporary), state_dict=state)
        temporary.replace(path)
        return path
