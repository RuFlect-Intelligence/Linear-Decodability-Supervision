"""InternW0-Delta binding (RoboTwin, Wan2.2-TI2V-5B video expert in a mixture of transformers).

The native RoboTwin canvas is 384x256: the head camera (256x256) on top, the
two wrist cameras side by side below. Each latent frame is a 12x8 token grid,
so the head camera is tokens 0-63 of a frame. A training sequence is
``[anchor, recent, current, future x2]``; the supervised tokens are the head
camera of the *current* observation, tokens 192-255. Supervised layers are the
video-expert blocks 9/19/29.

Native training reuses the upstream LeRobot dataset (with its own memory
frames, padding, augmentation, normalization) and ``frame_training_loss``
unchanged. The adapter writes a LeRobot v2.1 view of the RoboTwin-IF bundle
(parquet of joint14 action/state, videos symlinked) and the upstream T5 text
cache, then reads native samples through the upstream dataset.

Requires the InternW0-Delta source (upstream revision
``90801baa3bdc7829c1e4989edfe3d0fb2410ea6a``, unmodified) on ``sys.path``:
add both ``<internw0>`` and ``<internw0>/src``. ``--checkpoint-dir`` is the
upstream ``checkpoints/`` root (``docs/models.md``): ``robotwin.pt``,
``RynnBrain1.1-2B/``, ``Wan-AI/Wan2.2-TI2V-5B``, ``Wan-AI/Wan2.1-T2V-1.3B``
and ``DiffSynth-Studio/Wan-Series-Converted-Safetensors``. A Wan directory
holding the original ``Wan2.2_VAE.pth`` / ``models_t5_umt5-xxl-enc-bf16.pth``
and ``google/umt5-xxl`` also works. ``--weights`` is a policy ``.pt``
(default ``robotwin.pt``).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ..labels import Geometry
from .base import LocalizationOutput

# Labels are projected through 256x320 before the 32x32 sub-cell grid, as in
# the paper run (the head slot itself is 256x256; coverages differ by < 0.012).
GEOMETRY = Geometry(image_hw=(256, 320), grid=(8, 8))
LAYERS = (9, 19, 29)
HIDDEN_DIM = 3072
TOKENS_PER_FRAME = 96
CURRENT_FRAME = 2
HEAD_TOKENS = slice(CURRENT_FRAME * TOKENS_PER_FRAME, CURRENT_FRAME * TOKENS_PER_FRAME + 64)
CANVAS_HW = (384, 256)
ACTION_HORIZON = 32
ACTION_DIM = 80
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
CONTEXT_LEN = 128
FPS = 25
NATIVE_CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
VLM_VIEWS = "cam_high|cam_left_wrist|cam_right_wrist"
SAMPLING_SEED = 2026102727
ACTION_PREFIXES = ("mot.mixtures.action.", "action_proprio_encoder.")
# With VLM conditioning the action expert never calls its T5 text embedding.
UNUSED_PREFIX = "mot.mixtures.action.text_embedding."


def source_root() -> Path:
    import wam
    return Path(wam.__file__).resolve().parents[2]


def text_file(directory: Path, prompt: str) -> Path:
    """Upstream text-cache file name (``RobotVideoDataset._get_cached_text_context``)."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path(directory) / f"{digest}.t5_len{CONTEXT_LEN}.wan22ti2v5b.pt"


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def wan_layout(root: Path) -> dict:
    """Model ids for the upstream loader: release layout, or one Wan dir with the original ``.pth`` files."""
    wan = root / "Wan-AI" / "Wan2.2-TI2V-5B"
    if (wan / "Wan2.2_VAE.pth").exists():
        tokenizer = wan if (wan / "google" / "umt5-xxl").exists() else root / "Wan-AI" / "Wan2.1-T2V-1.3B"
        return dict(model_id=str(wan), tokenizer_model_id=str(tokenizer), redirect_common_files=False)
    return dict(model_id="Wan-AI/Wan2.2-TI2V-5B", tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
                redirect_common_files=True)


def configuration(root: Path, export: Path, text: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from wam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    os.environ["WAM_CHECKPOINT_ROOT"] = str(root)
    source = source_root()
    with initialize_config_dir(config_dir=str(source / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=robotwin"])
    with open_dict(cfg):
        cfg.paths.checkpoints = str(root)
        cfg.model.update(wan_layout(root))
        cfg.model.load_text_encoder = False
        cfg.model.understanding.vlm_model_path = str(root / "RynnBrain1.1-2B")
        train = cfg.data.train
        train.dataset_dirs = [str(export)]
        train.pretrained_norm_stats = str(source / "assets" / "stats" / "robotwin.json")
        train.text_embedding_cache_dir = str(text)
        train.episode_selection = None
        train.global_sample_stride = 1
        train.load_episode_stats = False
        train.check_local_files = True
        train.action_hz = FPS
    return cfg


def strict_load(model, path) -> dict:
    """Load a native ``.pt``; reject the upstream loader's permissive missing / shape-skip behaviour."""
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    counts = {}
    for name in ("mot", "proprio_encoder", "action_proprio_encoder", "track_bridge"):
        module = getattr(model, name)
        if module is None:
            if payload.get(name):
                raise ValueError(f"Checkpoint unexpectedly contains {name}")
            continue
        state, expected = payload.get(name), module.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError(f"Incomplete InternW0 checkpoint keys for {name}")
        if any(tuple(state[k].shape) != tuple(expected[k].shape) for k in expected):
            raise ValueError(f"Incompatible InternW0 checkpoint shapes for {name}")
        module.load_state_dict(state, strict=True)
        counts[name] = len(state)
    if model.understanding is not None:
        state = payload.get("understanding")
        if not isinstance(state, dict) or set(state) != set(model.understanding.adapter_state_dict()):
            raise ValueError("Incomplete InternW0 understanding adapter state")
        model.understanding.load_adapter_state_dict(state, strict=True)
    return counts


def export_lerobot(bundle, output: Path) -> dict:
    """LeRobot v2.1 view of the bundle for the upstream dataset (joint14, 25 Hz, videos symlinked)."""
    import pandas as pd
    output = Path(output)
    receipt = output / "export.json"
    with file_lock(output.parent / f".{output.name}.lock"):
        if receipt.exists():
            return json.loads(receipt.read_text())
        episodes = sorted(bundle.episodes.values(), key=lambda e: e.episode_id)
        instructions = sorted({e.instruction for e in episodes})
        rows, offset = [], 0
        for index, episode in enumerate(episodes):
            with np.load(episode.root / f"{episode.episode_id}.npz", allow_pickle=False) as data:
                action, state = data["joint_action"], data["joint_state"]
            n = episode.training_frames
            if action.shape[1:] != (14,) or state.shape[1:] != (14,) or len(action) < n:
                raise ValueError(f"{episode.episode_id}: InternW0 needs joint14 action/state")
            table = pd.DataFrame(dict(action=list(action[:n]), **{"observation.state": list(state[:n])},
                                      timestamp=np.arange(n, dtype=np.float32) / FPS,
                                      frame_index=np.arange(n, dtype=np.int64),
                                      episode_index=np.full(n, index, np.int64),
                                      task_index=np.full(n, instructions.index(episode.instruction), np.int64),
                                      index=np.arange(offset, offset + n, dtype=np.int64)))
            path = output / f"data/chunk-000/episode_{index:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
            for native, camera in NATIVE_CAMERAS.items():
                link = output / f"videos/chunk-000/observation.images.{native}/episode_{index:06d}.mp4"
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    link.symlink_to(episode.cameras()[camera].resolve())
            rows.append(dict(episode_index=index, length=n, tasks=[episode.instruction]))
            offset += n
        features = {k: dict(dtype="int64", shape=[1], names=None)
                    for k in ("episode_index", "frame_index", "index", "task_index")}
        features["timestamp"] = dict(dtype="float32", shape=[1], names=None)
        for key in ("action", "observation.state"):
            features[key] = dict(dtype="float32", shape=[14], names=[f"joint14_{i}" for i in range(14)])
        for native in NATIVE_CAMERAS:
            features[f"observation.images.{native}"] = dict(
                dtype="video", shape=[240, 320, 3], names=["height", "width", "channels"],
                video_info={"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False, "has_audio": False})
        info = dict(codebase_version="v2.1", robot_type="aloha_agilex", total_episodes=len(episodes),
                    total_frames=offset, total_tasks=len(instructions), total_videos=3 * len(episodes),
                    total_chunks=1, chunks_size=1000, fps=FPS, splits={"train": f"0:{len(episodes)}"},
                    data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                    features=features)
        meta = output / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text(json.dumps(info, indent=2))
        (meta / "tasks.jsonl").write_text("".join(json.dumps(dict(task_index=i, task=t)) + "\n"
                                                  for i, t in enumerate(instructions)))
        (meta / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        value = dict(episodes=[e.episode_id for e in episodes], frames=offset, instructions=instructions)
        temporary = receipt.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(receipt)
        return value


@torch.no_grad()
def prepare_text(cfg, prompts, directory: Path, device) -> int:
    """Upstream T5 text cache (``tools/text_cache.py`` format) for the given prompts."""
    from wam.model.backbones.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from wam.model.backbones.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    with file_lock(Path(directory) / ".lock"):
        todo = sorted({p for p in prompts if not text_file(directory, p).exists()})
        if not todo:
            return 0
        _, text, _, tokenizer = _resolve_configs(model_id=cfg.model.model_id,
                                                 tokenizer_model_id=cfg.model.tokenizer_model_id,
                                                 redirect_common_files=bool(cfg.model.redirect_common_files))
        text.download_if_necessary()
        tokenizer.download_if_necessary()
        encoder = _load_registered_model(text.path, "wan_video_text_encoder", torch_dtype=torch.bfloat16,
                                         device=str(device)).eval()
        tokenize = HuggingfaceTokenizer(name=tokenizer.path, seq_len=CONTEXT_LEN, clean="whitespace")
        for start in range(0, len(todo), 4):
            batch = todo[start:start + 4]
            ids, mask = tokenize(batch, return_mask=True, add_special_tokens=True)
            mask = mask.to(device=device, dtype=torch.bool)
            context = encoder(ids.to(device), mask)
            for i, prompt in enumerate(batch):
                path = text_file(directory, prompt)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(dict(context=context[i].cpu().contiguous(), mask=mask[i].cpu().contiguous()), temporary)
                temporary.replace(path)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return len(todo)


def cached_context(directory: Path, instruction: str):
    """Same as the upstream dataset: padded positions zeroed, all-ones mask."""
    payload = torch.load(text_file(directory, PROMPT.format(task=instruction)), map_location="cpu", weights_only=True)
    context, mask = payload["context"].clone(), payload["mask"].bool()
    context[~mask] = 0.0
    return context[None], torch.ones_like(mask)[None]


def deploy_inputs(cameras: dict, state, processor, device, dtype):
    """Official RoboTwin deploy preprocessing: canvas, normalized proprio, VLM images."""
    from eval.robotwin.policy import RobotWinPolicySession
    holder = SimpleNamespace(video_height=CANVAS_HW[0], video_width=CANVAS_HW[1], processor=processor,
                             model=SimpleNamespace(device=device, torch_dtype=dtype))
    observation = dict(observation={k: dict(rgb=np.ascontiguousarray(cameras[k])) for k in NATIVE_CAMERAS.values()},
                       joint_action=dict(vector=np.asarray(state, np.float32)))
    return (RobotWinPolicySession._build_robotwin_image_tensor(holder, observation),
            RobotWinPolicySession._normalize_state(holder, observation["joint_action"]["vector"]),
            RobotWinPolicySession._build_robotwin_vlm_images(holder, observation))


def all_parameter_zero(model) -> torch.Tensor:
    """Every trainable tensor in the graph with zero gradient (identical ZeRO membership on all ranks)."""
    return sum(p.reshape(-1)[0].float() * 0 for p in model.parameters() if p.requires_grad)


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    return []


def localization_forward(model, sample: dict, seed: int):
    """First native denoising step from the clean current observation (no future frames or actions).

    Memory: the current frame as anchor, a padded recent slot, then the current
    frame and two future latents of pure noise at t=1000; actions are pure noise
    at t=1000. This is the prefix the deployed policy builds at its first step.
    """
    from wam.model.modules.conditioning.frame_attention import build_frame_memory_attention_mask
    from wam.model.modules.memory.proprio_encoder import append_proprio_to_context
    from wam.model.modules.understanding.sequence_conditioning import build_frame_vlm_pack
    device, dtype = model.device, model.torch_dtype
    values = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
              for k, v in sample.items() if torch.is_tensor(v)}
    current = values["first_frame_latents"]
    if current.shape != (1, 48, 1, 24, 16):
        raise ValueError(f"Unexpected InternW0 current VAE shape: {tuple(current.shape)}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    future = torch.randn((1, 48, 2, 24, 16), generator=generator).to(device=device, dtype=dtype)
    action = torch.randn((1, ACTION_HORIZON, ACTION_DIM), generator=generator).to(device=device, dtype=dtype)
    latents = torch.cat([current, torch.zeros_like(current), current, future], dim=2)
    context, context_mask = append_proprio_to_context(model, values["context"], values["context_mask"],
                                                      values["proprio"])
    vlm = build_frame_vlm_pack(model, inputs=dict(vlm_context_cache=values["vlm_context"],
                                                  vlm_mask_cache=values["vlm_mask"], current_proprio=values["proprio"]))
    video = model.video_expert.pre_dit(
        x=latents, timestep=torch.tensor([[0, 0, 1000, 1000]], device=device, dtype=dtype), context=context,
        context_mask=context_mask, action=None,
        fuse_vae_embedding_in_latents=bool(model.video_expert.fuse_vae_embedding_in_latents),
        video_layout="single", video_view_names="canvas")
    model.video_expert.append_future_delta_queries(video, current_frame_index=CURRENT_FRAME)
    acting = model.action_expert.pre_dit(action_tokens=action,
                                         timestep=torch.full((1,), 1000, device=device, dtype=dtype),
                                         context=vlm["vlm_context"], context_mask=vlm["vlm_mask"])
    per_frame = int(video["meta"]["tokens_per_frame"])
    if per_frame != TOKENS_PER_FRAME:
        raise ValueError(f"Expected {TOKENS_PER_FRAME} tokens per InternW0 canvas frame, got {per_frame}")
    delta = int(video["meta"].get("num_future_delta_tokens", 0))
    mask = build_frame_memory_attention_mask(video_seq_len=video["tokens"].shape[1], action_seq_len=ACTION_HORIZON,
                                             video_tokens_per_frame=per_frame, num_condition_frames=3,
                                             device=device, num_future_delta_tokens=delta,
                                             num_anchor_condition_frames=1)
    valid = torch.tensor([[True, False, True, True, True]], device=device).repeat_interleave(per_frame, dim=1)
    valid = torch.cat([valid, torch.ones((1, delta), device=device, dtype=torch.bool)], dim=1)
    tokens, hidden = model.mot(
        embeds_all=dict(video=video["tokens"], action=acting["tokens"]), attention_mask=mask,
        freqs_all=dict(video=video["freqs"], action=acting["freqs"]),
        context_all=dict(video=dict(context=video["context"], mask=video["context_mask"]),
                         action=dict(context=acting["context"], mask=acting["context_mask"])),
        t_mod_all=dict(video=video["t_mod"], action=acting["t_mod"]),
        key_masks_all=dict(video=valid, action=torch.ones((1, ACTION_HORIZON), device=device, dtype=torch.bool)),
        capture_layers={"video": set(LAYERS)})
    predicted_video = model.video_expert.post_dit(tokens["video"], video)
    predicted_action = model.action_expert.post_dit(tokens["action"], acting)
    taps = {layer: hidden["video"][layer][:, HEAD_TOKENS] for layer in LAYERS}
    return taps, [*tensors(predicted_video), *tensors(predicted_action), all_parameter_zero(model)]


def scope_native_reduction(group) -> None:
    """Upstream loss statistics are all-reduced over the native ranks only (localization ranks never join)."""
    import torch.distributed as dist
    from wam.training import losses

    def reduce(values):
        result = values.detach().clone()
        if not dist.is_available() or not dist.is_initialized():
            return result, 1
        dist.all_reduce(result, group=group)
        return result, dist.get_world_size(group=group)
    losses._distributed_sum_detached = reduce


def freeze_unused(model) -> None:
    """Freeze parameters no forward path reads; otherwise AdamW weight decay alone would change them."""
    if model.action_expert.uses_vlm_conditioning:
        for name, parameter in model.named_parameters():
            if name.startswith(UNUSED_PREFIX):
                parameter.requires_grad_(False)


class InternW0Forward(nn.Module):
    """Single module handed to DeepSpeed: upstream ``frame_training_loss`` or the localization forward."""

    def __init__(self, model):
        super().__init__()
        self.wam = model

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.wam.set_trainable_modules()
            freeze_unused(self.wam)
        return self

    def forward(self, kind: str, sample: dict, seed: int):
        if kind == "native":
            from wam.training.losses import frame_training_loss
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(seed))
            loss, parts = frame_training_loss(self.wam, sample)
            return dict(loss=loss + all_parameter_zero(self.wam), loss_video=parts["loss_video"],
                        loss_action=parts["loss_action"]), None, None
        hidden, outputs = localization_forward(self.wam, sample, seed)
        return None, hidden, outputs


class InternW0DeltaAdapter:
    geometry = GEOMETRY
    layers = LAYERS
    hidden_dim = HIDDEN_DIM

    def __init__(self, checkpoint_dir, weights, device, cache_dir):
        from hydra.utils import instantiate
        from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        self.root = Path(checkpoint_dir).resolve()
        self.weights = Path(weights) if weights else self.root / "robotwin.pt"
        if not self.weights.is_absolute():
            self.weights = self.root / self.weights
        self.device = torch.device(device)
        self.cache = Path(cache_dir)
        self.export, self.text = self.cache / "lerobot_v21", self.cache / "text"
        self.cfg = configuration(self.root, self.export, self.text)
        self.model = instantiate(self.cfg.model, model_dtype=torch.bfloat16, device=str(self.device))
        self.load_counts = strict_load(self.model, self.weights)
        self.model.set_trainable_modules()
        freeze_unused(self.model)
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(self.cfg.data.train.pretrained_norm_stats))
        self.named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        self.module = InternW0Forward(self.model)
        self.dataset = None

    def bind(self, bundle) -> None:
        """Called on every rank: scope native loss statistics, write the LeRobot view and T5 cache."""
        import torch.distributed as dist
        group = None
        if dist.is_available() and dist.is_initialized():
            group = dist.new_group(ranks=list(range(dist.get_world_size() // 2)))
        scope_native_reduction(group)
        export_lerobot(bundle, self.export)
        prompts = {PROMPT.format(task=e.instruction) for e in bundle.episodes.values()}
        prompts |= {PROMPT.format(task=q.instruction) for q in bundle.queries}
        prepare_text(self.cfg, prompts, self.text, self.device)

    def windows(self, bundle) -> list[dict]:
        """Every valid-action row of every episode (stride 1), as indices into the upstream dataset."""
        rows, offset = [], 0
        for episode in sorted(bundle.episodes.values(), key=lambda e: e.episode_id):
            rows.extend(dict(id=f"{episode.episode_id}_f{start:06d}", episode_id=episode.episode_id, start=start,
                             training_frames=episode.training_frames, index=offset + start)
                        for start in range(episode.training_frames))
            offset += episode.training_frames
        return rows

    def schedule(self, bundle, num_windows: int, world: int, step: int):
        """Paper sampling: windows and (original, counterfactual) pairs from one RNG stream."""
        from ..train.batching import SharedStreamSchedule
        trained = {e.scene_spec_id: e.instruction for e in bundle.episodes.values()}
        groups: dict[str, list[int]] = {}
        for index, query in enumerate(bundle.queries):
            groups.setdefault(query.scene_spec_id, []).append(index)
        pairs = [sorted(groups[scene], key=lambda i: bundle.queries[i].instruction != trained.get(scene))
                 for scene in sorted(groups)]
        return SharedStreamSchedule(num_windows, pairs, world, SAMPLING_SEED, step=step)

    def action_parameter(self, name: str) -> bool:
        return name.startswith(ACTION_PREFIXES)

    def frozen_to(self, device) -> None:
        """VAE and VLM are part of the native forward and stay on the training device."""

    def encode_window(self, bundle, window: dict) -> dict:
        return dict(index=int(window["index"]))

    @torch.no_grad()
    def encode_query(self, query) -> dict:
        from wam.model.modules.codecs.video_latent_codec import encode_input_image_latents_tensor
        from ..labels import query_labels
        arrays = query.arrays()
        if "joint_state" not in arrays:
            raise ValueError(f"{query.id}: InternW0 needs the joint14 query state")
        model = self.model
        image, proprio, rgb = deploy_inputs(arrays, arrays["joint_state"], self.processor, model.device,
                                            model.torch_dtype)
        latent = encode_input_image_latents_tensor(model, input_image=image, tiled=False, video_layout="single",
                                                   video_view_names="canvas")
        vlm = model.understanding(frames=rgb[None, None], prompts=[PROMPT.format(task=query.instruction)],
                                  view_names=[VLM_VIEWS], frame_labels=["current"])
        context, mask = cached_context(self.text, query.instruction)
        target, competitor = query_labels(torch.from_numpy(arrays["target_mask"]).float(),
                                          [torch.from_numpy(arrays["other_mask"]).float()], self.geometry)
        return dict(first_frame_latents=latent.cpu(), proprio=proprio.cpu(), context=context, context_mask=mask,
                    vlm_context=vlm["vlm_context"].cpu(), vlm_mask=vlm["vlm_mask"].cpu(),
                    target=target, competitor=competitor)

    def _dataset(self):
        if self.dataset is None:
            from hydra.utils import instantiate
            from wam.utils import misc
            work = self.cache / "native_work"
            work.mkdir(parents=True, exist_ok=True)
            misc.register_work_dir(str(work))
            self.dataset = instantiate(self.cfg.data.train)
        return self.dataset

    def sample(self, kind: str, record: dict) -> dict:
        value = torch.load(record["file"], map_location="cpu", weights_only=True)
        if kind == "native":
            from torch.utils.data import default_collate
            # ``_get`` rather than ``__getitem__``: upstream retries a random
            # index on read errors, which would silently change the window.
            return default_collate([self._dataset()._get(int(value["index"]))])
        return value

    def localization_forward(self, sample: dict, noise_seed: int) -> LocalizationOutput:
        _, hidden, outputs = self.module("localization", sample, noise_seed)
        return LocalizationOutput(hidden=hidden, native_outputs=outputs)

    def native_loss(self, sample: dict, noise_seed: int) -> torch.Tensor:
        return self.module("native", sample, noise_seed)[0]["loss"]

    def save(self, directory, step: int) -> Path:
        """Upstream ``save_wam_checkpoint`` (``policy.pt``) plus resolved config and stats; no readout heads."""
        import shutil
        from omegaconf import OmegaConf
        from wam.training.checkpoint import save_wam_checkpoint
        target = Path(directory) / "checkpoints" / f"checkpoint_step_{step}"
        target.mkdir(parents=True, exist_ok=True)
        path = target / "policy.pt"
        temporary = target / "policy.pt.tmp"
        save_wam_checkpoint(self.model, temporary, step=step)
        temporary.replace(path)
        OmegaConf.save(self.cfg, target / "native_config.yaml", resolve=True)
        shutil.copyfile(self.cfg.data.train.pretrained_norm_stats, target / "dataset_stats.json")
        return path
