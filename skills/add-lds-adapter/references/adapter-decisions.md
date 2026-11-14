# Adapter decisions in the built-in adapters

## Contents
- OpenWAM
- LingBot-VA
- InternW0-Delta
- Common pitfalls
- Debugging

## OpenWAM (`adapters/openwam.py`)

- **Layout**: three cameras on one 384x320 L-shaped canvas, made by
  OpenWAM's own `assemble_multiview_layout`. The head camera is the top
  256x320 slot, which is the first 80 tokens (8x10) of the first latent frame.
- **Loss**: `arch.compute_loss(..., lambda_video=1, lambda_action=1)`,
  called directly.
- **Localization forward**: `arch(actions_noise, t=1000, ...)`. The latents are
  pure noise except frame 0, which is the encoded observation
  (`fuse_vae_embedding_in_latents=True`). `num_clean_prefix_frames=0`.
- **Taps**: wrap `video_backbone.post_attn_at_layer`, the block-output hook
  OpenWAM already exposes, and record `result.hidden_states[:, :120]`.
- **Checkpointing**: `stable_step` replaces `_step_checkpointed` with the same
  math. It only changes what the closure keeps alive, which avoids a reference
  cycle that held every layer's activations.
- **Action params**: names start with `action_backbone.`.
- **Windows**: every 16 frames, including tails. OpenWAM masks padded steps
  itself.
- **Save**: `arch.save_checkpoint` into a directory together with
  `config.yaml`, `normalization_stats.npy`, and `tokenizer/`, which is the
  release layout.

## LingBot-VA (`adapters/lingbot_va.py`)

- **Layout**: the cameras are encoded separately. Each latent frame is 12x10
  tokens: two 128x160 wrist views side by side in rows 0-3, and the 256x320
  head view in rows 4-11. So the head camera is tokens 40-119 of a frame. In
  the localization forward, the conditioning frame comes first, so the tapped
  sequence is `[:, 120:240][:, 40:120]`.
- **Loss**: upstream `Trainer._prepare_input_dict` + `Trainer.compute_loss`,
  bound to a `Trainer.__new__` object that holds only config, device, patch
  size, and the two `FlowMatchScheduler`s.
- **Localization forward**: video timesteps `scheduler.timesteps[:1]` (the
  noisiest), action latents zero with an all-false mask, `chunk_size=1`,
  `window_size=4`.
- **Taps**: forward hooks on `transformer.blocks[i]` after `apply_ac`. The
  hook is on the checkpoint wrapper, so it does not fire again during
  recomputation.
- **Prompt dropout**: the native sample swaps in the empty-prompt embedding
  with probability 0.1 (upstream `cfg_prob`). Localization never drops the
  prompt.
- **Actions**: 20-D `[xyz, rot6d, grip] x 2` -> 16-D
  `[xyz, quat(scipy xyzw), grip] x 2` -> upstream `_action_post_process` (33
  rows, 9 frames). The released policy uses this convention, so keep it.
- **Windows**: complete 32-step windows only (`start + 32 <= frames`).
- **Frozen extras**: `condition_embedder_action.text_embedder.*` is never used
  by the upstream forward, so it is frozen. Otherwise AdamW weight decay would
  still change it.
- **Save**: `transformer.state_dict()` (checkpoint wrappers already strip
  their prefix) to `checkpoints/checkpoint_step_N/transformer/` as diffusers
  safetensors plus `config.json`.

## InternW0-Delta (`adapters/internw0_delta.py`)

- **Layout**: one 384x256 canvas, with the head camera (256x256) on top and the
  two wrist views below. That gives 12x8 = 96 tokens per latent frame, and the
  head camera is tokens 0-63 of a frame. The video sequence is
  `[anchor, recent, current, future, future]` followed by FutureDelta queries.
  So the supervised tokens are those of the *current* frame, `192:256`.
- **Labels**: the masks go through 256x320 before area pooling to 32x32, as
  in the paper run. The resulting coverages differ from a 256x256 projection
  by less than 0.012.
- **Native path reuses upstream data handling**: the anchor/recent memory
  frames, padding retry, color/domain/camera jitter, and canonical-80
  normalization all live in the upstream LeRobot dataset. Re-implementing them
  would drift from the original. So `bind()` writes a LeRobot v2.1 view of the
  bundle (joint14 parquet, symlinked videos) and the upstream T5 cache.
  `encode_window` stores only the dataset index, and `sample()` calls
  `dataset._get(index)`. `_get` is used instead of `__getitem__` because the
  latter silently substitutes a random sample on errors.
- **Loss**: upstream `frame_training_loss`, seeded from the noise seed. Its
  loss statistics use a world all-reduce. `bind()` rebinds them to a group of
  the native ranks, because localization ranks never call them and the
  collective would hang.
- **Localization forward**: the deployed first-step prefix built from
  upstream pieces (`pre_dit`, `append_future_delta_queries`, the
  frame-memory attention mask, and the cached-VLM pack). The current latent is
  the anchor, the recent slot is padded, and the current frame is followed by
  two noise latents at t=1000. Actions are noise at t=1000. Noise comes from a
  CPU generator seeded per pair. Taps come from `mot(..., capture_layers=...)`.
- **Query encoding**: the official deploy preprocessing
  (`RobotWinPolicySession` canvas / state normalization / VLM images), the
  VAE, and the frozen RynnBrain context. Queries need `joint_state` (joint14).
- **Sampling**: `schedule()` returns `SharedStreamSchedule`, where windows and
  pairs come from one `default_rng(2026102727)` stream. This is how the paper
  run sampled.
- **Zero anchor**: both paths add `0 * p[0]` for every trainable tensor. This
  keeps ZeRO gradient membership identical even for FutureDelta queries and
  proprio encoders that one path does not touch.
- **Frozen extras**: with VLM conditioning the action expert never calls its
  T5 `text_embedding`, so it is frozen (`verify` check 7 caught this).
- **Save**: upstream `save_wam_checkpoint` -> `policy.pt`, plus the resolved
  config and `dataset_stats.json`.

## Common pitfalls

- Tapping the inference path (KV-cache server) instead of the training
  forward. The token layout or normalization can differ.
- Computing coverage on the original camera image when the model crops or
  pads it. `geometry.image_hw` must describe the model's view.
- Making the localization forward depend on global RNG state. Use the seed
  argument, or `torch.random.fork_rng` plus `manual_seed(seed)` as the
  native path does.
- Detaching the tapped hidden states, or returning a copy taken under
  `torch.no_grad()`. Then the loss never reaches the backbone.
- Registering heads on the policy module. Heads are owned by the trainer and
  must not appear in saved policy checkpoints.

## Debugging

| verify failure | Usual cause |
|---|---|
| `layer N: got (...)` | wrong token slice; hook output is a tuple or dataclass; batch dim missing |
| `action parameter ... got localization gradient` | action tokens attend into the tapped tokens before the tap layer (usually fine, but then exclude those params from `action_parameter`, since they are not action-only); or the hidden states were taken after an action-conditioned layer |
| `localization loss did not reach any backbone parameter` | taps detached; forward under `no_grad`; hooks registered on a module copy |
| `not deterministic` | noise drawn from global RNG; dropout active in the localization path |
| `... get no gradient from either path` | modules unused by the training forward (duplicated embedders, inference-only heads); set `requires_grad_(False)` in the adapter and drop them from `named` |
| `empty target` | mask orientation or resolution mismatch; target outside the head camera |
