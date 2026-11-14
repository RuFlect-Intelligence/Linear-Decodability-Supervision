# InternW0-Delta on RoboTwin-IF (paper main table)

Starts from InternW0-Delta-RoboTwin after native training on the same 150
episodes, and continues for 2 hours on 8 GPUs with Linear-Decodability
Supervision. Optimizers, schedules, loss weight, heads, and the time budget
match `openwam_robotwin_if.md`, with these differences:

| Setting | Value |
|---|---|
| Native objective | upstream `frame_training_loss` unchanged (video, action, FutureDelta, semantic alignment), reading samples through the upstream LeRobot dataset with its memory frames, padding, augmentation, and normalization |
| Native windows | 45942: every valid-action row of every episode (stride 1, padded tails included) |
| Sampling | windows and instruction pairs drawn from one shared RNG stream (seed 2026102727); same noise seeds as the other recipes |
| Supervised tokens | tokens 192-255: head camera (8x8 of the 12x8 canvas frame) of the current observation, after the anchor and recent memory frames |
| Localization forward | current frame as anchor, padded recent slot, current frame and two pure-noise future latents at t=1000, pure-noise actions at t=1000, frozen RynnBrain context of the current views |
| Trainable | MoT (video + action experts) and both proprio encoders; VAE, T5, and the VLM are frozen, as is the action expert's T5 `text_embedding`, which no forward reads under VLM conditioning. Action-only: `mot.mixtures.action.*`, `action_proprio_encoder.*` |
| Native loss statistics | all-reduced over the four native ranks only |

Put the InternW0-Delta source (revision
`90801baa3bdc7829c1e4989edfe3d0fb2410ea6a`) on the path and install its
dependencies (`transformers==5.13.0`, which the RynnBrain processor needs).
`--checkpoint-dir` is the upstream `checkpoints/` root described in its
`docs/models.md` (`robotwin.pt`, `RynnBrain1.1-2B/`, and the Wan backbone
files). Then run:

```bash
export PYTHONPATH=<internw0>:<internw0>/src:$PYTHONPATH
torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \
    --model internw0_delta \
    --checkpoint-dir <internw0 checkpoints/> \
    --weights <policy .pt to start from> \
    --bundle <robotwin_if_bundle> \
    --output runs/internw0_delta_robotwin_if
```

The first launch writes a LeRobot v2.1 view of the bundle (parquet files, with
videos symlinked) and the upstream T5 cache under `<output>/cache`. It then
encodes each localization query once (VAE latent and frozen VLM context).
Checkpoints are written to `checkpoints/checkpoint_step_N/policy.pt` (upstream
`save_wam_checkpoint` format, loadable by `eval/robotwin`) with
`native_config.yaml` and `dataset_stats.json`. They contain no readout heads.

Reproduction checks against the paper run (4396 updates):
- These match exactly: the LeRobot export (metadata, parquet, videos), the
  window list and dataset indices, every update's window ids, query pairs, and
  noise seeds, the rank-0 sample log, the loss-weight warm-up, the
  localization labels, the head initialization, the text-cache keys, and the
  trainable parameter list. The one difference: the paper run left the unused
  action `text_embedding` (4 tensors) trainable. It received only weight decay
  there, and it has no effect on outputs.
- Upstream dataset samples from the exported view are bit-identical to the
  paper export.
- Given the paper's cached query encodings, the localization forward
  reproduces the paper outputs exactly.
- Query encodings computed on CPU in this check differ slightly from the
  paper's GPU encodings (bf16 VAE/VLM numerics). The difference is about 1.5%
  for the latent and about 2% for the tapped features.
