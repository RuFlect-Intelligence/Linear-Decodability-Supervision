---
name: add-lds-adapter
description: Add Linear-Decodability Supervision (LDS) to a new world-action model, VLA, or video-action policy by writing an adapter for the `linear_decodability` package, verifying it, and launching split-rank training. Use when a user wants to train their own policy with LDS, port LDS to a model other than the built-in OpenWAM / LingBot-VA / InternW0-Delta adapters, or debug an adapter that fails `python -m linear_decodability.verify`.
---

# Add an LDS adapter

LDS adds a training-only loss: linear heads must read, from each visual token
of the head camera, which sub-cells contain the instructed referent. The policy
architecture, inputs, and inference stay unchanged, and the heads are dropped
before deployment. All model-specific code lives in one adapter class. The
package provides the loss, labels, heads, schedules, data reader, and trainer.

## Workflow

1. **Read the contract and an example.** Read
   `src/linear_decodability/adapters/base.py` and `examples/toy_adapter.py`.
   Then read whichever built-in adapter is closer to the user's model:
   `adapters/openwam.py` (one fused multi-view canvas, own `compute_loss`),
   `adapters/lingbot_va.py` (per-camera VAE, upstream Trainer methods, diffusers
   transformer), or `adapters/internw0_delta.py` (mixture of transformers with
   memory frames, upstream dataset and loss reused as-is through the optional
   `bind` / `windows` / `schedule` hooks). `references/adapter-decisions.md`
   explains every non-obvious choice in these adapters. Read it before step 3.

2. **Map the user's model.** Before writing code, find these in the user's
   training code and record them, with file and line:
   - the native training loss function and the batch it consumes;
   - the forward call used in training (not the inference server loop);
   - the transformer block list, the hidden size, and the block output type;
   - where head-camera tokens sit in the sequence, and the token grid of the
     head camera (pixels / patch size);
   - which parameters are action-only (action embed / decode / action expert);
   - frozen encoders (VAE, text, image) and how training batches are cached;
   - the native noise / timestep convention, and its maximum timestep;
   - the checkpoint format the model's own inference loader reads.
   If any item is unclear, ask the user. Do not guess token layout or timestep
   direction.

3. **Write the adapter**, in the user's codebase or as
   `src/linear_decodability/adapters/<model>.py`. Rules:
   - Use the upstream model's own functions for losses, preprocessing, and
     action normalization. Bind methods to minimal objects
     (`Trainer.__new__(Trainer)`) instead of re-implementing them, and do not
     patch upstream source.
   - `module` is one `nn.Module` that dispatches on `kind` (`"native"` /
     `"localization"`). DeepSpeed wraps it, so both paths must run through it.
   - Localization forward: the clean current observation as conditioning, the
     instruction text, and pure noise at the maximum diffusion timestep for
     every predicted stream (future video, actions). Never feed ground-truth
     future frames or actions.
   - Capture the taps with forward hooks that return `None` (see
     `linear_decodability.taps.capture`). If blocks use activation
     checkpointing, hook the wrapper so a hook fires once per forward and not
     again on recomputation.
   - Return only the head-camera tokens, `[B, geometry.tokens, hidden_dim]`,
     in row-major order matching `Geometry.grid`.
   - `geometry.image_hw` is the pixel size the model actually sees for the
     head camera after its own resize. Masks are projected into that frame.
   - Return the policy's prediction tensors as `native_outputs`. The trainer
     multiplies them by zero so ZeRO sees every parameter.
   - Default layers: the ends of the three thirds,
     `linear_decodability.default_layers(depth)`.
   - Draw all noise from the `seed` argument. The two instructions of a pair
     get the same seed, and their outputs must match for a fixed seed.
   - `save()` writes the model's native checkpoint format without heads.

4. **Verify.** Run
   `python -m linear_decodability.verify --model <module>:<Class> --checkpoint-dir ... --bundle ...`.
   Fix every failure; do not weaken a check. Then run the package tests
   (`pytest`). See `references/adapter-decisions.md#debugging` for the usual
   causes of each failure.

5. **Train.** With 8 GPUs, start with `--cache-only`, then a short
   `--max-steps 20` run, then the real budget:
   ```bash
   torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \
       --model <module>:<Class> --checkpoint-dir <ckpt> --weights <weights> \
       --bundle <bundle> --output runs/<name>
   ```
   Defaults are the paper settings: localization weight 0.25, lr 1e-5 for the
   backbone and 3e-4 for the heads, 32-update warm-up, cosine decay to 10%,
   and 2 h. The world size must be divisible by 4. In `metrics.jsonl`,
   `localization` should fall well below its initial value within a few
   hundred updates, and `native` should stay close to its pre-LDS level.

## Data

The trainer reads the RoboTwin-IF bundle (`linear_decodability.data.Bundle`).
To use other data, produce the same layout (see the docstring of
`data/robotwin_if.py`). Each localization view needs exactly two instructions
that refer to different objects, plus target and competitor masks in head-camera
pixels. Masks must be in the camera's own orientation, the same one the model
sees.

## Report back

Tell the user which files changed, the verify output, any upstream behavior
you reproduced on purpose (e.g. prompt dropout, window rules), and anything you
could not check (e.g. no GPU).
