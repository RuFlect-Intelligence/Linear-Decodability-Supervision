# OpenWAM on RoboTwin-IF (paper main table)

Starts from the released OpenWAM RoboTwin policy (native flow-matching only) and
continues training for 2 hours on 8 GPUs with Linear-Decodability Supervision.

| Setting | Value |
|---|---|
| GPUs / batch | 8 (4 native windows + 2 instruction pairs per update) |
| Budget | 7200 s wall clock, checkpoint every 1800 s |
| Backbone optimizer | AdamW, lr 1e-5, betas (0.9, 0.95), wd 0.01, ZeRO-2, bf16, clip 1.0 |
| Head optimizer | AdamW, lr 3e-4, same betas/wd, clip 1.0 |
| Schedule | 32-update linear warm-up, cosine decay over wall clock to 10% |
| Localization weight | 0.25, warmed up over the first 32 updates |
| Supervised layers | video DiT blocks 9 / 19 / 29, 80 head-camera tokens (8x10), 4x4 sub-cells |
| Data | 150 expert episodes (2945 windows), 50 instruction pairs (100 queries) |

```bash
torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \
    --checkpoint-dir <openwam_robotwin_checkpoint> \
    --weights <checkpoint_step_N.safetensors> \
    --bundle <robotwin_if_bundle> \
    --output runs/openwam_robotwin_if
```

Use `--cache-only` first to encode the VAE/T5 cache, which is reused by later runs.
The output directory holds `checkpoint_step_*.safetensors` plus `config.yaml`,
`normalization_stats.npy`, and `tokenizer/`, which is the same layout as an
OpenWAM release checkpoint. Readout heads are saved separately
(`readout_heads_step_*.pt`) and are not part of the policy.

Reproduction checks against the paper run: the per-update window ids, query
pairs, noise seeds, head initialization, and loss-weight warm-up all match
exactly for all 6678 updates. Bundle actions, states, padding masks,
localization labels, and decoded video frames are bit-identical to the paper
run's encoder inputs. The learning rate matches to about 1e-6, except right
after checkpoint saves, where it depends on wall-clock time.
