# LingBot-VA on RoboTwin-IF (paper main table)

Starts from the RoboTwin LingBot-VA policy after native flow-matching on the
same 150 episodes, and continues for 2 hours on 8 GPUs with
Linear-Decodability Supervision. Settings match `openwam_robotwin_if.md`
except:

| Setting | Value |
|---|---|
| Native windows | 2658 complete 32-step windows (LingBot does not train on padded tails) |
| Prompt dropout | 10% of native samples use the empty prompt (upstream `cfg_prob`) |
| Supervised tokens | tokens 40-119 of the 12x10 latent frame (head camera under the two wrist views) |
| Frozen | VAE, T5, and the unused `condition_embedder_action.text_embedder` |

Put the LingBot-VA source (revision `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`)
on the path, then run:

```bash
export PYTHONPATH=<lingbot-va>:<lingbot-va>/wan_va:$PYTHONPATH
torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \
    --model lingbot_va \
    --checkpoint-dir <lingbot-va-posttrain-robotwin> \
    --weights <transformer dir to start from> \
    --bundle <robotwin_if_bundle> \
    --output runs/lingbot_va_robotwin_if
```

`--checkpoint-dir` supplies `vae/`, `text_encoder/` and `tokenizer/`.
`--weights` defaults to its `transformer/`. Checkpoints are written to
`checkpoints/checkpoint_step_N/transformer/` (diffusers format, loadable by
LingBot-VA's server) and contain no readout heads.

Reproduction checks against the paper run (9052 updates): the window set, the
per-update window ids, query pairs, and noise seeds match exactly; encoded
action tensors match bit for bit; localization labels and text-cache keys are
identical.
