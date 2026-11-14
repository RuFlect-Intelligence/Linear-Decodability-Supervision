# Linear-Decodability Supervision

Training-only supervision that makes the instructed referent linearly decodable
from a world-action model's visual tokens. Architecture, inputs, action
parameterization, and inference stay unchanged; the readout heads are discarded
before deployment.

> Work in progress. The README will be finalized together with the paper.

## Status

| Component | State |
|---|---|
| Core (heads, labels, region-balanced loss, taps, schedules) | available |
| OpenWAM adapter + RoboTwin-IF recipe | available (GPU run pending) |
| LingBot-VA adapter + RoboTwin-IF recipe | available (GPU run pending) |
| InternW0-Delta adapter + RoboTwin-IF recipe | available (GPU run pending) |
| Checkpoints / RoboTwin-IF data | planned (Hugging Face) |

## Install

```bash
pip install -e ".[dev]"
pytest
```

## Add LDS to your own model

The fastest route is to hand your coding agent the skill in
`skills/add-lds-adapter/`. Copy it into your agent's skills directory
(Codex: `~/.codex/skills/`, Claude Code: `.claude/skills/`) and ask, for example,
"Use $add-lds-adapter to add Linear-Decodability Supervision to my policy".
The skill walks the agent through mapping your model, writing one adapter
class, and passing the contract checks:

```bash
python -m linear_decodability.verify --model my_pkg.adapter:MyAdapter \
    --checkpoint-dir <ckpt> --bundle <bundle>
```

`examples/toy_adapter.py` is a complete minimal adapter that runs on CPU.

## Training OpenWAM on RoboTwin-IF

Install OpenWAM at revision `d8dd33d8576b475f5a5cdc6fb8ca902778a199b3`, then:

```bash
pip install -e ".[openwam]"
torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \
    --checkpoint-dir <openwam_checkpoint> --weights <weights.safetensors> \
    --bundle <robotwin_if_bundle> --output runs/openwam_robotwin_if
```

See `recipes/openwam_robotwin_if.md` for the exact settings,
`recipes/lingbot_va_robotwin_if.md` for LingBot-VA (`--model lingbot_va`), and
`recipes/internw0_delta_robotwin_if.md` for InternW0-Delta (`--model internw0_delta`).

## License

Apache-2.0. Upstream models keep their own licenses.
