# Beyond Linear Decodability

### Understanding and Improving Instruction Following in World Action Models

**Linear-Decodability Supervision (LDS)**

[![Project page](https://img.shields.io/badge/Project-Page-2563EB?style=for-the-badge)](https://ruflect-intelligence.github.io/LDS-Homepage/)
[![arXiv 2611.27401](https://img.shields.io/badge/arXiv-2611.27401-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.27401)
[![GitHub LDS code](https://img.shields.io/badge/GitHub-LDS_Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RuFlect-Intelligence/Linear-Decodability-Supervision)
[![Hugging Face checkpoints](https://img.shields.io/badge/HF-Checkpoints-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](#model-checkpoints)

[![GitHub RoboTwin-IF code](https://img.shields.io/badge/GitHub-RoboTwin--IF_Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RuFlect-Intelligence/RoboTwin-IF)
[![Hugging Face RoboTwin-IF dataset](https://img.shields.io/badge/HF-RoboTwin--IF_Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/RuFlect/RoboTwin-IF)
[![ModelScope mirrors](https://img.shields.io/badge/ModelScope-Mirrors-624AFF?style=for-the-badge)](https://www.modelscope.ai/organization/Ruflect-Intelligence)

![Overview of LDS, prior–instruction competition, and amplification of the instruction signal.](https://ruflect-intelligence.github.io/LDS-Homepage/static/img/paper/main.png)

## Overview

World action models (WAMs) can encode the instructed target yet still act on a
learned visual prior. We explain this failure through **prior–instruction
competition**: an instruction-dependent signal competes with a stronger,
prior-associated signal in the model's visual representations.

**LDS improves instruction following by making the instructed referent linearly
decodable from visual tokens during training.** The auxiliary linear heads are
discarded at inference. The policy keeps its architecture, observations, action
representation, and inference procedure.

The [project page](https://ruflect-intelligence.github.io/LDS-Homepage/) presents
the representation analysis, training-free interventions, simulation results,
held-out evaluations, and real-world experiments. This repository provides the
reusable LDS training components and model adapters.

## Method

![Linear heads supervise fine-grained grounding at selected visual layers and are removed at inference.](https://ruflect-intelligence.github.io/LDS-Homepage/static/img/paper/LDS.png)

LDS combines the policy's native training objective with an auxiliary grounding
loss:

```text
L_total = L_native + λ · L_grounding
```

- Read current-observation head-camera tokens at selected layers, typically
  blocks **9, 19, and 29** (zero-indexed).
- Apply a position-shared linear head to each token, predicting **4 × 4
  sub-cells** for fine-grained localization. Token geometry follows each model.
- Balance the target, competing objects, and background with region weights
  **0.50 / 0.45 / 0.05**, using **λ = 0.25** with warm-up.
- Keep native trajectory training alongside localization supervision.
  Counterfactual localization queries provide grounding labels; they do not
  supply action or future-video targets.

The shared readout encourages the existing instruction signal to become strong
enough to distinguish the referent across paired instructions. See
[why LDS works](https://ruflect-intelligence.github.io/LDS-Homepage/#why-lds)
for the mechanism and measurements.

## Results

Success rates (%) from the project page. Each cell shows **Base → + LDS**.

| Model | LIBERO-CF | LangGap | RoboTwin-IF |
| --- | ---: | ---: | ---: |
| OpenWAM | 5.1 → **65.1** | 19.9 → **60.7** | 7.6 → **54.4** |
| LingBot-VA | 12.3 → **29.8** | 16.3 → **21.5** | 9.6 → **24.0** |
| InternW0-Δ | 23.5 → **74.6** | 37.5 → **71.4** | 9.0 → **47.4** |

LIBERO-CF reports the average faithful success rate; LangGap averages Spatial,
Object, and Goal success; RoboTwin-IF averages Pick and Placement success.
OpenWAM's faithful Grounding on LIBERO-CF also improves from **14.4% to 90.3%**.
Full success, grounding, biased-goal, and retention results are on the
[project page](https://ruflect-intelligence.github.io/LDS-Homepage/#experiments).

These are the paper's evaluated policies. End-to-end GPU validation of the
standalone training package in this repository remains pending.

## Model checkpoints

The nine benchmark-specific **LDS (Fine)** policies are available on Hugging Face:

| Model | LIBERO-CF | LangGap | RoboTwin-IF |
| --- | --- | --- | --- |
| OpenWAM | [Checkpoint](https://huggingface.co/RuFlect/LIBERO-CF-OpenWAM-LDS) | [Checkpoint](https://huggingface.co/RuFlect/LangGap-OpenWAM-LDS) | [Checkpoint](https://huggingface.co/RuFlect/RoboTwin-IF-OpenWAM-LDS) |
| LingBot-VA | [Checkpoint](https://huggingface.co/RuFlect/LIBERO-CF-LingBot-VA-LDS) | [Checkpoint](https://huggingface.co/RuFlect/LangGap-LingBot-VA-LDS) | [Checkpoint](https://huggingface.co/RuFlect/RoboTwin-IF-LingBot-VA-LDS) |
| InternW0-Δ | [Checkpoint](https://huggingface.co/RuFlect/LIBERO-CF-InternW0-Delta-LDS) | [Checkpoint](https://huggingface.co/RuFlect/LangGap-InternW0-Delta-LDS) | [Checkpoint](https://huggingface.co/RuFlect/RoboTwin-IF-InternW0-Delta-LDS) |

Each model card specifies the evaluated step, base model, normalization, required
frozen assets, and loading instructions. Use the policy for the intended
benchmark; the releases contain inference weights without auxiliary LDS heads.
The LIBERO-CF and LangGap releases are the main-evaluation, all-train policies;
held-out-scene results use separate training splits.

ModelScope mirrors are published individually after upload and verification.
Available releases appear under
[Ruflect-Intelligence](https://www.modelscope.ai/organization/Ruflect-Intelligence),
using the same repository names as the table above.

## RoboTwin-IF data and evaluation

- **Expert trajectories:** the unified
  [Hugging Face dataset](https://huggingface.co/datasets/RuFlect/RoboTwin-IF)
  contains **275,270 episodes**, with LeRobot metadata and source provenance.
  A [ModelScope mirror](https://www.modelscope.ai/datasets/Ruflect-Intelligence/RoboTwin-IF)
  is also available.
- **Mixed 50 evaluation:**
  [RoboTwin-IF](https://github.com/RuFlect-Intelligence/RoboTwin-IF) provides
  **25 Pick + 25 Placement scenes**, paired instructions, Grounding and strict
  task-success metrics, unified `validate`, `run`, and `report` entry points,
  and a custom-policy example. Evaluation assets are included in the dataset
  repository's `benchmark/mixed50/` directory.

## Installation

Requires **Python ≥ 3.10** and **PyTorch ≥ 2.1**.

```bash
git clone https://github.com/RuFlect-Intelligence/Linear-Decodability-Supervision.git
cd Linear-Decodability-Supervision
pip install -e .
```

Install the dependencies for your model and its upstream implementation as
described in the corresponding recipe:

| Model | Package extra | RoboTwin-IF recipe |
| --- | --- | --- |
| OpenWAM | `pip install -e ".[openwam]"` | [Recipe](recipes/openwam_robotwin_if.md) |
| LingBot-VA | `pip install -e ".[lingbot]"` | [Recipe](recipes/lingbot_va_robotwin_if.md) |
| InternW0-Δ | `pip install -e ".[internw0]"` | [Recipe](recipes/internw0_delta_robotwin_if.md) |

For development checks:

```bash
pip install -e ".[dev,data]"
pytest
```

## Training on RoboTwin-IF

The split-rank trainer uses eight GPUs: four ranks optimize the native objective
and four optimize localization on two instruction pairs per update.

Training requires a prepared episode-and-localization bundle with schema
`robotwin_if_release_v1`: `manifest.json`, `episodes.jsonl`, `queries.jsonl`,
episode arrays/videos, and localization-query arrays/masks. The expected layout
is documented in the [bundle reader](src/linear_decodability/data/robotwin_if.py).
The public LeRobot expert dataset provides trajectories; the trainer additionally
needs the prepared localization bundle for `--bundle`.

For OpenWAM, install upstream revision
`d8dd33d8576b475f5a5cdc6fb8ca902778a199b3` and the `openwam` extra, then run:

```bash
torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \
    --model openwam \
    --checkpoint-dir <openwam_checkpoint> \
    --weights <weights.safetensors> \
    --bundle <prepared_robotwin_if_bundle> \
    --output runs/openwam_robotwin_if
```

Use `--cache-only` to prepare frozen encodings first. For the other adapters,
use `--model lingbot_va` or `--model internw0_delta` with the checkpoint and
runtime settings in their recipes. Exported policy weights and auxiliary heads
are stored separately.

## Add LDS to your own model

[examples/toy_adapter.py](examples/toy_adapter.py) is a minimal CPU adapter.
Implement the same interface for your policy, then run the contract checks:

```bash
python -m linear_decodability.verify \
    --model my_pkg.adapter:MyAdapter \
    --checkpoint-dir <checkpoint> \
    --bundle <prepared_bundle>
```

The checker verifies feature geometry, deterministic localization forwards,
non-empty labels, backbone gradients, and zero localization gradients for
action-only parameters.

For coding-agent-assisted integration, use
[skills/add-lds-adapter](skills/add-lds-adapter/SKILL.md). Copy the directory into
your agent's skills directory (for example, `~/.codex/skills/` for Codex or
`.claude/skills/` for Claude Code), then ask it to use `$add-lds-adapter` to add
LDS to your policy.

## License

This code is released under [Apache-2.0](LICENSE). Upstream models retain their
own licenses. The released expert trajectories use **CC BY 4.0**; benchmark
assets retain their applicable upstream licenses, as documented in the dataset.
