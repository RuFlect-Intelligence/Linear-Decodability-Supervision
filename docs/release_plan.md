# Release plan

- M1 core: readout heads, labels, region-balanced loss, read-only taps, loss mixing, optimizers, export, tests.
- M2 OpenWAM: adapter, split-rank trainer, RoboTwin-IF bundle reader and recipe (done). Still to do: an 8-GPU run to reproduce the main-table number, and publishing the bundle exporter output.
- M3 LingBot-VA and InternW0-Delta adapters and RoboTwin-IF recipes (done; GPU runs pending), `add-lds-adapter` agent skill with `linear_decodability.verify` and a toy adapter (done). Remaining: checkpoints on Hugging Face.
- M4 RoboTwin-IF data release and standalone evaluation benchmark (construction pipeline stays private).

Not released: FastWAM adapter, real-robot data (checkpoints and hyperparameters only), toy example (TBD).

Before release: check base-model weight licenses for derived checkpoints; rename `Fine` to
Linear-Decodability Supervision consistently in paper, code, and checkpoint names.
