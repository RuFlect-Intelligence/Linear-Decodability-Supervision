# Release plan

- M1 core: readout heads, labels, region-balanced loss, read-only taps, loss mixing, optimizers, export, tests.
- M2 OpenWAM: adapter + RoboTwin-IF recipe; reproduce one main-table number from released data/checkpoints.
- M3 LingBot-VA and InternW0-Delta adapters and recipes; checkpoints on Hugging Face.
- M4 RoboTwin-IF data release and standalone evaluation benchmark (construction pipeline stays private).

Not released: FastWAM adapter, real-robot data (checkpoints and hyperparameters only), toy example (TBD).

Before release: check base-model weight licenses for derived checkpoints; rename `Fine` to
Linear-Decodability Supervision consistently in paper, code, and checkpoint names.
