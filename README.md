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
| OpenWAM adapter + RoboTwin-IF recipe | planned |
| LingBot-VA adapter | planned |
| InternW0-Delta adapter | planned |
| Checkpoints / RoboTwin-IF data | planned (Hugging Face) |

## Install

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0. Upstream models keep their own licenses.
