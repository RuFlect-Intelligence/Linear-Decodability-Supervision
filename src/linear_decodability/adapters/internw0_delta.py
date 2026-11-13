"""InternW0-Delta binding (not yet ported).

Head camera: (256, 256) px, (8, 8) tokens, tokens 192-255 of the current-frame segment.
"""
from __future__ import annotations

from ..labels import Geometry

GEOMETRY = Geometry(image_hw=(256, 256), grid=(8, 8))
LAYERS = (9, 19, 29)
HIDDEN_DIM = 3072


class InternW0DeltaAdapter:
    geometry = GEOMETRY
    layers = LAYERS
    hidden_dim = HIDDEN_DIM

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("InternW0-Delta adapter is scheduled for a later milestone")
