"""LingBot-VA binding (not yet ported).

Head camera: (256, 320) px, (8, 10) tokens, tokens 40-119 of each 120-token latent frame.
"""
from __future__ import annotations

from ..labels import Geometry

GEOMETRY = Geometry(image_hw=(256, 320), grid=(8, 10))
LAYERS = (9, 19, 29)
HIDDEN_DIM = 3072


class LingBotVAAdapter:
    geometry = GEOMETRY
    layers = LAYERS
    hidden_dim = HIDDEN_DIM

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LingBot-VA adapter is scheduled for a later milestone")
