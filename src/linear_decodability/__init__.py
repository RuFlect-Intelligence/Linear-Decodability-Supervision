"""Linear-Decodability Supervision for world-action models."""
from .heads import SubTokenReadout, default_layers
from .labels import Geometry, coverage, query_labels, token_major
from .loss import REGION_BUDGETS, region_balanced_bce
from .mixing import colocated_loss, split_rank_loss, zero_anchor
from .optim import build_optimizers, clip_and_step, set_lr
from .sampler import query_stream
from .schedule import localization_weight, lr_scale
from .taps import capture

__version__ = "0.1.0.dev0"

__all__ = [
    "Geometry", "REGION_BUDGETS", "SubTokenReadout", "build_optimizers", "capture", "clip_and_step",
    "colocated_loss", "coverage", "default_layers", "localization_weight", "lr_scale", "query_labels",
    "query_stream", "region_balanced_bce", "set_lr", "split_rank_loss", "token_major", "zero_anchor",
]
