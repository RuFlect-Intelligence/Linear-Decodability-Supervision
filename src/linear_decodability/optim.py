"""Separate optimizers for the backbone and the readout heads."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .schedule import BACKBONE_LR, HEAD_LR, lr_scale

BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01


def build_optimizers(backbone_params: Iterable[nn.Parameter], heads: nn.Module,
                     backbone_lr: float = BACKBONE_LR, head_lr: float = HEAD_LR):
    backbone = torch.optim.AdamW(backbone_params, lr=backbone_lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    head = torch.optim.AdamW(heads.parameters(), lr=head_lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    for optimizer in (backbone, head):
        for group in optimizer.param_groups:
            group["peak_lr"] = group["lr"]
    return backbone, head


def set_lr(step: int, total: int, *optimizers, warmup: int | None = None) -> float:
    scale = lr_scale(step, total) if warmup is None else lr_scale(step, total, warmup)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["peak_lr"] * scale
    return scale


def sync_head_grads(heads: nn.Module) -> None:
    """All-reduce head gradients (heads live outside ZeRO); no-op without a process group."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size()
    for param in heads.parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad)
        param.grad /= world


def clip_and_step(heads: nn.Module, head_optimizer: torch.optim.Optimizer, max_norm: float = 1.0) -> float:
    sync_head_grads(heads)
    norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm)
    head_optimizer.step()
    return float(norm)
