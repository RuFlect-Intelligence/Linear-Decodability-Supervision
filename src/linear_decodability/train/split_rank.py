"""Split-rank Linear-Decodability training (DeepSpeed ZeRO-2, one sample per GPU).

Half of the ranks run the policy's native objective on expert windows, the
other half run the localization forward on instruction pairs. Gradients are
averaged over all ranks, giving ``L_native + lambda * L_loc`` per update.

Launch (8 GPUs)::

    torchrun --nproc_per_node 8 -m linear_decodability.train.split_rank \\
        --model openwam --checkpoint-dir <checkpoint dir> --weights <weights> \\
        --bundle <RoboTwin-IF bundle> --output <run dir>
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import random
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..heads import SubTokenReadout
from ..mixing import split_rank_loss, zero_anchor
from ..schedule import BACKBONE_LR, HEAD_LR, LOCALIZATION_WEIGHT, WARMUP_UPDATES, budget_lr_scale, localization_weight
from .batching import SplitRankSchedule

HEAD_SEED = 42
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def zero_config(world: int) -> dict:
    return dict(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=1, train_batch_size=world,
                bf16=dict(enabled=True), gradient_clipping=1.0, zero_allow_untested_optimizer=True,
                zero_optimization=dict(stage=2, contiguous_gradients=True, overlap_comm=True, reduce_scatter=True,
                                       reduce_bucket_size=25_000_000, allgather_bucket_size=25_000_000),
                steps_per_print=1_000_000, wall_clock_breakdown=False)


def parameter_groups(named, limit: int = 150_000_000) -> list[dict]:
    groups, current, size = [], [], 0
    for _, parameter in named:
        if current and size + parameter.numel() > limit:
            groups.append({"params": current})
            current, size = [], 0
        current.append(parameter)
        size += parameter.numel()
    if current:
        groups.append({"params": current})
    return groups


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def loading_slot(output: Path, slots: int = 2):
    """Bound how many ranks build the model on CPU at once."""
    handles = [(output / f".loading_{i}.lock").open("a") for i in range(slots)]
    try:
        while True:
            for handle in handles:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    continue
            else:
                time.sleep(1)
                continue
            break
        yield
    finally:
        for handle in handles:
            handle.close()


def all_max(value: float, device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def sync_heads(heads: torch.nn.Module, world: int) -> None:
    for parameter in heads.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world)
    torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0, error_if_nonfinite=True)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


ADAPTERS = {
    "openwam": "linear_decodability.adapters.openwam:OpenWAMAdapter",
    "lingbot_va": "linear_decodability.adapters.lingbot_va:LingBotVAAdapter",
    "internw0_delta": "linear_decodability.adapters.internw0_delta:InternW0DeltaAdapter",
}


def adapter_class(name: str):
    """Built-in name or ``package.module:Class`` for a user adapter."""
    import importlib
    module, _, attribute = ADAPTERS.get(name, name).partition(":")
    if not attribute:
        raise ValueError(f"Unknown model {name!r}; use one of {sorted(ADAPTERS)} or 'module:Class'")
    return getattr(importlib.import_module(module), attribute)


def adapter_windows(adapter, bundle) -> list[dict]:
    return adapter.windows(bundle) if hasattr(adapter, "windows") else bundle.windows()


def adapter_schedule(adapter, bundle, num_windows: int, world: int, step: int):
    if hasattr(adapter, "schedule"):
        return adapter.schedule(bundle, num_windows, world, step)
    return SplitRankSchedule(num_windows, bundle.pairs(), world, step=step)


def prepare_cache(adapter, bundle, cache: Path, device) -> tuple[list[dict], list[dict]]:
    """Encode windows and queries once with the frozen VAE/T5 (sharded over ranks)."""
    rank, world = dist.get_rank(), dist.get_world_size()
    windows = adapter_windows(adapter, bundle)
    adapter.frozen_to(device)
    for kind, rows, encode in (("windows", windows, lambda row: adapter.encode_window(bundle, row)),
                               ("queries", bundle.queries, adapter.encode_query)):
        for index, row in enumerate(rows):
            if index % world != rank:
                continue
            key = row["id"] if isinstance(row, dict) else row.id
            path = cache / kind / f"{key}.pt"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                value = encode(row)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save(value, temporary)
                temporary.replace(path)
    adapter.frozen_to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    window_records = [dict(id=row["id"], file=str(cache / "windows" / f"{row['id']}.pt")) for row in windows]
    query_records = [dict(id=query.id, file=str(cache / "queries" / f"{query.id}.pt")) for query in bundle.queries]
    missing = [r["file"] for r in window_records + query_records if not Path(r["file"]).exists()]
    if missing:
        raise RuntimeError(f"Encoder cache incomplete: {missing[:3]}")
    return window_records, query_records


def latest_tag(output: Path) -> str | None:
    path = output / "latest.json"
    return json.loads(path.read_text())["tag"] if path.exists() else None


def train(args) -> None:
    import deepspeed
    from ..data import Bundle

    deepspeed.init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache or output / "cache")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    bundle = Bundle(args.bundle, verify=rank == 0)
    dist.barrier()
    with loading_slot(output):
        adapter = adapter_class(args.model)(args.checkpoint_dir, args.weights, device, cache)
    if hasattr(adapter, "bind"):
        adapter.bind(bundle)
    window_records, query_records = prepare_cache(adapter, bundle, cache, device)
    if args.cache_only:
        return

    seed_everything()
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(HEAD_SEED)
        heads = SubTokenReadout(adapter.hidden_dim, adapter.layers, adapter.geometry.tokens).to(device)
    named = adapter.named
    optimizer = torch.optim.AdamW(parameter_groups(named), lr=args.lr, betas=BETAS, weight_decay=WEIGHT_DECAY,
                                  foreach=False)
    engine, _, _, _ = deepspeed.initialize(model=adapter.module, optimizer=optimizer,
                                           model_parameters=[p for _, p in named], config=zero_config(world))
    head_optimizer = torch.optim.AdamW(heads.parameters(), lr=args.head_lr, betas=BETAS, weight_decay=WEIGHT_DECAY)

    step, elapsed_before = 0, 0.0
    tag = latest_tag(output)
    if tag is not None:
        _, state = engine.load_checkpoint(str(output / "states"), tag=tag, load_module_strict=False)
        heads.load_state_dict(state["heads"])
        head_optimizer.load_state_dict(state["head_optimizer"])
        step, elapsed_before = int(state["step"]), float(state["elapsed"])
    schedule = adapter_schedule(adapter, bundle, len(window_records), world, step)
    engine.train()
    seed_everything()
    dist.barrier()
    started = time.monotonic() - elapsed_before
    last_save, saved = elapsed_before, step

    def save(elapsed: float) -> None:
        tag = f"step_{step:08d}"
        engine.save_checkpoint(str(output / "states"), tag=tag, exclude_frozen_parameters=True,
                               client_state=dict(step=step, elapsed=elapsed, heads=heads.state_dict(),
                                                 head_optimizer=head_optimizer.state_dict()))
        if rank == 0:
            policy = adapter.save(output, step)
            torch.save(heads.state_dict(), output / f"readout_heads_step_{step}.pt")
            write_json(output / "latest.json", dict(step=step, elapsed=elapsed, tag=tag, checkpoint=str(policy)))
        dist.barrier()

    while True:
        elapsed = all_max(time.monotonic() - started, device)
        if STOP or elapsed >= args.train_seconds or (args.max_steps and step >= args.max_steps):
            break
        windows, queries, noise = schedule.next()
        kind, index = schedule.item(rank, windows, queries)
        weight = localization_weight(step, args.localization_weight, WARMUP_UPDATES)
        head_optimizer.zero_grad(set_to_none=True)
        if kind == "native":
            sample = adapter.sample(kind, window_records[index])
            native = engine("native", sample, noise[rank])[0]
            loss = split_rank_loss(True, native_loss=native["loss"])
            auxiliary = None
        else:
            sample = adapter.sample(kind, query_records[index])
            _, hidden, outputs = engine("localization", sample, noise[rank])
            auxiliary = heads(hidden, sample["target"][None].to(device), sample["competitor"][None].to(device))
            loss = split_rank_loss(False, localization_loss=auxiliary, anchor=zero_anchor(outputs), weight=weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        engine.backward(loss)
        sync_heads(heads, world)
        scale = budget_lr_scale(step, elapsed, args.train_seconds)
        for group in engine.optimizer.param_groups:
            group["lr"] = args.lr * scale
        for group in head_optimizer.param_groups:
            group["lr"] = args.head_lr * scale
        engine.step()
        head_optimizer.step()
        values = torch.tensor([2 * float(native[k].detach()) if kind == "native" and k in native else 0.0
                               for k in ("loss", "loss_video", "loss_action")]
                              + [2 * float(auxiliary.detach()) if auxiliary is not None else 0.0],
                              device=device, dtype=torch.float64)
        dist.all_reduce(values)
        values /= world
        step += 1
        elapsed = all_max(time.monotonic() - started, device)
        if rank == 0:
            record = dict(step=step, elapsed=elapsed, native=float(values[0]), native_video=float(values[1]),
                          native_action=float(values[2]), localization=float(values[3]),
                          localization_weight=weight, lr=args.lr * scale, noise_seeds=noise,
                          windows=[window_records[i]["id"] for i in windows],
                          queries=[query_records[i]["id"] for i in queries])
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if step % args.log_every == 0:
                print(json.dumps({k: v for k, v in record.items() if not isinstance(v, list)}), flush=True)
        del sample, loss
        if elapsed - last_save >= args.save_seconds:
            save(elapsed)
            saved, last_save = step, elapsed
    if step > saved:
        save(all_max(time.monotonic() - started, device))
    if rank == 0:
        write_json(output / ("interrupted.json" if STOP else "complete.json"), dict(step=step))
    dist.barrier()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openwam", help=f"{sorted(ADAPTERS)} or 'package.module:AdapterClass'")
    p.add_argument("--checkpoint-dir", required=True, help="Released checkpoint directory of the policy")
    p.add_argument("--weights", default="", help="Weights to start from (OpenWAM: .safetensors file; "
                   "LingBot-VA: transformer directory; InternW0-Delta: policy .pt; default: the release weights)")
    p.add_argument("--bundle", required=True, help="RoboTwin-IF release bundle")
    p.add_argument("--output", required=True)
    p.add_argument("--cache", help="Encoder cache directory (default: <output>/cache)")
    p.add_argument("--train-seconds", type=float, default=7200.0)
    p.add_argument("--save-seconds", type=float, default=1800.0)
    p.add_argument("--max-steps", type=int, default=0, help="Optional step cap (0 = time budget only)")
    p.add_argument("--lr", type=float, default=BACKBONE_LR)
    p.add_argument("--head-lr", type=float, default=HEAD_LR)
    p.add_argument("--localization-weight", type=float, default=LOCALIZATION_WEIGHT)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--cache-only", action="store_true", help="Only build the encoder cache")
    return p


def main(argv=None) -> None:
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
