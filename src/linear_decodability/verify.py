"""Contract checks for a new adapter, runnable on one GPU (or CPU for small models).

    python -m linear_decodability.verify --model package.module:MyAdapter \\
        --checkpoint-dir <ckpt> --bundle <bundle> --cache /tmp/lds_verify

Checks, in order:
  1. geometry / layers / hidden_dim are consistent;
  2. a native sample gives a finite scalar loss;
  3. the localization forward returns ``[1, tokens, hidden_dim]`` per layer;
  4. the localization forward is deterministic for a fixed noise seed (paired
     instructions must see identical noise);
  5. the localization loss reaches the backbone, and gives exactly zero
     gradient to action-only parameters (the anchor keeps them in the graph);
  6. labels are non-empty for every checked query;
  7. every trainable parameter gets a gradient from at least one path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def check(adapter, bundle, cache: Path, *, queries: int = 2) -> dict:
    from .heads import SubTokenReadout
    from .mixing import split_rank_loss, zero_anchor
    from .train.split_rank import adapter_windows

    report: dict = {}
    geometry = adapter.geometry
    assert len(adapter.layers) >= 1 and adapter.hidden_dim > 0, "layers / hidden_dim missing"
    report["tokens"] = geometry.tokens

    if hasattr(adapter, "bind"):
        adapter.bind(bundle)
    adapter.frozen_to(adapter_device(adapter))
    windows = adapter_windows(adapter, bundle)
    cache.mkdir(parents=True, exist_ok=True)
    window_file = cache / "window.pt"
    torch.save(adapter.encode_window(bundle, windows[0]), window_file)
    query_files = []
    for index, query in enumerate(bundle.queries[:queries]):
        value = adapter.encode_query(query)
        assert float(value["target"].sum()) > 0, f"empty target for {query.id}"
        path = cache / f"query_{index}.pt"
        torch.save(value, path)
        query_files.append(path)
    adapter.frozen_to("cpu")

    native = adapter.native_loss(adapter.sample("native", dict(file=str(window_file))), 42)
    assert native.ndim == 0 and torch.isfinite(native), f"native loss must be a finite scalar, got {native}"
    report["native_loss"] = float(native.detach())
    del native

    sample = adapter.sample("localization", dict(file=str(query_files[0])))
    first = adapter.localization_forward(sample, 7)
    for layer in adapter.layers:
        shape = tuple(first.hidden[layer].shape)
        assert shape == (1, geometry.tokens, adapter.hidden_dim), f"layer {layer}: got {shape}"
    with torch.no_grad():
        again = adapter.localization_forward(sample, 7)
    for a, b in zip(first.native_outputs, again.native_outputs):
        assert torch.equal(a.detach(), b.detach()), "localization forward is not deterministic for a fixed seed"
    report["deterministic"] = True

    device = first.hidden[adapter.layers[0]].device
    heads = SubTokenReadout(adapter.hidden_dim, adapter.layers, geometry.tokens, geometry.sub).to(device)
    loss = heads(first.hidden, sample["target"][None].to(device), sample["competitor"][None].to(device))
    total = split_rank_loss(False, localization_loss=loss, anchor=zero_anchor(first.native_outputs), weight=0.25)
    for _, parameter in adapter.named:
        parameter.grad = None
    total.backward()
    action, backbone = [], 0
    for name, parameter in adapter.named:
        grad = parameter.grad
        if adapter.action_parameter(name):
            action.append(name)
            assert grad is None or torch.count_nonzero(grad) == 0, f"action parameter {name} got localization gradient"
        elif grad is not None and torch.count_nonzero(grad) > 0:
            backbone += 1
    assert backbone > 0, "localization loss did not reach any backbone parameter"
    localization_grad = {name for name, p in adapter.named if p.grad is not None and torch.count_nonzero(p.grad) > 0}
    report.update(localization_loss=float(loss), action_parameters=len(action), backbone_parameters_with_grad=backbone)
    for _, parameter in adapter.named:
        parameter.grad = None
    adapter.native_loss(adapter.sample("native", dict(file=str(window_file))), 42).backward()
    native_grad = {name for name, p in adapter.named if p.grad is not None and torch.count_nonzero(p.grad) > 0}
    unused = sorted({name for name, _ in adapter.named} - native_grad - localization_grad)
    assert not unused, (f"{len(unused)} trainable parameters get no gradient from either path "
                        f"(freeze them in the adapter so weight decay does not change them): {unused[:5]}")
    for _, parameter in adapter.named:
        parameter.grad = None
    _, hidden, outputs = adapter.module("localization", sample, 7)
    assert set(hidden) == set(adapter.layers) and outputs, "module('localization', ...) must return (None, hidden, outputs)"
    assert "loss" in adapter.module("native", adapter.sample("native", dict(file=str(window_file))), 42)[0], \
        "module('native', ...) must return ({'loss': ...}, None, None)"
    return report


def adapter_device(adapter):
    for _, parameter in adapter.named:
        return parameter.device
    return torch.device("cpu")


def main(argv=None) -> None:
    from .data import Bundle
    from .train.split_rank import adapter_class
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--weights", default="")
    p.add_argument("--bundle", required=True)
    p.add_argument("--cache", default="lds_verify")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    adapter = adapter_class(args.model)(args.checkpoint_dir, args.weights, args.device, args.cache)
    print(json.dumps(check(adapter, Bundle(args.bundle), Path(args.cache)), indent=2))
    print("All adapter checks passed.")


if __name__ == "__main__":
    main()
