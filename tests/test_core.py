import math

import pytest
import torch
from torch import nn

import linear_decodability as lds


def test_region_loss_matches_manual():
    z = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
    t = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    o = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    sp = nn.functional.softplus
    expected = .50 * sp(-z[0, 0]) + .45 * sp(z[0, 1]) + .05 * (sp(z[0, 2]) + sp(z[0, 3])) / 2
    assert torch.allclose(lds.region_balanced_bce(z, t, o), expected)


def test_missing_region_is_renormalized_and_target_wins():
    z = torch.tensor([[1.0, -2.0]])
    t = torch.tensor([[1.0, 0.0]])
    o = torch.tensor([[1.0, 0.0]])
    sp = nn.functional.softplus
    expected = (.50 * sp(-z[0, 0]) + .05 * sp(z[0, 1])) / .55
    assert torch.allclose(lds.region_balanced_bce(z, t, o), expected)


def test_empty_target_rejected():
    with pytest.raises(ValueError):
        lds.region_balanced_bce(torch.zeros(1, 4), torch.zeros(1, 4), torch.ones(1, 4))


def test_coverage_is_weight_not_soft_target():
    t = torch.tensor([[0.1, 0.0]])
    o = torch.zeros(1, 2)
    high = lds.region_balanced_bce(torch.tensor([[8.0, -8.0]]), t, o)
    low = lds.region_balanced_bce(torch.tensor([[-2.0, -8.0]]), t, o)
    assert high < low


def test_token_major_layout():
    geometry = lds.Geometry(image_hw=(64, 80), grid=(2, 3), sub=4)
    fine = torch.arange(8 * 12, dtype=torch.float32).reshape(8, 12)
    out = lds.token_major(fine, geometry.grid, geometry.sub)
    assert out.shape == (6, 16)
    assert torch.equal(out[1], fine[0:4, 4:8].reshape(-1))
    assert torch.equal(out[3], fine[4:8, 0:4].reshape(-1))


def test_query_labels_coverage():
    geometry = lds.Geometry(image_hw=(32, 32), grid=(2, 2), sub=4)
    target = torch.zeros(64, 64)
    target[:16, :16] = 1
    other = torch.zeros(64, 64)
    other[-16:, -16:] = 1
    t, o = lds.query_labels(target, [other], geometry)
    assert t.shape == o.shape == (4, 16)
    assert t[0].sum() > 0 and t[3].sum() == 0 and o[3].sum() > 0
    with pytest.raises(ValueError):
        lds.query_labels(torch.zeros(64, 64), [], geometry)


def test_default_layers():
    assert lds.default_layers(30) == (9, 19, 29)


def test_schedules():
    assert lds.localization_weight(0) == pytest.approx(0.25 / 32)
    assert lds.localization_weight(100) == 0.25
    assert lds.lr_scale(31, 1000) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 31 / 1000)))
    assert lds.lr_scale(1000, 1000) == pytest.approx(0.1)


def test_query_stream_covers_each_query_per_epoch_and_keeps_pairs():
    stream = lds.query_stream(8, 4, seed=3)
    seen = [i for _ in range(2) for i in next(stream)]
    assert sorted(seen) == list(range(8))
    for k in range(0, 8, 2):
        assert seen[k] // 2 == seen[k + 1] // 2 and seen[k] % 2 == 0
    resumed = lds.query_stream(8, 4, seed=3, start_update=1)
    assert next(resumed) == seen[4:]


class TinyPolicy(nn.Module):
    """Shared visual blocks feeding a video head and an action head."""

    def __init__(self, dim=16, depth=3, tokens=4):
        super().__init__()
        self.tokens = tokens
        self.embed = nn.Linear(8, dim)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(depth))
        self.video = nn.Linear(dim, 8)
        self.action = nn.Linear(dim, 2)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h)
        return self.video(h), self.action(h.mean(1))


def test_taps_read_only_and_action_branch_gets_zero_gradient():
    torch.manual_seed(0)
    policy = TinyPolicy()
    x = torch.randn(2, 6, 8)
    reference = policy(x)
    readout = lds.SubTokenReadout(dim=16, layers=(0, 2), tokens=4, sub=2)
    with lds.capture(policy.blocks, readout.layers, select=lambda h: h[:, :4]) as hidden:
        video, action = policy(x)
    assert torch.equal(video, reference[0]) and torch.equal(action, reference[1])
    assert set(hidden) == {0, 2} and hidden[0].shape == (2, 4, 16)
    target = torch.zeros(2, 4, 4)
    target[:, 0] = 1
    competitor = torch.zeros(2, 4, 4)
    competitor[:, 1] = 1
    loss = lds.split_rank_loss(False, localization_loss=readout(hidden, target, competitor),
                               anchor=lds.zero_anchor([video, action]), weight=0.25)
    loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in policy.action.parameters())
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in policy.video.parameters())
    assert torch.count_nonzero(policy.blocks[0][0].weight.grad) > 0


def test_optimizers_and_export():
    policy = TinyPolicy()
    readout = lds.SubTokenReadout(dim=16, layers=(2,), tokens=4)
    backbone, head = lds.build_optimizers(policy.parameters(), readout)
    lds.set_lr(0, 100, backbone, head)
    assert head.param_groups[0]["lr"] / backbone.param_groups[0]["lr"] == pytest.approx(30)
    policy.readout = readout
    from linear_decodability.export import strip_readout
    assert not any(k.startswith("readout.") for k in strip_readout(policy.state_dict()))
