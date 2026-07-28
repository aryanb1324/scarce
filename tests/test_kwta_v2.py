"""
Unit tests for the kWTA v2 module (configurable axis + duty-cycle boosting).

Per architecture/skills.md, a new mechanism is validated standalone before it is
wired into a training run. These also cover two gaps the v1 test file left open,
recorded in architecture/skills.md:

  * v1 tested only on `torch.randn`, which has no zeros. In the real network kWTA
    sits after ReLU, where many activations are exactly 0 and `topk` happily
    "keeps" them -- so the true sparsity is data-dependent and lower than k.
  * v1's gradient test asserted only that *some* gradient was nonzero. The actual
    contract is stronger: exactly 0 for masked units, exactly 1 for survivors.

Run with: python -m pytest tests/test_kwta_v2.py -v
"""

import pytest
import torch
import torch.nn as nn

from architecture.modules.kwta import KWinnersTakeAll
from architecture.modules.kwta_v2 import KWinnersTakeAllV2


# ---------------------------------------------------------------- compatibility

def test_global_no_boost_is_bit_identical_to_v1():
    """
    The anchor-arm guarantee. If this ever fails, the Stage 1 factorial loses its
    comparison point and the v2 results stop being usable as a baseline.
    """
    for shape in [(8, 16), (4, 16, 7, 7), (2, 32, 5, 5)]:
        x = torch.randn(*shape)
        old = KWinnersTakeAll(k=0.2)(x)
        new = KWinnersTakeAllV2(k=0.2, dim="global", boost=0.0)(x)
        assert torch.equal(old, new), f"diverged on shape {shape}"


def test_global_no_boost_matches_v1_on_post_relu_input():
    """Same guarantee on the distribution the module actually sees in the net."""
    x = torch.relu(torch.randn(8, 16, 7, 7))
    old = KWinnersTakeAll(k=0.2)(x)
    new = KWinnersTakeAllV2(k=0.2, dim="global", boost=0.0)(x)
    assert torch.equal(old, new)


def test_channel_and_global_agree_on_2d_input():
    """For a linear layer there is only one axis, so the two rules must coincide."""
    x = torch.randn(6, 128)
    a = KWinnersTakeAllV2(k=0.2, dim="channel")(x)
    b = KWinnersTakeAllV2(k=0.2, dim="global")(x)
    assert torch.equal(a, b)


def test_rejects_unknown_axis():
    with pytest.raises(ValueError):
        KWinnersTakeAllV2(k=0.2, dim="spatial")


# ------------------------------------------------------------- competition axis

def test_channel_mode_keeps_exactly_k_channels_at_every_position():
    """
    The structural claim that motivates the axis change: under channel-wise
    competition exactly k channels win at EVERY spatial position, so no channel
    can lose everywhere by construction.
    """
    x = torch.randn(4, 16, 7, 7)
    out = KWinnersTakeAllV2(k=0.25, dim="channel")(x)   # 25% of 16 = 4
    per_position = (out != 0).sum(dim=1)                # [B, H, W]
    assert torch.all(per_position == 4), per_position.unique()


def test_channel_mode_leaves_no_globally_dead_channel_on_random_input():
    x = torch.randn(32, 16, 7, 7)
    out = KWinnersTakeAllV2(k=0.25, dim="channel")(x)
    wins_per_channel = (out != 0).sum(dim=(0, 2, 3))
    assert torch.all(wins_per_channel > 0)


def test_global_mode_can_starve_a_channel_which_is_the_point():
    """
    Documents the failure mode the axis change is meant to remove: with a global
    top-k, a uniformly weak channel wins nowhere and receives no gradient.
    """
    x = torch.randn(8, 16, 7, 7).abs()
    x[:, 3, :, :] *= 0.001                              # channel 3 is weak everywhere
    out = KWinnersTakeAllV2(k=0.2, dim="global")(x)
    assert (out[:, 3] != 0).sum() == 0


# -------------------------------------------------------------------- boosting

def test_boost_is_disabled_in_eval_mode():
    """Inference must not depend on training-time duty statistics."""
    x = torch.relu(torch.randn(4, 16, 7, 7))
    m = KWinnersTakeAllV2(k=0.2, dim="channel", boost=1.0)
    m.train()
    for _ in range(20):
        m(x)                                            # accumulate a duty cycle
    m.eval()
    plain = KWinnersTakeAllV2(k=0.2, dim="channel", boost=0.0).eval()
    assert torch.equal(m(x), plain(x))


def test_boost_revives_a_starved_channel():
    """The mechanism's whole purpose: a chronic loser must re-enter competition."""
    x = torch.randn(8, 16, 7, 7).abs()
    x[:, 3, :, :] *= 0.2                                # channel 3 rarely wins

    cold = KWinnersTakeAllV2(k=0.25, dim="channel", boost=0.0).train()
    hot = KWinnersTakeAllV2(k=0.25, dim="channel", boost=3.0, duty_alpha=0.2).train()
    for _ in range(60):
        cold(x)
        hot(x)

    cold_wins = (cold(x)[:, 3] != 0).float().mean().item()
    hot_wins = (hot(x)[:, 3] != 0).float().mean().item()
    assert hot_wins > cold_wins, (hot_wins, cold_wins)


def test_boost_zero_never_creates_a_duty_buffer():
    x = torch.randn(4, 16, 7, 7)
    m = KWinnersTakeAllV2(k=0.2, dim="channel", boost=0.0).train()
    m(x)
    assert m.duty.numel() == 0


def test_duty_buffer_survives_state_dict_roundtrip():
    """
    Boosting adds state to the module. Checkpoint selection deep-copies and
    reloads state_dict mid-run, so a buffer that does not round-trip would
    silently reset the duty cycle at the end of every training run.
    """
    x = torch.relu(torch.randn(4, 16, 7, 7))
    a = KWinnersTakeAllV2(k=0.2, dim="channel", boost=1.0).train()
    for _ in range(10):
        a(x)
    assert a.duty.numel() == 16

    b = KWinnersTakeAllV2(k=0.2, dim="channel", boost=1.0).train()
    b(x)                                                # size the buffer first
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a.duty, b.duty)


def test_boost_does_not_scale_the_values_that_pass_through():
    """Boosting must change WHICH units survive, never their magnitudes."""
    x = torch.relu(torch.randn(4, 16, 7, 7))
    m = KWinnersTakeAllV2(k=0.25, dim="channel", boost=2.0, duty_alpha=0.5).train()
    for _ in range(30):
        m(x)
    out = m(x)
    surviving = out != 0
    assert torch.allclose(out[surviving], x[surviving])


# ------------------------------------------------------- gradient contract

@pytest.mark.parametrize("dim", ["channel", "global"])
def test_gradient_is_exactly_one_for_winners_and_zero_for_losers(dim):
    """
    The contract v1's test never checked. kWTA is a hard mask, so d(out)/d(x) is
    1 on survivors and 0 on everything else -- and a masked unit receiving any
    gradient at all would mean the sparsity is not actually sparse in the
    backward pass.
    """
    x = torch.randn(4, 16, 5, 5, requires_grad=True)
    out = KWinnersTakeAllV2(k=0.25, dim=dim)(x)
    out.sum().backward()
    survived = (out != 0)
    assert torch.all(x.grad[survived] == 1.0)
    assert torch.all(x.grad[~survived] == 0.0)


# ------------------------------------------------------- post-ReLU semantics

def test_post_relu_sparsity_is_capped_by_the_positive_count():
    """
    Documents real behaviour rather than assuming it: after ReLU there may be
    fewer than k positive units, and `topk` still returns k entries -- so it
    selects zeros and the effective sparsity is data-dependent, not k.

    Protocol v2 saw this in the wild: the FC layer selects k=26 units but only
    ~21.6 were nonzero at the largest budget.
    """
    x = torch.zeros(1, 20)
    x[0, :3] = torch.tensor([5.0, 3.0, 1.0])            # only 3 positives
    out = KWinnersTakeAllV2(k=10, dim="global")(x)
    assert (out != 0).sum().item() == 3
    assert torch.allclose(out[0, :3], x[0, :3])


# ------------------------------------------------------------- integration

def test_drops_into_the_frozen_baseline_and_trains():
    from architecture.model import build_kwta_v2_model

    for dim, boost in [("channel", 0.0), ("channel", 1.0), ("global", 1.0)]:
        model = build_kwta_v2_model(dim=dim, boost=boost)
        x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
        model.train()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"no gradients flowed for dim={dim} boost={boost}"
        assert any((g != 0).any() for g in grads)
        assert not any(torch.isnan(g).any() for g in grads)


def test_v2_builder_with_defaults_matches_v1_builder():
    """dim='global', boost=0 must reproduce the v1/v2 experimental model exactly."""
    from architecture.model import build_kwta_model, build_kwta_v2_model

    torch.manual_seed(0)
    a = build_kwta_model()
    torch.manual_seed(0)
    b = build_kwta_v2_model(dim="global", boost=0.0)
    a.eval()
    b.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
