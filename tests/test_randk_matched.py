"""
Unit tests for the sparsity-MATCHED random-k control (Stage 2b).

Every test that samples is seeded, and every count test feeds an input with a
REALISTIC POST-RELU ZERO FRACTION. That is the whole point of this file:
`tests/test_randk.py::test_sparsity_count_matches_kwta_exactly` passes vacuously
because it feeds `relu(randn) + 1e-3`, which is strictly positive, so no zeros
exist and the two counts agree trivially. On the input the module actually sees
(~50-70% exact zeros) the old control keeps 3.4x fewer live units than kWTA --
the confound this module exists to remove.

The contract under test, which both `KWinnersTakeAllV2(dim="channel")` and
`RandomKMatchedChannel` must satisfy at EVERY competition site:

    number of surviving nonzero entries == min(k, number of nonzero inputs)

kWTA satisfies it because after ReLU every nonzero outranks every zero, so top-k
spends its budget on live units first. The matched control satisfies it by
restricting the random draw to the live set.
"""

import pytest
import torch
import torch.nn as nn

from architecture.modules.kwta_v2 import KWinnersTakeAllV2
from architecture.modules.randk import RandomKChannel
from architecture.modules.randk_matched import RandomKMatchedChannel


def post_relu(shape, zero_frac=0.0, seed=0):
    """
    An input with the distribution the module actually sees: nonnegative, with a
    controllable fraction of EXACT zeros. `zero_frac` is extra zeroing on top of
    ReLU's own ~50%, so the realized zero fraction is 0.5 + 0.5 * zero_frac.
    """
    torch.manual_seed(seed)
    x = torch.relu(torch.randn(*shape))
    if zero_frac > 0.0:
        x = x * (torch.rand_like(x) >= zero_frac).to(x.dtype)
    return x


def live_per_site(t):
    """Nonzero entries along the channel/feature dim, for 4-D or 2-D tensors."""
    return (t != 0).sum(dim=1)


# --------------------------------------------------------------------------
# THE HEADLINE CONTRACT: live-unit count matches kWTA on post-ReLU input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("zero_frac", [0.0, 0.4, 0.6])
def test_live_count_matches_kwta_exactly_on_post_relu_input(zero_frac):
    x = post_relu((8, 16, 7, 7), zero_frac=zero_frac, seed=1)
    realized = (x == 0).float().mean().item()
    assert 0.45 < realized < 0.85, f"test input is not realistic: {realized:.2f} zeros"

    torch.manual_seed(2)
    matched = RandomKMatchedChannel(k=0.25)(x)
    kwta = KWinnersTakeAllV2(k=0.25, dim="channel")(x)

    # Site by site, not just in aggregate.
    assert torch.equal(live_per_site(matched), live_per_site(kwta))
    assert (matched != 0).sum().item() == (kwta != 0).sum().item()


@pytest.mark.parametrize("zero_frac", [0.0, 0.4, 0.6])
def test_live_count_equals_min_k_and_available(zero_frac):
    """The explicit formula, so a future refactor cannot drift off it silently."""
    x = post_relu((8, 16, 7, 7), zero_frac=zero_frac, seed=3)
    k = 4  # 0.25 * 16
    torch.manual_seed(4)
    out = RandomKMatchedChannel(k=0.25)(x)
    assert torch.equal(live_per_site(out), live_per_site(x).clamp(max=k))


def test_fc_layer_live_count_is_26_like_kwta():
    """
    The exact number the pre-registration commits to: 128 units, k = 0.2 -> 26
    nominal, and 26.0 live for both arms at the real network's zero fraction.
    """
    x = post_relu((256, 128), zero_frac=0.4, seed=5)   # ~70% zeros, as measured
    torch.manual_seed(6)
    matched = RandomKMatchedChannel(k=0.2)(x)
    kwta = KWinnersTakeAllV2(k=0.2, dim="channel")(x)
    assert live_per_site(matched).float().mean().item() == pytest.approx(26.0, abs=0.05)
    assert live_per_site(kwta).float().mean().item() == pytest.approx(26.0, abs=0.05)


def test_old_nominal_control_does_NOT_match_which_is_why_this_exists():
    """
    Regression witness for the confound. If this ever starts passing as a match,
    either `RandomKChannel` changed or the input stopped having zeros -- and in
    the latter case every count test above has gone vacuous too.
    """
    x = post_relu((256, 128), zero_frac=0.4, seed=7)
    torch.manual_seed(8)
    nominal = live_per_site(RandomKChannel(k=0.2)(x)).float().mean().item()
    kwta = live_per_site(KWinnersTakeAllV2(k=0.2, dim="channel")(x)).float().mean().item()
    assert kwta / nominal > 2.5, (kwta, nominal)   # measured 3.4x in the network


# --------------------------------------------------------------------------
# Selection semantics
# --------------------------------------------------------------------------

def test_never_selects_a_zero_while_live_units_remain():
    x = post_relu((16, 16, 5, 5), zero_frac=0.3, seed=9)
    k = 4
    torch.manual_seed(10)
    out = RandomKMatchedChannel(k=0.25)(x)
    enough = live_per_site(x) >= k
    assert enough.any(), "no site had k live units; test would be vacuous"
    assert torch.all(live_per_site(out)[enough] == k)


def test_keeps_all_nonzeros_when_fewer_than_k_exist():
    x = torch.zeros(4, 16, 3, 3)
    x[:, 0] = 1.0
    x[:, 1] = 2.0            # only 2 live channels, k = 4
    torch.manual_seed(11)
    out = RandomKMatchedChannel(k=0.25)(x)
    assert torch.all(live_per_site(out) == 2)
    assert torch.equal(out, x)   # nothing live was dropped, nothing dead revived


def test_selection_is_uniform_over_the_live_channels():
    """Uniform among live units -- not among all units, and not by magnitude."""
    torch.manual_seed(12)
    x = torch.rand(400, 16, 5, 5) + 0.1      # all live
    out = RandomKMatchedChannel(k=0.25)(x)
    rate = (out != 0).float().mean(dim=(0, 2, 3))
    assert torch.allclose(rate, torch.full((16,), 0.25), atol=0.02), rate


def test_selection_ignores_magnitude_where_kwta_does_not():
    """The defining property of the control: a dominant channel gets no edge."""
    torch.manual_seed(13)
    x = torch.rand(300, 16, 5, 5) + 0.1
    x[:, 3] *= 100.0
    out = RandomKMatchedChannel(k=0.25)(x)
    assert abs((out[:, 3] != 0).float().mean().item() - 0.25) < 0.03
    kout = KWinnersTakeAllV2(k=0.25, dim="channel")(x)
    assert (kout[:, 3] != 0).float().mean().item() > 0.99


def test_a_dead_channel_is_never_selected_but_a_weak_live_one_can_be():
    torch.manual_seed(14)
    x = torch.rand(200, 16, 4, 4) + 0.5
    x[:, 5] = 0.0            # structurally dead
    x[:, 6] = 1e-6           # live but negligible
    out = RandomKMatchedChannel(k=0.25)(x)
    assert (out[:, 5] != 0).sum().item() == 0
    assert abs((out[:, 6] != 0).float().mean().item() - 0.25) < 0.04


def test_surviving_values_pass_through_unchanged_and_unrescaled():
    x = post_relu((6, 16, 5, 5), zero_frac=0.3, seed=15)
    torch.manual_seed(16)
    out = RandomKMatchedChannel(k=0.25)(x)
    surv = out != 0
    assert torch.equal(out[surv], x[surv])


# --------------------------------------------------------------------------
# Shapes, modes, gradients
# --------------------------------------------------------------------------

def test_2d_input_selects_k_live_units_per_sample():
    x = post_relu((32, 128), zero_frac=0.2, seed=17)
    torch.manual_seed(18)
    out = RandomKMatchedChannel(k=0.2)(x)
    assert out.shape == x.shape
    assert torch.equal(live_per_site(out), live_per_site(x).clamp(max=26))


def test_masking_applies_in_eval_mode_like_kwta():
    x = post_relu((4, 16, 7, 7), seed=19)
    torch.manual_seed(20)
    m = RandomKMatchedChannel(k=0.25).eval()
    out = m(x)
    assert (out == 0).any()
    assert (out != 0).sum().item() < (x != 0).sum().item()


def test_gradient_is_masked_exactly():
    torch.manual_seed(21)
    x = (torch.relu(torch.randn(4, 16, 5, 5))).requires_grad_(True)
    out = RandomKMatchedChannel(k=0.25)(x)
    out.sum().backward()
    survived = out != 0
    assert survived.any()
    assert torch.all(x.grad[survived] == 1.0)
    assert torch.all(x.grad[~survived] == 0.0)


def test_builder_arm_trains_end_to_end():
    from architecture.model import build_randk_matched_model

    torch.manual_seed(22)
    model = build_randk_matched_model()
    model.train()
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    nn.CrossEntropyLoss()(model(x), y).backward()
    g = [p.grad for p in model.parameters() if p.grad is not None]
    assert g and any((t != 0).any() for t in g)
    assert not any(torch.isnan(t).any() for t in g)


def test_builder_arm_matches_kwta_live_counts_inside_the_real_network():
    """
    End-to-end version of the headline contract: same weights, same input, the
    live-unit count at every activation site must agree with the kWTA arm.
    """
    from architecture.model import build_kwta_v2_model, build_randk_matched_model

    torch.manual_seed(23)
    matched = build_randk_matched_model(k=0.2).eval()
    torch.manual_seed(23)
    kwta = build_kwta_v2_model(k=0.2, dim="channel", boost=0.0).eval()

    counts = {}

    def attach(model, tag):
        for name in ("features.1", "features.4", "classifier.2"):
            mod = dict(model.named_modules())[name]
            mod.register_forward_hook(
                lambda _m, _i, o, key=(tag, name): counts.__setitem__(
                    key, (o.detach() != 0).sum(dim=1).float().mean().item()))

    attach(matched, "matched")
    attach(kwta, "kwta")

    torch.manual_seed(24)
    xb = torch.randn(64, 1, 28, 28)
    with torch.no_grad():
        matched(xb)
        kwta(xb)

    for name, nominal in (("features.1", 3), ("features.4", 6), ("classifier.2", 26)):
        m, k = counts[("matched", name)], counts[("kwta", name)]
        assert m == pytest.approx(k, abs=1e-6), (name, m, k)
        assert m <= nominal + 1e-6, (name, m, nominal)
