"""Unit tests for the aryanet library layer.

Tiny toy tensors only -- these check shapes, statistics, and the decision rule's
refusals, not model quality (tests/skills.md).
"""

import math

import pytest
import torch

import aryanet
from aryanet.nets import build_net
from aryanet.protocol import Normalizer, as_float_tensor, stratified_split
from aryanet.stats import decide, paired_stats, seeds_needed, t_pvalue


# ------------------------------- shape adaptation ----------------------------

@pytest.mark.parametrize("shape", [(1, 28, 28), (3, 32, 32), (1, 8, 8), (20,)])
def test_build_net_adapts_to_input_shape(shape):
    net = build_net(shape, num_classes=7)
    out = net(torch.randn(4, *shape))
    assert out.shape == (4, 7)


def test_build_net_rejects_unsupported_rank():
    with pytest.raises(ValueError):
        build_net((2, 3, 4, 5), num_classes=3)


def test_penultimate_is_hookable_and_feeds_the_head():
    """Diagnostics depend on this module being the representation the head reads."""
    net = build_net((1, 12, 12), num_classes=3)
    seen = []
    h = net.penultimate.register_forward_hook(lambda m, i, o: seen.append(o.shape))
    net(torch.randn(2, 1, 12, 12))
    h.remove()
    assert seen and seen[0][-1] == 128


def test_every_mechanism_builds_and_backprops():
    for cand in aryanet.default_candidates() + aryanet.control_candidates():
        net = build_net((1, 10, 10), 4, cand.factory)
        loss = net(torch.randn(6, 1, 10, 10)).sum()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, cand.name
        assert any(g.abs().sum() > 0 for g in grads), cand.name


# --------------------------------- data handling -----------------------------

def test_uint8_is_rescaled_but_float_is_left_alone():
    u8 = torch.randint(0, 256, (5, 1, 4, 4), dtype=torch.uint8)
    assert as_float_tensor(u8).max() <= 1.0
    f = torch.randn(5, 3) * 100
    assert torch.allclose(as_float_tensor(f), f)


def test_normalizer_uses_only_what_it_was_fitted_on():
    """Statistics from held-out data would leak; fit must ignore anything else."""
    train = torch.ones(10, 2) * 5.0
    train[:, 1] = 1.0
    n = Normalizer.fit(train)
    assert torch.allclose(n.mean.flatten(), torch.tensor([5.0, 1.0]))


def test_normalizer_survives_a_constant_feature():
    n = Normalizer.fit(torch.ones(8, 3))
    assert torch.isfinite(n(torch.ones(8, 3))).all()


def test_split_is_disjoint_balanced_and_refuses_impossible_requests():
    y = torch.tensor([0] * 20 + [1] * 20)
    tr, va = stratified_split(y, val_per_class=5)
    assert set(tr.tolist()).isdisjoint(set(va.tolist()))
    assert torch.bincount(y[va]).tolist() == [5, 5]
    with pytest.raises(ValueError):
        stratified_split(y, val_per_class=50)


# ----------------------------------- statistics ------------------------------

def test_t_pvalue_matches_known_values():
    # t=2.228, df=10 is the classic two-sided 0.05 critical value.
    assert abs(t_pvalue(2.228, 10) - 0.05) < 0.002
    assert t_pvalue(0.0, 5) == pytest.approx(1.0)


def test_paired_stats_are_paired_not_two_means():
    d = [1.0, 1.2, 0.8, 1.1, 0.9]
    s = paired_stats("arm", d)
    assert s.mean == pytest.approx(1.0)
    assert s.signs == "+++++"
    assert s.sign_fraction == 1.0
    assert s.se == pytest.approx(s.sd / math.sqrt(5))


def test_seeds_needed_grows_with_noise_and_shrinks_with_effect():
    assert seeds_needed(1.0, 1.375) > seeds_needed(1.0, 0.5)
    assert seeds_needed(2.0, 1.0) < seeds_needed(0.5, 1.0)


# ------------------------------- the decision rule ---------------------------

def _arms(dense_ok=True, **named):
    out = [paired_stats("dense", [0.0] * 5)] if dense_ok else []
    for n, d in named.items():
        out.append(paired_stats(n, d))
    return out


def test_declines_when_the_effect_is_inside_the_noise():
    """The 100-label lesson: a real-looking mean with a huge spread is nothing."""
    noisy = [1.4, -1.5, 0.9, -1.2, 0.6]     # mean ~0.04, sd ~1.3
    d = decide(_arms(mech=noisy))
    assert d.winner == "dense"
    assert not d.confident


def test_declines_on_one_seed_because_variance_is_unmeasurable():
    d = decide(_arms(mech=[5.0]))
    assert d.winner == "dense"
    assert "seed" in d.reason


def test_declines_when_one_lucky_seed_carries_the_mean():
    d = decide(_arms(mech=[9.0, -0.2, -0.3, -0.1, -0.2]))
    assert d.winner == "dense"


def test_names_a_winner_when_the_effect_is_real_and_consistent():
    d = decide(_arms(mech=[1.5, 1.6, 1.4, 1.7, 1.5]))
    assert d.winner == "mech"
    assert d.confident


def test_bonferroni_makes_more_arms_a_higher_bar():
    """Testing many mechanisms against one reference inflates false positives.

    This arm sits at p ~= 0.024: significant on its own, not significant once
    the threshold is split four ways. Identical evidence, different verdict --
    which is the correction doing its job.
    """
    borderline = [0.1, 0.3, 0.5, 0.7, 0.9]        # mean 0.5, t ~= 3.54, p ~= 0.024
    assert paired_stats("a", borderline).p < 0.05
    assert paired_stats("a", borderline).p > 0.05 / 4

    alone = decide(_arms(a=borderline))
    many = decide(_arms(a=borderline,
                        b=[0.1, -0.1, 0.0, 0.1, -0.1],
                        c=[0.0, 0.1, -0.1, 0.0, 0.1],
                        d=[-0.1, 0.0, 0.1, -0.1, 0.0]))
    assert alone.winner == "a" and alone.confident
    assert many.winner == "dense" and not many.confident


# Real per-seed deltas vs dense from examples/mnist_low_data, 5 seeds. Using the
# measured values rather than invented ones matters here: a hand-written fixture
# made the two arms track each other far more tightly than they really do, which
# turned a genuine tie into a significant difference and hid the behaviour under
# test. Real data is the fixture.
REAL_300 = {"kwta": [0.46, 1.76, 2.48, 2.12, 2.74],
            "dropout": [1.16, 1.78, 1.70, 1.00, 2.78]}
REAL_600 = {"kwta": [1.50, 0.70, 2.30, 1.88, 2.12],
            "dropout": [0.94, 0.50, 1.72, 0.96, 1.76]}
COMPLEXITY = {"kwta": 2, "dropout": 1}


def test_an_unresolvable_margin_does_not_sell_the_fancier_arm():
    """300 MNIST labels: kwta +1.91, dropout +1.68, head-to-head p = 0.52.

    Both clear the significance bar against dense, but the margin between them is
    noise -- two of five seeds favour dropout. Declaring kwta the winner there
    would be the library flattering its own mechanism, so the tie breaks toward
    the standard component.
    """
    d = decide(_arms(kwta=REAL_300["kwta"], dropout=REAL_300["dropout"]),
               complexity=COMPLEXITY)
    assert d.confident
    assert d.winner == "dropout", d.reason
    assert "kwta" in d.tied_with


def test_a_resolvable_margin_does_pick_the_stronger_arm():
    """600 MNIST labels: same two arms, head-to-head p = 0.0124 < 0.0167.

    Complexity must break ties, never override evidence. Here kwta beats dropout
    on all five seeds by a resolvable margin, so it wins despite being the more
    complex component.
    """
    d = decide(_arms(kwta=REAL_600["kwta"], dropout=REAL_600["dropout"]),
               complexity=COMPLEXITY)
    assert d.confident
    assert d.winner == "kwta", d.reason
    assert d.tied_with == []


def test_reports_seeds_needed_when_it_declines():
    d = decide(_arms(mech=[1.4, -1.5, 0.9, -1.2, 0.6]))
    assert "seeds" in d.reason


def test_requires_a_dense_reference():
    """Without the reference arm there is nothing to measure a delta against.

    An explicit val set is supplied so this fails on the missing reference and
    not on the auto-split, which would raise for an unrelated reason.
    """
    x, y = torch.randn(8, 3), torch.tensor([0, 1] * 4)
    with pytest.raises(ValueError, match="dense"):
        aryanet.fit(x, y, x_val=x, y_val=y, seeds=1,
                    candidates=[aryanet.default_candidates()[1]], verbose=False)


def test_mismatched_x_and_y_is_caught_before_training():
    with pytest.raises(ValueError, match="rows"):
        aryanet.fit(torch.randn(8, 3), torch.tensor([0, 1, 0]), verbose=False)
