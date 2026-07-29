"""Unit tests for the architecture axis and the staged screen-then-confirm search.

Tiny toy tensors and recorded per-seed numbers only -- these check structure,
compute accounting, and the statistical guarantees, not model quality
(tests/skills.md).

Every fixture used to test a statistical decision is a REAL measured value from a
recorded run, cited at its definition. `tests/skills.md` records why: a
hand-written fixture made two arms track each other far too tightly seed-for-seed,
turned a genuine tie into a significant difference, and read as a broken feature.
Invented numbers smuggle in an assumption about the correlation structure, which
is usually the exact quantity the rule is deciding on.
"""

import math

import numpy as np
import pytest
import torch

import aryanet
from aryanet.architectures import (
    CNN,
    CNN_NARROW,
    CNN_WIDE,
    LINEAR,
    REFERENCE_ARM,
    Arm,
    cross,
    default_architectures,
    default_arms,
    resolve_arms,
)
from aryanet.mechanisms import control_candidates, default_candidates
from aryanet.nets import build_cnn, build_linear, build_net, count_parameters
from aryanet.protocol import TrainConfig
from aryanet.staged import (
    BUDGETS,
    Budget,
    contamination_check,
    dropped,
    resolve_budget,
    screen,
)
from aryanet.stats import Paired, decide, paired_stats


# ============================================================================
# RECORDED FIXTURES
#
# Per-seed test-accuracy deltas vs dense, 8 paired seeds, MNIST, from
# experiments/results/20260728_151333_stage2_control/results.csv (protocol
# stage2_control, commit ecf887f). Recomputed from that CSV, not transcribed
# from a summary.
# ============================================================================

KWTA_300 = [0.71, 0.77, 1.57, 1.58, 3.12, 1.73, 2.46, 0.71]      # +1.58, sd 0.87
DROPOUT_300 = [1.05, 1.14, 2.18, 1.33, 3.24, 1.99, 2.26, 0.38]   # +1.70, sd 0.90
KWTA_600 = [1.11, 1.67, 1.43, 1.71, 1.24, 1.84, 2.73, 1.53]      # +1.66, sd 0.50
DROPOUT_600 = [0.26, 0.23, 0.39, 0.61, 1.32, 0.46, 1.70, 0.22]   # +0.65, sd 0.56
RANDK_300 = [-11.44, -10.99, -10.86, -12.24, -9.14,
             -10.02, -14.04, -11.53]                             # -11.28, a real loser


# --------------------------- the builders are additive ------------------------

def test_build_net_is_byte_for_byte_unchanged_by_the_new_axis():
    """The live CIFAR run imports `build_net`; `build_cnn` must not have moved it.

    Not a style check. `set_seed(seed)` runs immediately before each arm is
    built, so parameter initialization consumes RNG in a fixed order. If
    `build_cnn`'s defaults drew even one extra number, the reference arm and the
    capacity arms would start from different initializations at the same seed and
    the comparison would silently stop being paired.
    """
    torch.manual_seed(0)
    a = build_net((1, 28, 28), 10)
    torch.manual_seed(0)
    b = build_cnn((1, 28, 28), 10)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@pytest.mark.parametrize("shape", [(1, 28, 28), (3, 32, 32), (1, 8, 8), (20,)])
def test_every_architecture_builds_and_backprops_on_every_shape(shape):
    for arch in default_architectures():
        net = arch.builder(shape, 5, torch.nn.ReLU)
        out = net(torch.randn(4, *shape))
        assert out.shape == (4, 5), arch.name
        out.sum().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), arch.name


def test_capacity_variants_actually_differ_in_capacity():
    """A capacity axis that does not change capacity is decoration."""
    sizes = {a.name: count_parameters(a.builder((1, 28, 28), 10, torch.nn.ReLU))
             for a in default_architectures()}
    assert sizes["linear"] < sizes["cnn_narrow"] < sizes["cnn"] < sizes["cnn_wide"]
    assert sizes["cnn_wide"] > 3 * sizes["cnn"]
    assert sizes["cnn_narrow"] < sizes["cnn"] / 5


def test_shallow_is_a_depth_ablation_at_matched_parameter_count():
    """cnn_shallow exists to vary DEPTH, so it must not also vary size much.

    Within 5% of the default at 28x28, or it is a second width knob wearing a
    depth label and a delta against it would be uninterpretable.
    """
    sizes = {a.name: count_parameters(a.builder((1, 28, 28), 10, torch.nn.ReLU))
             for a in default_architectures()}
    ratio = sizes["cnn_shallow"] / sizes["cnn"]
    assert 0.95 < ratio < 1.05, ratio
    shallow = [a for a in default_architectures() if a.name == "cnn_shallow"][0]
    convs = [m for m in shallow.builder((1, 28, 28), 10, torch.nn.ReLU).modules()
             if isinstance(m, torch.nn.Conv2d)]
    deep = [m for m in CNN.builder((1, 28, 28), 10, torch.nn.ReLU).modules()
            if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 1 and len(deep) == 2


def test_linear_is_a_linear_model_and_says_so_in_its_parameters():
    """No hidden layer at all -- one weight matrix and one bias, nothing else."""
    net = build_linear((1, 28, 28), 10)
    assert count_parameters(net) == 784 * 10 + 10
    assert len([m for m in net.modules() if isinstance(m, torch.nn.Linear)]) == 1
    assert not [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]


def test_linear_penultimate_is_hookable_so_diagnostics_still_report():
    """Every mechanism ships a measurement (root skills.md rule 5).

    A linear model's representation IS its input, so the hook must fire on the
    flattened input rather than being absent -- an arm that reports no
    diagnostics at all would be exempt from the rule by accident.
    """
    net = build_linear((1, 6, 6), 3)
    seen = []
    h = net.penultimate.register_forward_hook(lambda m, i, o: seen.append(o.shape))
    net(torch.randn(2, 1, 6, 6))
    h.remove()
    assert seen and tuple(seen[0]) == (2, 36)


# ------------------------------- the search space ----------------------------

def test_default_space_is_twelve_arms_with_one_reference():
    arms = default_arms()
    assert len(arms) == 12
    refs = [a for a in arms if a.is_reference]
    assert [a.name for a in refs] == [REFERENCE_ARM]
    assert arms[0].is_reference          # first, so deltas print as runs complete
    assert len({a.name for a in arms}) == 12


def test_linear_is_never_paired_with_a_mechanism_it_cannot_apply():
    """It has no hidden activation site.

    Pairing it anyway and letting `build_linear` drop the mechanism would report
    `linear/kwta_channel_k0.2` as though kWTA had been applied to something. The
    arm would be a lie about what was measured, which is worse than its absence.
    """
    assert not LINEAR.supports_mechanism
    for arms in (default_arms(), cross(default_architectures(), default_candidates())):
        linear_arms = [a for a in arms if a.architecture == "linear"]
        assert len(linear_arms) == 1
        assert linear_arms[0].mechanism == "dense"


def test_cross_restores_the_reference_even_if_the_caller_omitted_it():
    """Every delta is measured against cnn/dense; without it the search is empty.

    Silently substituting a different reference would make the numbers
    incomparable with every recorded run while still printing a table.
    """
    arms = cross([CNN_WIDE], [c for c in default_candidates() if c.name != "dense"])
    assert any(a.is_reference for a in arms)
    assert arms[0].name == REFERENCE_ARM


def test_mechanism_only_search_keeps_its_original_arm_names():
    """Recorded outputs and existing callers must stay comparable."""
    names = [a.name for a in resolve_arms(None, None)]
    assert names[0] == "dense"
    assert set(names) == {c.name for c in default_candidates()}


def test_controls_attach_to_the_reference_architecture_only():
    arms = resolve_arms(None, "default", control_candidates())
    controls = [a for a in arms if "randk" in a.mechanism]
    assert len(controls) == 1
    assert controls[0].architecture == "cnn"


def test_unknown_architecture_keyword_is_refused_not_guessed():
    with pytest.raises(ValueError, match="architectures"):
        resolve_arms(None, "everything")


# ------------------------------ complexity ranks ------------------------------

def test_architecture_dominates_the_complexity_rank():
    """The tie-break must prefer a smaller model over a fancier mechanism.

    Ranks are arch*10 + mech, so every arm on a simpler architecture outranks
    every arm on a more complex one regardless of mechanism. If the mechanism
    could dominate, a tie between `linear/dense` and `cnn_wide/kwta` would return
    the wide kWTA net -- the library selling its own mechanism on noise.
    """
    rank = {a.name: a.complexity for a in default_arms()}
    assert rank["linear/dense"] < rank["cnn_narrow/kwta_channel_k0.2"]
    assert rank["cnn_narrow/kwta_channel_k0.2"] < rank["cnn/dense"]
    assert rank["cnn/dense"] < rank["cnn_wide/dropout_k0.2"]
    # within one architecture, the mechanism orders them, simplest first
    assert (rank["cnn/dense"] < rank["cnn/dropout_k0.2"]
            < rank["cnn/kwta_channel_k0.2"])


def test_complexity_ranks_follow_measured_parameter_counts():
    """"Simpler" is defined by something measurable, not by taste."""
    order = sorted(default_architectures(), key=lambda a: a.complexity)
    sizes = [count_parameters(a.builder((1, 28, 28), 10, torch.nn.ReLU))
             for a in order]
    # cnn_shallow and cnn are matched on purpose, so require non-decreasing
    # rather than strictly increasing.
    assert all(x <= y * 1.05 for x, y in zip(sizes, sizes[1:])), sizes


def test_an_unranked_arm_cannot_win_a_tie_by_default():
    """Ranks are an open scale (0..42 here), so no fixed default sits safely in it.

    The unranked arm is given the HIGHER mean, so it is the arm the rule starts
    from; only the tie-break can dislodge it. A default rank in the middle of the
    scale would hand it the win over a properly-ranked simpler arm.
    """
    d = decide([paired_stats("dense", [0.0] * 8),
                paired_stats("unranked", DROPOUT_300),      # +1.70, no rank
                paired_stats("ranked", KWTA_300)],          # +1.58, rank 30
               complexity={"ranked": 30})
    assert d.confident
    assert d.winner == "ranked", d.reason
    assert "unranked" in d.tied_with


# ------------------------- the decision rule at many arms ---------------------

def _many_arms(n_filler, real_a=KWTA_300, real_b=DROPOUT_300, seeds=8):
    """A realistic 300-label space: two genuine arms plus `n_filler` null arms.

    Filler deltas are the recorded kWTA series shuffled against itself so they
    average to ~0 while keeping the real seed-to-seed spread -- a null arm with
    an unrealistically small sd would make the Bonferroni correction look
    harmless.
    """
    arms = [paired_stats("dense", [0.0] * seeds),
            paired_stats("a", real_a[:seeds]),
            paired_stats("b", real_b[:seeds])]
    centred = [v - sum(real_a[:seeds]) / seeds for v in real_a[:seeds]]
    for i in range(n_filler):
        rolled = centred[i % seeds:] + centred[:i % seeds]
        arms.append(paired_stats("filler{}".format(i), rolled))
    return arms


def test_bonferroni_at_twelve_arms_still_resolves_the_measured_effect():
    """The +1.58 pt kWTA effect must survive a 12-arm search, or the axis is
    unusable: adding architectures would destroy the finding that motivated
    the library.
    """
    d = decide(_many_arms(9), complexity={"a": 32, "b": 31})
    assert d.confident, d.reason
    assert d.winner in ("a", "b")


def test_bonferroni_at_many_arms_costs_a_real_but_bounded_amount():
    """The correction is gentle: t_crit grows like sqrt(log k), not like k.

    Measured here rather than asserted, because "more arms makes it impossible"
    is the intuitive belief and it is wrong -- which is the whole reason the
    architecture axis is affordable at all.
    """
    from aryanet.stats import t_pvalue

    def min_effect(n_tested, sd=0.87, n=8):
        lo, hi = 0.0, 40.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if t_pvalue(mid, n - 1) > 0.05 / n_tested:
                lo = mid
            else:
                hi = mid
        return hi * sd / math.sqrt(n)

    three, eleven = min_effect(3), min_effect(11)
    assert three < eleven                       # more arms is a higher bar
    assert eleven < 1.45 * three                # but under 45% higher, not 4x
    assert eleven < 1.58                        # measured kWTA still clears it


def test_a_small_true_effect_is_lost_at_twelve_arms_and_recovered_by_staging():
    """The honest cost of many arms, and precisely what the two-stage search buys.

    dropout at 600 labels is +0.65 pts, 8/8 seeds, p = 0.0132 -- a real effect.
    It passes a 3-arm search (alpha 0.0167) and FAILS a 12-arm one (alpha
    0.00455). Identical evidence, different verdict. Screening down to 3
    confirming arms restores the looser correction, which is why the staged
    search is not only a compute saving.
    """
    small = paired_stats("dropout", DROPOUT_600)
    assert small.sign_fraction == 1.0            # 8/8 seeds, not a lucky mean
    assert 0.05 / 11 < small.p < 0.05 / 3

    fillers = _many_arms(9)[3:]                  # 9 null arms, realistic spread
    assert len(fillers) == 9

    # 12-arm space: dense + dropout + 10 others => 11 tested, alpha = 0.00455.
    wide = decide([paired_stats("dense", [0.0] * 8), small]
                  + fillers + [paired_stats("extra", fillers[0].deltas[::-1])],
                  complexity={"dropout": 31})
    assert len([a for a in fillers]) + 2 == 11
    assert wide.winner == "dense" and not wide.confident, wide.reason
    assert "0.0045" in wide.reason               # the corrected alpha it failed

    # After screening: dense + dropout + 2 survivors => 3 tested, alpha = 0.0167.
    narrow = decide([paired_stats("dense", [0.0] * 8), small] + fillers[:2],
                    complexity={"dropout": 31})
    assert narrow.winner == "dropout" and narrow.confident, narrow.reason


def test_a_tie_between_a_big_arm_and_a_small_one_returns_the_small_one():
    """300 labels: the recorded kwta (+1.58) and dropout (+1.70) pair, whose
    head-to-head paired difference is unresolvable (p = 0.52 at 5 seeds).

    Re-labelled onto two architectures. The big/exotic arm is given the HIGHER
    mean, so the rule starts from it and only the architecture-dominant rank can
    dislodge it. This is the case the whole axis exists for: when a wide kWTA net
    ties a narrow dropout net, the library must hand back the narrow one.
    """
    d = decide([paired_stats("cnn/dense", [0.0] * 8),
                paired_stats("cnn_wide/kwta_channel_k0.2", DROPOUT_300),  # +1.70
                paired_stats("cnn_narrow/dropout_k0.2", KWTA_300)],       # +1.58
               complexity={"cnn_wide/kwta_channel_k0.2": 42,
                           "cnn_narrow/dropout_k0.2": 11},
               reference="cnn/dense")
    assert d.confident
    assert d.winner == "cnn_narrow/dropout_k0.2", d.reason
    assert "cnn_wide/kwta_channel_k0.2" in d.tied_with


def test_a_resolvable_margin_still_beats_a_lower_complexity_rank():
    """Complexity breaks ties; it must never override evidence.

    600 labels, recorded: kwta +1.66 vs dropout +0.65, and the head-to-head
    difference IS resolvable (8/8 seeds favour kwta). Put kwta on the widest,
    most complex arm and it must still win, or the rank has stopped being a
    tie-break and become a thumb on the scale toward small models.
    """
    d = decide([paired_stats("cnn/dense", [0.0] * 8),
                paired_stats("cnn_wide/kwta_channel_k0.2", KWTA_600),
                paired_stats("linear/dense", DROPOUT_600)],
               complexity={"cnn_wide/kwta_channel_k0.2": 42, "linear/dense": 0},
               reference="cnn/dense")
    assert d.confident
    assert d.winner == "cnn_wide/kwta_channel_k0.2", d.reason
    assert d.tied_with == []


def test_reference_arm_name_is_configurable_without_changing_the_verdict():
    """Renaming the reference must not alter what the rule concludes."""
    a = decide([paired_stats("dense", [0.0] * 8), paired_stats("m", KWTA_600)])
    b = decide([paired_stats("cnn/dense", [0.0] * 8), paired_stats("m", KWTA_600)],
               reference="cnn/dense")
    assert (a.winner, a.confident) == (b.winner, b.confident)
    assert b.reason.count("cnn/dense") >= 1


# ------------------------------ the staged search -----------------------------

def test_budget_run_counts_are_what_the_docs_promise():
    """The compute number is the reason anyone picks a budget; it must be exact."""
    assert BUDGETS["quick"].runs(12) == 12 * 2 + 3 * 5
    assert BUDGETS["standard"].runs(12) == 12 * 3 + 4 * 8
    assert BUDGETS["thorough"].runs(12) == 12 * 5 + 5 * 20
    for b in BUDGETS.values():
        assert b.runs(12) < 12 * b.confirm_seeds     # cheaper than running all
        assert b.pays_off(12)


def test_staging_admits_when_it_costs_more_than_the_direct_search():
    """It is not always cheaper. On 2 arms, `quick` costs 14 runs against 10 --
    the screen eliminates nothing and the screening runs are pure overhead, for a
    test that is also weaker. Charging extra compute for that without saying so
    would be the tool taking someone's time to look sophisticated.
    """
    quick = BUDGETS["quick"]
    assert not quick.pays_off(2)
    assert quick.runs(2) > 2 * quick.confirm_seeds
    assert "costs MORE" in quick.describe(2)
    assert "costs MORE" not in quick.describe(12)


def test_every_budget_confirms_on_at_least_five_seeds():
    """Below 5 the variance estimate is too unstable to support a conclusion.

    The screen may be as cheap as it likes; the stage that DECIDES may not.
    """
    for b in BUDGETS.values():
        assert b.confirm_seeds >= 5, b.name
        assert b.screen_seeds < b.confirm_seeds, b.name


def test_unknown_budget_is_refused_not_guessed():
    with pytest.raises(ValueError, match="budget"):
        resolve_budget("cheap")
    assert resolve_budget(BUDGETS["quick"]) is BUDGETS["quick"]


def test_screen_keeps_the_reference_unconditionally():
    """Deltas are measured against the reference, so screening it on performance
    would push the selection bias into the baseline, where it is invisible.
    """
    stats = [paired_stats("dense", [0.0, 0.0]),
             paired_stats("good", [3.0, 3.2]),
             paired_stats("bad", [-9.0, -8.0])]
    keep = screen(stats, keep=1, reference="dense")
    assert keep[0] == "dense"
    assert keep == ["dense", "good"]


def test_screen_ranks_and_never_tests():
    """No significance threshold, and negative arms are not auto-dropped.

    At 2-3 seeds a p-value tracks luck in the spread, not effect size: measured
    in this repo, a -0.19 pt effect came out 'significant' at n=3 and a -1.58 pt
    effect did not. A screen that tested would be noise with a threshold on it,
    so `keep` slots are filled by rank even when every mean is negative.
    """
    stats = [paired_stats("dense", [0.0, 0.0]),
             paired_stats("less_bad", [-0.2, -0.1]),
             paired_stats("worse", [-5.0, -6.0])]
    assert screen(stats, keep=1, reference="dense") == ["dense", "less_bad"]


def test_screen_breaks_ties_toward_the_simpler_arm():
    """Saturated data ties many arms exactly at 2 seeds.

    Breaking those ties alphabetically eliminates the simpler arm before it can
    be tested, so a search whose expensive stage would have preferred the linear
    baseline never gets to see it. Observed on synthetic data during development,
    where linear/dense tied at +0.00 and was dropped for cnn_narrow/dense.
    """
    stats = [paired_stats("cnn/dense", [0.0, 0.0]),
             paired_stats("cnn_wide/kwta_channel_k0.2", [1.0, 1.0]),
             paired_stats("linear/dense", [1.0, 1.0])]
    keep = screen(stats, keep=1, reference="cnn/dense",
                  complexity={"cnn_wide/kwta_channel_k0.2": 42,
                              "linear/dense": 0})
    assert keep == ["cnn/dense", "linear/dense"]


def test_eliminated_arms_are_reported_not_silently_dropped():
    stats = [paired_stats("dense", [0.0, 0.0]),
             paired_stats("good", [3.0, 3.2]),
             paired_stats("bad", [-9.0, -8.0])]
    gone = dropped(stats, ["dense", "good"])
    assert [a.name for a in gone] == ["bad"]


def test_contamination_check_catches_shared_seeds():
    """The guarantee is structural, but a guarantee that is never checked is a
    comment. `fit` runs this on every staged search and raises rather than
    printing a p-value that does not mean what it says.
    """
    s1 = [paired_stats("m", [1.0, 1.0, 1.0])]
    s2 = [paired_stats("m", [1.0] * 5)]
    assert contamination_check(s1, s2, [0, 1, 2], [3, 4, 5, 6, 7]) is None
    assert contamination_check(s1, s2, [0, 1, 2], [2, 3, 4, 5, 6]) is not None


def test_the_staged_procedure_controls_false_positives_and_pooling_does_not():
    """The contamination argument, measured instead of asserted.

    Simulate a world where every arm is genuinely identical to the reference, so
    any declared winner is a false positive. Paired deltas are drawn N(0, 0.87),
    the sd measured at 300 MNIST labels in the recorded 8-seed run, and the full
    12-arm screen-then-confirm procedure is run 1,500 times.

    A valid procedure declares a winner at most alpha = 5% of the time. Three
    procedures are compared:

      * disjoint seeds (as implemented)  -- must stay at or below nominal;
      * POOLED seeds (the trap)          -- the same runs that selected the arms
                                            also judge them, so the maximum over
                                            11 noisy estimates leaks into the
                                            test statistic and the rate inflates;
      * no screening at all              -- the reference point staging must match.

    Measured at 4,000 replicates during development: 0.026 / 0.060 / 0.025. The
    staged search matches the unscreened one and the pooled variant exceeds
    nominal. If this test ever fails, the seed split has stopped working and the
    reported p-values do not mean what they say.
    """
    rng = np.random.default_rng(0)
    sd, n_arms, s1_n, keep_n, s2_n = 0.87, 12, 3, 3, 8
    names = ["dense"] + ["arm{}".format(i) for i in range(n_arms - 1)]
    rank = {n: i for i, n in enumerate(names)}

    honest = pooled = unscreened = 0
    trials = 1500
    for _ in range(trials):
        d = {n: rng.normal(0.0, sd, s1_n + s2_n) for n in names[1:]}
        s1 = ([paired_stats("dense", [0.0] * s1_n)]
              + [paired_stats(n, d[n][:s1_n]) for n in names[1:]])
        kept = set(screen(s1, keep_n, "dense", rank))
        surv = [n for n in names[1:] if n in kept]

        honest += decide(
            [paired_stats("dense", [0.0] * s2_n)]
            + [paired_stats(n, d[n][s1_n:]) for n in surv],
            complexity=rank).confident
        pooled += decide(
            [paired_stats("dense", [0.0] * (s1_n + s2_n))]
            + [paired_stats(n, d[n]) for n in surv],
            complexity=rank).confident
        unscreened += decide(
            [paired_stats("dense", [0.0] * s2_n)]
            + [paired_stats(n, d[n][s1_n:]) for n in names[1:]],
            complexity=rank).confident

    honest_rate = honest / trials
    pooled_rate = pooled / trials
    unscreened_rate = unscreened / trials

    assert honest_rate <= 0.05, honest_rate
    assert unscreened_rate <= 0.05, unscreened_rate
    # Staging must not cost validity relative to running everything. Slack for
    # Monte Carlo error at 1,500 trials (se ~= 0.004), per tests/skills.md on
    # thresholds over sampled quantities.
    assert honest_rate < unscreened_rate + 0.02
    # And the trap must be detectably worse, or this test proves nothing.
    assert pooled_rate > honest_rate + 0.01, (pooled_rate, honest_rate)


def test_contamination_check_catches_pooled_screening_seeds():
    """The garden-of-forking-paths failure, detected by arithmetic.

    Stage 2 ran 5 seeds; a stage-2 statistic built from 8 deltas has absorbed the
    3 screening seeds that chose the arm, and its p-value is inflated by a
    maximum-over-11-noisy-estimates selection.
    """
    s1 = [paired_stats("m", [1.0, 1.0, 1.0])]
    pooled = [paired_stats("m", [1.0] * 8)]
    problem = contamination_check(s1, pooled, [0, 1, 2], [3, 4, 5, 6, 7])
    assert problem is not None and "pooled" in problem


# ------------------------------ end to end, tiny ------------------------------

def _toy(n=240, classes=3, seed=0):
    """Separable toy images: small enough to train in seconds, real enough that
    the arms do not all tie at 100%."""
    torch.manual_seed(seed)
    w = torch.randn(classes, 1, 8, 8)
    y = torch.randint(0, classes, (n,))
    return w[y] + 1.5 * torch.randn(n, 1, 8, 8), y


TINY = TrainConfig(max_steps=60, eval_every=20, patience=2, batch_size=32)


def test_staged_fit_runs_and_decides_only_on_confirming_seeds():
    x, y = _toy()
    r = aryanet.fit(x, y, architectures="default",
                    budget=Budget("tiny", screen_seeds=2, keep=2, confirm_seeds=5),
                    config=TINY, val_per_class=20, verbose=False)
    assert r.reference == "cnn/dense"
    assert [s.name for s in r.stages] == ["screen", "confirm"]

    screen_seeds = set(r.stages[0].seeds)
    confirm_seeds = set(r.stages[1].seeds)
    assert not (screen_seeds & confirm_seeds)

    # The reported arms are the CONFIRMING statistics: every one has exactly the
    # confirming seed count, so no screening seed was pooled in.
    for a in r.arms:
        assert a.n == len(confirm_seeds), a.name
    assert r.seeds == 5
    assert len(r.arms) == 3                      # reference + 2 survivors
    assert r.runs == 12 * 2 + 3 * 5


def test_staged_fit_screens_out_arms_and_says_which():
    x, y = _toy()
    r = aryanet.fit(x, y, architectures="default",
                    budget=Budget("tiny", screen_seeds=2, keep=2, confirm_seeds=5),
                    config=TINY, val_per_class=20, verbose=False)
    assert len(r.eliminated) == 9
    surviving = {a.name for a in r.arms}
    assert not (surviving & {a.name for a in r.eliminated})
    text = r.report()
    for a in r.eliminated:
        assert a.name in text, a.name


def test_report_renders_and_states_the_parameter_cost():
    """An architecture search varies model size on purpose, so the reader must be
    able to tell a capacity effect from a mechanism effect.
    """
    x, y = _toy()
    r = aryanet.fit(x, y, architectures="default",
                    budget=Budget("tiny", screen_seeds=2, keep=2, confirm_seeds=5),
                    config=TINY, val_per_class=20, verbose=False)
    text = r.report()
    assert "params" in text
    assert "STAGE 1 SCREEN" in text and "STAGE 2 CONFIRM" in text
    assert "ranking only" in text
    assert "{:,}".format(count_parameters(build_linear((1, 8, 8), 3))) in text
    assert r.cost["linear/dense"]["params"] < r.cost["cnn_wide/dense"]["params"]


def test_single_stage_fit_is_unchanged_when_no_budget_is_given():
    """The original API path must not have moved under existing callers."""
    x, y = _toy()
    r = aryanet.fit(x, y, seeds=2, config=TINY, val_per_class=20, verbose=False)
    assert r.reference == "dense"
    assert r.stages == [] and r.eliminated == []
    assert {a.name for a in r.arms} == {c.name for c in default_candidates()}
    assert r.runs == 4 * 2
    assert "STAGE 1" not in r.report()


def test_the_winner_that_is_returned_is_the_model_that_was_trained():
    """The architecture axis makes this failable in a way it was not before.

    With one architecture, loading any arm's weights into any arm's skeleton
    works by accident. With five, a mismatch between the chosen arm and the built
    skeleton raises -- so this asserts the returned `.model` really is the winning
    architecture and can predict.
    """
    x, y = _toy()
    r = aryanet.fit(x, y, architectures="default",
                    budget=Budget("tiny", screen_seeds=2, keep=2, confirm_seeds=5),
                    config=TINY, val_per_class=20, verbose=False)
    arch = r.winner.split("/")[0]
    expected = {"linear": build_linear, "cnn": build_net}.get(arch)
    if expected is not None:
        assert count_parameters(r.model) == count_parameters(
            expected((1, 8, 8), 3))
    assert r.predict(x[:4]).shape == (4,)


def test_a_linear_only_space_still_produces_a_reference_and_an_answer():
    """Someone will pass `architectures=[LINEAR]`; it must not silently degenerate
    into a search with no reference to measure against.
    """
    x, y = _toy()
    r = aryanet.fit(x, y, architectures=[LINEAR], seeds=2, config=TINY,
                    val_per_class=20, verbose=False)
    assert r.reference == "cnn/dense"
    assert {a.name for a in r.arms} == {"cnn/dense", "linear/dense"}
