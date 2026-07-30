"""
Unit tests for the CIFAR-100 data pipeline.

data/skills.md: "a rule in a skills file is not in force until something executes
it." Every guarantee `data/cifar100.py` claims gets a test here that fails when the
guarantee is violated -- split disjointness, exact class balance across all 100
classes, seed reproducibility, raise-don't-truncate on an oversized budget,
subset-derived normalization statistics, per-channel (not pooled) normalization,
and that the leakage assertion itself actually fires.

These mirror `tests/test_cifar10.py` and, like it, run on synthetic labels rather
than downloading CIFAR-100, so the suite stays in milliseconds and is CI-safe.
Every random draw is seeded. The one thing that cannot be checked synthetically --
that torchvision's arrays land as (N, 3, 32, 32) uint8 -- is checked by
`test_to_chw_layout` on a fake array of the real shape.

Run with: python -m pytest tests/test_cifar100.py -v
"""

import numpy as np
import pytest
import torch

from data.cifar100 import (
    CIFAR100_VAL_SPLIT_SEED,
    MAX_SHOTS_PER_CLASS,
    NUM_CLASSES,
    VAL_PER_CLASS,
    _to_chw,
    assert_disjoint,
    make_batch,
    preflight,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

# 100 classes x 50 fake images = 5,000 synthetic examples, shuffled. Small so the
# suite stays in milliseconds; the real dataset has 500/class.
FAKE_PER_CLASS = 50
N_FAKE = NUM_CLASSES * FAKE_PER_CLASS
VAL_PC = 10          # scaled-down stand-in for the real VAL_PER_CLASS (100)


@pytest.fixture
def labels():
    y = torch.arange(N_FAKE) % NUM_CLASSES
    return y[torch.randperm(N_FAKE, generator=torch.Generator().manual_seed(7))]


# --------------------------- the frozen val split ---------------------------

def test_val_split_is_deterministic(labels):
    a_pool, a_val = train_val_split(labels, val_per_class=VAL_PC)
    b_pool, b_val = train_val_split(labels, val_per_class=VAL_PC)
    assert torch.equal(a_val, b_val)
    assert torch.equal(a_pool, b_pool)


def test_val_split_is_a_partition(labels):
    pool, val = train_val_split(labels, val_per_class=VAL_PC)
    assert val.numel() == VAL_PC * NUM_CLASSES
    assert pool.numel() == N_FAKE - VAL_PC * NUM_CLASSES
    assert_disjoint(pool, val)
    assert set(pool.tolist()) | set(val.tolist()) == set(range(N_FAKE))


def test_val_split_is_class_balanced(labels):
    """The holdout is stratified across all 100 classes, so the pool ceiling is
    exact and seed-independent."""
    _, val = train_val_split(labels, val_per_class=VAL_PC)
    counts = torch.bincount(labels[val], minlength=NUM_CLASSES)
    assert torch.all(counts == VAL_PC), counts


def test_val_split_seed_constant_is_the_frozen_one():
    """Guards the constant itself, AND that it is distinct from the CIFAR-10 one.
    If either changes, recorded CIFAR-100 results stop being comparable, so the
    change should have to break a test."""
    assert CIFAR100_VAL_SPLIT_SEED == 20260729
    assert VAL_PER_CLASS == 100
    assert NUM_CLASSES == 100
    assert MAX_SHOTS_PER_CLASS == 400
    # Must NOT collide with the CIFAR-10 frozen seed -- a shared constant is one
    # careless edit from invalidating the other dataset's results.
    from data.cifar10 import CIFAR_VAL_SPLIT_SEED
    assert CIFAR100_VAL_SPLIT_SEED != CIFAR_VAL_SPLIT_SEED


def test_val_split_raises_when_a_class_is_too_small(labels):
    with pytest.raises(ValueError):
        train_val_split(labels, val_per_class=FAKE_PER_CLASS + 1)


# ---------------------------- subset sampling -------------------------------

def test_stratified_subset_is_exactly_balanced(labels):
    pool, _ = train_val_split(labels, val_per_class=VAL_PC)
    idx = stratified_subset(labels, pool, n_per_class=5, seed=0)
    assert idx.numel() == 5 * NUM_CLASSES
    counts = torch.bincount(labels[idx], minlength=NUM_CLASSES)
    assert torch.all(counts == 5), counts


def test_subset_never_touches_validation(labels):
    """The leakage check that matters. If this ever fails, every result is void."""
    pool, val = train_val_split(labels, val_per_class=VAL_PC)
    for seed in range(5):
        idx = stratified_subset(labels, pool, n_per_class=5, seed=seed)
        assert_disjoint(idx, val)


def test_subset_is_reproducible_and_seed_sensitive(labels):
    pool, _ = train_val_split(labels, val_per_class=VAL_PC)
    a = stratified_subset(labels, pool, n_per_class=5, seed=0)
    b = stratified_subset(labels, pool, n_per_class=5, seed=0)
    c = stratified_subset(labels, pool, n_per_class=5, seed=1)
    assert torch.equal(a, b), "same seed must give the same subset"
    assert not torch.equal(a, c), "different seeds must give different subsets"


def test_oversized_request_raises_rather_than_silently_shrinking(labels):
    """
    A silently-truncated subset would mean the run's real data budget differs from
    the config's claimed budget -- an invisible mismatch that invalidates a
    data-efficiency comparison.
    """
    pool, _ = train_val_split(labels, val_per_class=VAL_PC)
    with pytest.raises(ValueError):
        stratified_subset(labels, pool, n_per_class=10_000, seed=0)
    # And exactly at the boundary: the pool holds FAKE_PER_CLASS - VAL_PC per class.
    ceiling = FAKE_PER_CLASS - VAL_PC
    with pytest.raises(ValueError):
        stratified_subset(labels, pool, n_per_class=ceiling + 1, seed=0)
    ok = stratified_subset(labels, pool, n_per_class=ceiling, seed=0)
    assert ok.numel() == ceiling * NUM_CLASSES


# ------------------------- normalization statistics -------------------------

def test_norm_stats_come_from_the_subset_not_the_full_set():
    """Guards the v1 leak: statistics must reflect only examples the model may see."""
    x = torch.zeros(1000, 3, 4, 4, dtype=torch.uint8)
    x[:500] = 255                                  # first half white, second black
    bright, dark = torch.arange(0, 500), torch.arange(500, 1000)

    m_bright, _ = subset_norm_stats(x, bright)
    m_dark, _ = subset_norm_stats(x, dark)
    m_all, _ = subset_norm_stats(x, torch.arange(1000))

    assert torch.allclose(m_bright, torch.ones(3), atol=1e-4)
    assert torch.allclose(m_dark, torch.zeros(3), atol=1e-4)
    assert torch.allclose(m_all, torch.full((3,), 0.5), atol=1e-4)
    assert not torch.allclose(m_bright, m_all), "subset stats must not equal full-set"


def test_norm_stats_are_per_channel_not_pooled():
    """CIFAR's three channels have genuinely different statistics; a scalar mean
    would silently re-introduce the thing this function exists to prevent."""
    x = torch.zeros(64, 3, 4, 4, dtype=torch.uint8)
    x[:, 0] = 255
    x[:, 1] = 128
    x[:, 2] = 0
    mean, std = subset_norm_stats(x, torch.arange(64))
    assert mean.shape == (3,) and std.shape == (3,)
    assert mean[0] > mean[1] > mean[2]


def test_zero_std_channel_does_not_divide_by_zero():
    x = torch.full((10, 3, 4, 4), 128, dtype=torch.uint8)
    _, std = subset_norm_stats(x, torch.arange(10))
    assert torch.all(std > 0)
    xb, _ = make_batch(x, torch.zeros(10, dtype=torch.long), torch.arange(10),
                       *subset_norm_stats(x, torch.arange(10)), device=torch.device("cpu"))
    assert torch.isfinite(xb).all()


def test_make_batch_shape_and_standardization():
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(200, 3, 32, 32, generator=g) * 255).to(torch.uint8)
    y = torch.arange(200) % NUM_CLASSES
    idx = torch.arange(200)
    mean, std = subset_norm_stats(x, idx)
    xb, yb = make_batch(x, y, idx, mean, std, torch.device("cpu"))
    assert xb.shape == (200, 3, 32, 32) and xb.dtype == torch.float32
    assert torch.equal(yb, y)
    # standardized on its own statistics -> ~0 mean, ~1 std per channel
    assert torch.allclose(xb.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(xb.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-3)


def test_make_batch_does_not_mutate_the_source_tensor():
    """`make_batch` divides in place on a freshly-indexed copy. If it ever divided
    in place on a view, the dataset would be silently corrupted mid-run."""
    x = torch.full((8, 3, 4, 4), 200, dtype=torch.uint8)
    before = x.clone()
    mean, std = subset_norm_stats(x, torch.arange(8))
    make_batch(x, torch.zeros(8, dtype=torch.long), torch.arange(8), mean, std,
               torch.device("cpu"))
    assert torch.equal(x, before)


# ------------------------------ the checks themselves -----------------------

def test_assert_disjoint_actually_catches_overlap():
    """A leakage assertion that never fires is worse than none -- verify it fires."""
    with pytest.raises(AssertionError):
        assert_disjoint(torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5]))
    assert_disjoint(torch.tensor([1, 2]), torch.tensor([3, 4]))  # must not raise


def test_preflight_runs_the_checks_and_returns_every_subset(labels):
    """`preflight` is where `assert_disjoint` is actually CALLED in a run. v1's
    equivalent existed but was invoked from nowhere."""
    pool, val = train_val_split(labels, val_per_class=VAL_PC)
    subsets = preflight(labels, pool, val, shots_per_class=[5, 10], seeds=[0, 1],
                        verbose=False)
    assert set(subsets) == {(5, 0), (5, 1), (10, 0), (10, 1)}
    for (shots, _), idx in subsets.items():
        assert idx.numel() == shots * NUM_CLASSES
        assert_disjoint(idx, val)


def test_preflight_propagates_an_impossible_budget(labels):
    pool, val = train_val_split(labels, val_per_class=VAL_PC)
    with pytest.raises(ValueError):
        preflight(labels, pool, val, shots_per_class=[5, 99_999], seeds=[0],
                  verbose=False)


# ------------------------------- tensor layout ------------------------------

def test_to_chw_layout():
    """torchvision hands CIFAR back as (N, H, W, C) uint8 numpy. Getting this
    permutation wrong trains a perfectly convergent model on transposed images."""
    arr = np.zeros((7, 32, 32, 3), dtype=np.uint8)
    arr[:, :, :, 0] = 255                       # red channel only
    t = _to_chw(arr)
    assert t.shape == (7, 3, 32, 32) and t.dtype == torch.uint8
    assert t[:, 0].min().item() == 255
    assert t[:, 1].max().item() == 0
    assert t.is_contiguous()
