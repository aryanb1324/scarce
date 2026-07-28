"""
Unit tests for the protocol v2 data pipeline.

data/skills.md: "Add an automated check that asserts zero overlap between splits,
and run it after any change here." v1 shipped `assert_no_overlap` in
data/mnist.py but never called it from a test, so the highest-stakes directory in
the project had zero coverage. This closes that.

These tests use synthetic labels, not real MNIST, so they run in milliseconds and
need no download.

Run with: python -m pytest tests/test_sampling.py -v
"""

import pytest
import torch

from data.mnist_v2 import (
    assert_disjoint,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

N_FAKE = 6000


@pytest.fixture
def labels():
    # 600 of each digit, deliberately not in sorted order
    y = torch.arange(N_FAKE) % 10
    return y[torch.randperm(N_FAKE, generator=torch.Generator().manual_seed(7))]


def test_val_split_is_deterministic():
    a_pool, a_val = train_val_split(N_FAKE, val_size=1000)
    b_pool, b_val = train_val_split(N_FAKE, val_size=1000)
    assert torch.equal(a_val, b_val)
    assert torch.equal(a_pool, b_pool)


def test_val_split_is_a_partition():
    pool, val = train_val_split(N_FAKE, val_size=1000)
    assert val.numel() == 1000
    assert pool.numel() == N_FAKE - 1000
    assert_disjoint(pool, val)
    assert set(pool.tolist()) | set(val.tolist()) == set(range(N_FAKE))


def test_stratified_subset_is_exactly_balanced(labels):
    pool, _ = train_val_split(N_FAKE, val_size=1000)
    idx = stratified_subset(labels, pool, n_per_class=20, seed=0)
    assert idx.numel() == 200
    counts = torch.bincount(labels[idx], minlength=10)
    assert torch.all(counts == 20), counts


def test_subset_never_touches_validation(labels):
    """The leakage check that matters. If this ever fails, every result is void."""
    pool, val = train_val_split(N_FAKE, val_size=1000)
    for seed in range(5):
        idx = stratified_subset(labels, pool, n_per_class=20, seed=seed)
        assert_disjoint(idx, val)


def test_subset_is_reproducible_and_seed_sensitive(labels):
    pool, _ = train_val_split(N_FAKE, val_size=1000)
    a = stratified_subset(labels, pool, n_per_class=20, seed=0)
    b = stratified_subset(labels, pool, n_per_class=20, seed=0)
    c = stratified_subset(labels, pool, n_per_class=20, seed=1)
    assert torch.equal(a, b), "same seed must give the same subset"
    assert not torch.equal(a, c), "different seeds must give different subsets"


def test_oversized_request_raises_rather_than_silently_shrinking(labels):
    """
    A silently-truncated subset would mean the run's real data budget differs from
    the config's claimed budget -- exactly the kind of invisible mismatch that
    invalidates a data-efficiency comparison.
    """
    pool, _ = train_val_split(N_FAKE, val_size=1000)
    with pytest.raises(ValueError):
        stratified_subset(labels, pool, n_per_class=10_000, seed=0)


def test_norm_stats_come_from_the_subset_not_the_full_set():
    """
    Guards the v1 leak: normalization statistics must reflect only the examples
    the model is allowed to see.
    """
    x = torch.zeros(1000, 4, 4, dtype=torch.uint8)
    x[:500] = 255  # first half bright, second half black
    bright = torch.arange(0, 500)
    dark = torch.arange(500, 1000)

    m_bright, _ = subset_norm_stats(x, bright)
    m_dark, _ = subset_norm_stats(x, dark)
    m_all, _ = subset_norm_stats(x, torch.arange(1000))

    assert m_bright == pytest.approx(1.0, abs=1e-4)
    assert m_dark == pytest.approx(0.0, abs=1e-4)
    assert m_all == pytest.approx(0.5, abs=1e-4)
    assert m_bright != m_all, "subset stats must not equal full-set stats"


def test_assert_disjoint_actually_catches_overlap():
    """A leakage assertion that never fires is worse than none -- verify it fires."""
    with pytest.raises(AssertionError):
        assert_disjoint(torch.tensor([1, 2, 3]), torch.tensor([3, 4, 5]))
    assert_disjoint(torch.tensor([1, 2]), torch.tensor([3, 4]))  # must not raise


def test_zero_std_subset_does_not_divide_by_zero():
    x = torch.full((10, 4, 4), 128, dtype=torch.uint8)
    _, std = subset_norm_stats(x, torch.arange(10))
    assert std == 1.0
