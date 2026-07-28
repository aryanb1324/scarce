"""
MNIST loading + reproducible low-data subset sampling.

This file is the highest-stakes part of the whole project for silent
correctness bugs: if the "reduced data" subset isn't sampled correctly
(e.g. it overlaps with val/test, or the fraction is wrong), every
data-efficiency comparison downstream is invalid. See data/skills.md.
"""

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_mnist_datasets(data_dir: str = "./data_cache"):
    """
    Returns (train_dataset, test_dataset). Standard MNIST normalization.
    Downloads to `data_dir` on first run (requires internet access on
    your machine — this won't run inside a sandboxed environment with no
    network).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def sample_train_subset(train_dataset, fraction: float, seed: int) -> Subset:
    """
    Returns a reproducible random subset of the TRAINING set only.

    Never call this on the test set. `seed` fully determines which
    examples are chosen, so a given (fraction, seed) pair is always the
    same subset — this is what makes a "trained on 10% of data" result
    reproducible and comparable across the baseline and experimental runs.
    """
    assert 0 < fraction <= 1.0, "fraction must be in (0, 1]"

    n_total = len(train_dataset)
    n_subset = max(1, int(round(n_total * fraction)))

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    indices = perm[:n_subset].tolist()

    return Subset(train_dataset, indices)


def assert_no_overlap(subset_a: Subset, subset_b: Subset) -> None:
    """
    Sanity check: confirms two subsets don't share underlying indices.
    Call this in tests/ after any change to sampling logic — see root
    skills.md self-improvement rule about not skipping this.
    """
    idx_a = set(subset_a.indices)
    idx_b = set(subset_b.indices)
    overlap = idx_a & idx_b
    assert not overlap, f"Data leakage: {len(overlap)} overlapping indices"
