"""
CIFAR-10 under the protocol v2 data contract.

WHY THIS FILE EXISTS
--------------------
Every recorded result in this project is MNIST-only, and MNIST saturates near
99%, which flatters regularizers of every kind: at 92-97% accuracy a mechanism
that merely damps overfitting looks like a data-efficiency prior. CLAUDE.md calls
CIFAR-10 "the gate on everything" for exactly that reason. This module is the
data half of that gate.

It mirrors `data/mnist_v2.py` function for function so the training protocol can
be reused unchanged, and it carries the same four guarantees, each of which
closes a hole `data/skills.md` records the v1 pipeline actually shipping:

  1. A FROZEN validation split, under its own seed constant. It does NOT reuse
     or import `mnist_v2.VAL_SPLIT_SEED` -- that constant is frozen because every
     recorded v2+ MNIST result depends on it, and a shared mutable constant is
     one careless edit away from invalidating them. `CIFAR_VAL_SPLIT_SEED` is a
     separate frozen constant with the same "do not touch" status.
  2. Stratified subset sampling that RAISES when the requested budget cannot be
     honored. A silently-truncated subset makes the run's real data budget differ
     from the config's claimed budget, which is invisible in the results CSV and
     invalidates the comparison outright.
  3. Normalization statistics computed FROM THE TRAINING SUBSET, per channel.
     CIFAR has three channels with genuinely different statistics, so the v1 MNIST
     mistake (hardcoded full-dataset mean/std reused in the 1% condition) would be
     a materially larger leak here than it was there.
  4. `assert_disjoint`, and it is CALLED -- from `preflight` below, from
     `experiments/train_cifar_v1.py`, and from `tests/test_cifar10.py`. The MNIST
     v1 leakage checker existed but was never invoked from anywhere, so the
     highest-stakes directory in the project had zero coverage. A rule is not in
     force until something executes it.

TWO DELIBERATE DEVIATIONS FROM mnist_v2.py, both documented rather than silent:

  * The validation split is STRATIFIED (500 per class), where MNIST v2 used a
    uniform permutation. CIFAR-10 has exactly 5,000 training images per class, so
    a stratified holdout leaves exactly 4,500 per class in the pool -- a known,
    seed-independent ceiling for `stratified_subset`, instead of MNIST's "roughly
    4,800, depends on the split draw". Same size (5,000), strictly more
    determinism.
  * Normalization is per-channel (3 means, 3 stds) rather than scalar.

Images are held as uint8 (N, 3, 32, 32) -- about 150 MB for train+test -- and
converted per batch, which is the same memory/speed tradeoff mnist_v2 makes and
avoids a DataLoader in the inner loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torchvision import datasets

NUM_CLASSES = 10
VAL_PER_CLASS = 500                 # -> 5,000 validation images, matching MNIST v2
CIFAR_VAL_SPLIT_SEED = 20260728     # FROZEN. Changing this invalidates comparability
                                    # with every recorded CIFAR result. Don't.

# CIFAR-10 train holds exactly 5,000 images per class; the frozen val split removes
# VAL_PER_CLASS of each. This is the hard ceiling for `stratified_subset`.
MAX_SHOTS_PER_CLASS = 5000 - VAL_PER_CLASS


def load_cifar10_tensors(data_dir: str = "./data_cache"):
    """Returns (x_train_u8, y_train, x_test_u8, y_test). No normalization applied.

    Images are (N, 3, 32, 32) uint8, channel-first, so `make_batch` does not have
    to permute on every batch. Same `data_dir` as MNIST, so both datasets live in
    one gitignored cache.
    """
    tr = datasets.CIFAR10(data_dir, train=True, download=True)
    te = datasets.CIFAR10(data_dir, train=False, download=True)
    return (
        _to_chw(tr.data),
        torch.as_tensor(np.asarray(tr.targets), dtype=torch.long),
        _to_chw(te.data),
        torch.as_tensor(np.asarray(te.targets), dtype=torch.long),
    )


def _to_chw(arr) -> torch.Tensor:
    """(N, 32, 32, 3) uint8 numpy -> (N, 3, 32, 32) uint8 torch, contiguous."""
    t = torch.as_tensor(np.asarray(arr))
    return t.permute(0, 3, 1, 2).contiguous()


def train_val_split(labels: torch.Tensor,
                    val_per_class: int = VAL_PER_CLASS,
                    seed: int = CIFAR_VAL_SPLIT_SEED,
                    num_classes: int = NUM_CLASSES) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the training indices into (train_pool, val), balanced per class.

    Deterministic and independent of the run seed, so the validation set is the
    same set of images in every experiment forever -- which is what makes
    validation accuracies comparable across runs, and what lets the test set be
    touched exactly once per run.

    Raises rather than shrinking if a class cannot supply `val_per_class`, for the
    same reason `stratified_subset` does.
    """
    g = torch.Generator().manual_seed(seed)
    pool_parts, val_parts = [], []
    for c in range(num_classes):
        cls_idx = (labels == c).nonzero(as_tuple=True)[0]
        if cls_idx.numel() <= val_per_class:
            raise ValueError(
                "class {} has only {} examples, cannot hold out {} for validation"
                .format(c, cls_idx.numel(), val_per_class))
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]
        val_parts.append(perm[:val_per_class])
        pool_parts.append(perm[val_per_class:])
    return torch.cat(pool_parts), torch.cat(val_parts)


def stratified_subset(labels: torch.Tensor, pool_idx: torch.Tensor,
                      n_per_class: int, seed: int,
                      num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Draw exactly `n_per_class` examples of each class from `pool_idx`.

    `pool_idx` must be the train pool from `train_val_split` -- never the full
    index range, or the validation images leak into training.

    RAISES when the pool cannot honor the request. It never truncates: a
    quietly-shrunk subset means the run's real data budget differs from the one
    logged in config.json, and nothing downstream can detect that.
    """
    g = torch.Generator().manual_seed(seed)
    pool_labels = labels[pool_idx]
    chosen = []
    for c in range(num_classes):
        cls_idx = pool_idx[pool_labels == c]
        if cls_idx.numel() < n_per_class:
            raise ValueError(
                "class {} has only {} examples in pool, need {}".format(
                    c, cls_idx.numel(), n_per_class))
        perm = torch.randperm(cls_idx.numel(), generator=g)
        chosen.append(cls_idx[perm[:n_per_class]])
    idx = torch.cat(chosen)
    return idx[torch.randperm(idx.numel(), generator=g)]


def subset_norm_stats(x_u8: torch.Tensor, idx: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std computed from the training subset ONLY.

    Returns two (3,) float tensors in [0, 1] scale. Computing these over the full
    dataset would leak a summary of the examples the model is not allowed to see
    into the low-data condition -- the first entry under "Common Mistakes" in
    data/skills.md, and the one v1 actually shipped.

    std is floored so a degenerate (constant) channel cannot divide by zero.
    """
    sub = x_u8[idx].float().div_(255.0)          # (n, 3, 32, 32)
    mean = sub.mean(dim=(0, 2, 3))
    std = sub.std(dim=(0, 2, 3)).clamp(min=1e-6)
    return mean, std


def make_batch(x_u8: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
               mean: torch.Tensor, std: torch.Tensor, device):
    """uint8 [B,3,32,32] -> normalized float [B,3,32,32] on `device`."""
    xb = x_u8[idx].to(device).float().div_(255.0)
    m = mean.to(device).view(1, -1, 1, 1)
    s = std.to(device).view(1, -1, 1, 1)
    return xb.sub_(m).div_(s), y[idx].to(device)


def assert_disjoint(*index_tensors: torch.Tensor) -> None:
    """Hard leakage check. Call this in tests and at the top of every run.

    Not defining it is a bug; defining it and never calling it is the same bug
    with extra steps -- which is what happened in v1.
    """
    seen = set()
    for t in index_tensors:
        s = set(t.tolist())
        overlap = seen & s
        assert not overlap, "Data leakage: {} shared indices".format(len(overlap))
        seen |= s


def preflight(labels: torch.Tensor, pool_idx: torch.Tensor, val_idx: torch.Tensor,
              shots_per_class, seeds, verbose: bool = True):
    """Build and validate EVERY subset before training anything.

    Three checks, all of which are cheap now and expensive later: the budget is
    honorable (stratified sampling is capped by the pool), the subset is exactly
    balanced, and it is disjoint from validation. Discovering any of these at the
    last condition of a multi-hour sweep wastes the sweep.
    """
    subsets = {}
    if verbose:
        print("preflight:")
    for shots in shots_per_class:
        for seed in seeds:
            idx = stratified_subset(labels, pool_idx, shots, seed)
            assert_disjoint(idx, val_idx)
            counts = torch.bincount(labels[idx], minlength=NUM_CLASSES)
            assert torch.all(counts == shots), \
                "unbalanced subset at {}: {}".format(shots, counts)
            subsets[(shots, seed)] = idx
        if verbose:
            print("  {:>5}/class -> {:>6} examples   balanced, disjoint from val"
                  "  [{} seeds]".format(shots, shots * NUM_CLASSES, len(seeds)))
    if verbose:
        print("  val split: {} examples, frozen (seed {}), {}/class"
              .format(val_idx.numel(), CIFAR_VAL_SPLIT_SEED, VAL_PER_CLASS))
        print("  pool ceiling: {}/class\n".format(MAX_SHOTS_PER_CLASS))
    return subsets
