"""
CIFAR-100 under the protocol v2 data contract -- a harder-dataset test of the
project's central claim.

WHY THIS FILE EXISTS
--------------------
CIFAR-10 is "the gate on everything" (CLAUDE.md), but even CIFAR-10 has only ten
coarse classes a tiny CNN can separate from colour and shape at moderate accuracy.
CIFAR-100 keeps the same 32x32 natural images but splits them into 100 fine
classes with only 500 training images each, so at a few shots per class the task
is genuinely hard and a from-scratch tiny CNN sits far from saturation -- exactly
the regime where a data-efficiency prior, if it is real, should show, and where a
pretrained backbone should dominate. This module is the data half of that test.

It mirrors `data/cifar10.py` function for function -- same four guarantees, each
closing a hole `data/skills.md` records the v1 MNIST pipeline actually shipping:

  1. A FROZEN validation split, under its OWN seed constant. It does NOT reuse or
     import `cifar10.CIFAR_VAL_SPLIT_SEED` or `mnist_v2.VAL_SPLIT_SEED` -- those
     are frozen because every recorded result on those datasets depends on them,
     and a shared mutable constant is one careless edit from invalidating them.
     `CIFAR100_VAL_SPLIT_SEED` is a separate frozen constant with the same "do not
     touch" status.
  2. Stratified subset sampling that RAISES when the requested budget cannot be
     honored. A silently-truncated subset makes the run's real data budget differ
     from the config's claimed budget, invisible in the results CSV and fatal to
     the comparison.
  3. Normalization statistics computed FROM THE TRAINING SUBSET, per channel.
     CIFAR has three channels with genuinely different statistics, so the v1 MNIST
     mistake (hardcoded full-dataset mean/std reused in the low-data condition)
     would be a materially larger leak here than it was there.
  4. `assert_disjoint`, and it is CALLED -- from `preflight` below, from
     `experiments/train_cifar100_v1.py`, and from `tests/test_cifar100.py`. The
     MNIST v1 leakage checker existed but was never invoked from anywhere, so the
     highest-stakes directory in the project had zero coverage. A rule is not in
     force until something executes it.

THE 100-CLASS DIFFERENCE FROM cifar10.py, documented rather than silent:

  * CIFAR-100 train holds exactly 500 images per class (vs CIFAR-10's 5,000). The
    stratified holdout therefore CANNOT be 500/class. It is VAL_PER_CLASS = 100 ->
    10,000 validation images (the same size as the CIFAR test set), leaving exactly
    400/class = 40,000 images in the training pool. `MAX_SHOTS_PER_CLASS` is 400,
    the hard, seed-independent ceiling for `stratified_subset`.
  * Budgets are PER-CLASS shots, so a "5-shot" run is 5 x 100 = 500 labels. The
    per-class budget is small on purpose: the pool ceiling is 400/class, and the
    interesting regime is the scarce end.

Otherwise identical to cifar10.py: per-channel normalization, uint8 (N,3,32,32)
storage converted per batch, and the same data_cache root so all three datasets
share one gitignored cache.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torchvision import datasets

NUM_CLASSES = 100
VAL_PER_CLASS = 100                 # -> 10,000 validation images (== CIFAR test size)
CIFAR100_VAL_SPLIT_SEED = 20260729  # FROZEN. Its OWN constant, distinct from
                                    # cifar10.CIFAR_VAL_SPLIT_SEED (20260728).
                                    # Changing this invalidates comparability with
                                    # every recorded CIFAR-100 result. Don't.

# CIFAR-100 train holds exactly 500 images per class; the frozen val split removes
# VAL_PER_CLASS of each. This is the hard ceiling for `stratified_subset`.
MAX_SHOTS_PER_CLASS = 500 - VAL_PER_CLASS


def load_cifar100_tensors(data_dir: str = "./data_cache"):
    """Returns (x_train_u8, y_train, x_test_u8, y_test). No normalization applied.

    Images are (N, 3, 32, 32) uint8, channel-first, so `make_batch` does not have
    to permute on every batch. Same `data_dir` as CIFAR-10 and MNIST, so all three
    datasets live in one gitignored cache.
    """
    tr = datasets.CIFAR100(data_dir, train=True, download=True)
    te = datasets.CIFAR100(data_dir, train=False, download=True)
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
                    seed: int = CIFAR100_VAL_SPLIT_SEED,
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
    balanced across all 100 classes, and it is disjoint from validation.
    Discovering any of these at the last condition of a multi-hour sweep wastes the
    sweep.
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
              .format(val_idx.numel(), CIFAR100_VAL_SPLIT_SEED, VAL_PER_CLASS))
        print("  pool ceiling: {}/class\n".format(MAX_SHOTS_PER_CLASS))
    return subsets
