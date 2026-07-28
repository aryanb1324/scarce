"""
Protocol v2 data pipeline.

Why this file exists alongside `mnist.py`: the v1 pipeline has three properties
that make a data-efficiency claim unsafe, all of them flagged in `data/skills.md`
as things to avoid. This file fixes them without touching `mnist.py`, so the v1
results stay reproducible.

What changed vs. v1:

1. **A fixed validation split.** v1 had none, so the test set was the only eval
   signal and got consumed a little more with every idea tried. Here 5,000
   examples are carved off MNIST-train with a frozen seed, excluded from every
   training subset, and used for all model selection. Test is touched once per
   run, at the end, on the best-val checkpoint.

2. **Stratified sampling.** v1 drew a uniform random permutation, so at 600
   examples the per-class counts varied enough across seeds to rival the effect
   being measured. Here every class contributes exactly `n_per_class`, which is
   also the standard N-way K-shot framing.

3. **Normalization computed from the training subset.** v1 hardcoded
   (0.1307, 0.3081) — full-MNIST-train statistics — and used them even in the 1%
   condition, leaking a summary of the 59,400 excluded examples into the run.
   `data/skills.md` lists this exact thing under "Common Mistakes."

Images are kept as uint8 and converted per batch, which holds the whole dataset
at ~47 MB instead of ~190 MB and is meaningfully faster than a DataLoader on CPU.
"""

import torch
from torchvision import datasets

VAL_SIZE = 5000
VAL_SPLIT_SEED = 1234  # FROZEN. Changing this invalidates comparability with
                       # every previously-recorded v2 result. Don't.


def load_mnist_tensors(data_dir: str = "./data_cache"):
    """Returns (x_train_u8, y_train, x_test_u8, y_test). No normalization applied."""
    tr = datasets.MNIST(data_dir, train=True, download=True)
    te = datasets.MNIST(data_dir, train=False, download=True)
    return (
        tr.data.clone(),      # uint8 [60000, 28, 28]
        tr.targets.clone(),   # int64 [60000]
        te.data.clone(),
        te.targets.clone(),
    )


def train_val_split(n_total: int, val_size: int = VAL_SIZE, seed: int = VAL_SPLIT_SEED):
    """
    Splits the MNIST training indices into (train_pool, val). Deterministic and
    independent of the run seed, so the validation set is the same set of images
    in every experiment forever — which is what makes val accuracies comparable
    across runs.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[val_size:].clone(), perm[:val_size].clone()


def stratified_subset(labels: torch.Tensor, pool_idx: torch.Tensor,
                      n_per_class: int, seed: int) -> torch.Tensor:
    """
    Draws exactly `n_per_class` examples of each digit from `pool_idx`.

    `pool_idx` must be the train pool from `train_val_split` — never the full
    index range, or the validation images leak into training.
    """
    g = torch.Generator().manual_seed(seed)
    pool_labels = labels[pool_idx]
    chosen = []
    for c in range(10):
        cls_idx = pool_idx[pool_labels == c]
        take = min(n_per_class, cls_idx.numel())
        if take < n_per_class:
            raise ValueError(f"class {c} has only {cls_idx.numel()} examples in pool, "
                             f"need {n_per_class}")
        perm = torch.randperm(cls_idx.numel(), generator=g)
        chosen.append(cls_idx[perm[:take]])
    idx = torch.cat(chosen)
    return idx[torch.randperm(idx.numel(), generator=g)]


def subset_norm_stats(x_u8: torch.Tensor, idx: torch.Tensor):
    """
    Mean/std computed from the training subset ONLY. Returns floats in [0,1] scale.
    """
    sub = x_u8[idx].float().div_(255.0)
    mean = sub.mean().item()
    std = sub.std().item()
    return mean, (std if std > 1e-6 else 1.0)


def make_batch(x_u8: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
               mean: float, std: float, device):
    """uint8 [B,28,28] -> normalized float [B,1,28,28] on `device`."""
    xb = x_u8[idx].to(device).float().div_(255.0).sub_(mean).div_(std).unsqueeze(1)
    return xb, y[idx].to(device)


def assert_disjoint(*index_tensors: torch.Tensor) -> None:
    """Hard leakage check. Call this in tests and at the top of every run."""
    seen = set()
    for t in index_tensors:
        s = set(t.tolist())
        overlap = seen & s
        assert not overlap, f"Data leakage: {len(overlap)} shared indices"
        seen |= s
