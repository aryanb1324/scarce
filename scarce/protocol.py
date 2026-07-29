"""
The measurement protocol. This is the part of the project that is actually novel.

Every rule here exists because breaking it already produced a wrong answer in
this repo's own history:

  * TRAIN TO CONVERGENCE, NEVER FIXED EPOCHS. Holding epochs fixed while the
    training set varies makes dataset size and optimization budget the same
    variable. It manufactured a 5.4-point deficit out of a 1.6-point one and
    inverted the sign of the conclusion. Any mechanism that changes convergence
    speed -- and sparsity does, it updates ~a fifth of the weights per step -- is
    misread as a data-efficiency effect. Each arm trains until validation stops
    improving and is scored at its best-validation checkpoint.

  * IDENTICAL BATCH SEQUENCE PER SEED, ACROSS ARMS. The batch sampler gets its
    own generator seeded independently of model init, so two arms at the same
    seed see the same batches in the same order however much RNG their
    initialization consumed. Without this the comparison is not paired and the
    per-seed delta carries batch-order noise it does not need to.

  * NORMALIZATION FROM THE TRAINING SET ONLY. Statistics computed over data the
    model is not supposed to have seen leak information and inflate apparent
    data efficiency -- the exact failure recorded in data/skills.md.

  * MEASURE THE COST, NOT JUST THE BENEFIT. A mechanism can do exactly what it
    claims and still lose. kWTA bought better codes with half the network and
    the two effects cancelled to "no difference"; accuracy alone read as "does
    nothing." Every run reports representation statistics alongside accuracy.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TrainConfig(NamedTuple):
    """Optimization settings, held identical across every arm.

    These mirror the recorded runs. Changing them is legitimate for your own
    data, but change them for ALL arms or the comparison stops being about the
    mechanism.
    """
    max_steps: int = 4000
    eval_every: int = 100
    patience: int = 12
    batch_size: int = 64
    lr: float = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_float_tensor(x: Any) -> torch.Tensor:
    """Accept numpy or torch, uint8 or float; return float32.

    uint8 is rescaled to [0, 1] the way image pipelines expect. Anything already
    floating point is passed through untouched -- rescaling it would silently
    change data the caller had already prepared.
    """
    t = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x)
    if t.dtype == torch.uint8:
        return t.float().div_(255.0)
    return t.float()


class Normalizer:
    """Per-channel standardization fitted on the training split ONLY."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Normalizer":
        if x.dim() == 4:                      # (N, C, H, W) -> per channel
            dims = (0, 2, 3)
            shape = (1, x.shape[1], 1, 1)
        else:                                 # (N, F) -> per feature
            dims = (0,)
            shape = (1, x.shape[1])
        mean = x.mean(dim=dims).reshape(shape)
        std = x.std(dim=dims).reshape(shape).clamp(min=1e-6)  # never divide by 0
        return cls(mean, std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class RunResult(NamedTuple):
    val_acc: float
    best_step: int
    steps_run: int
    hit_cap: bool
    active_units: float
    dead_frac: float
    seconds: float


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             device: torch.device, batch: int = 512) -> float:
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(device)
        pred = model(xb).argmax(dim=1).cpu()
        correct += (pred == y[i:i + batch]).sum().item()
    return correct / x.shape[0]


@torch.no_grad()
def representation_stats(model: nn.Module, x: torch.Tensor,
                         device: torch.device) -> Tuple[float, float]:
    """Mean active units and the permanently-dead fraction at the penultimate layer.

    The dead fraction is the cost side of the ledger. A hard selection rule is a
    ratchet: a unit that loses early gets no gradient, so its weights never move,
    so it keeps losing. Left unmeasured, that silently converts a data-efficiency
    comparison into a model-size comparison.
    """
    acts = []
    handle = model.penultimate.register_forward_hook(
        lambda _m, _i, out: acts.append((out.detach() != 0).float().cpu()))
    model.eval()
    n = min(x.shape[0], 1024)                 # a sample is enough for both stats
    for i in range(0, n, 256):
        model(x[i:i + 256].to(device))
    handle.remove()

    if not acts:
        return float("nan"), float("nan")
    m = torch.cat(acts)
    m = m.flatten(1) if m.dim() > 2 else m
    active = m.sum(dim=1).mean().item()
    dead = (m.sum(dim=0) == 0).float().mean().item()
    return active, dead


def train_to_convergence(model: nn.Module,
                         x_tr: torch.Tensor, y_tr: torch.Tensor,
                         x_val: torch.Tensor, y_val: torch.Tensor,
                         seed: int, cfg: TrainConfig,
                         device: Optional[torch.device] = None) -> RunResult:
    """Train until validation accuracy stops improving; score the best checkpoint.

    Returns validation accuracy at the best checkpoint. Note this is the number
    the search selects on, so it is optimistically biased for the winner -- see
    `search.fit`, which is explicit about that and asks for a test set to report
    an unbiased figure.
    """
    device = device or torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # Independent of model-init RNG, so arms at the same seed see identical batches.
    g = torch.Generator().manual_seed(10_000 + seed)
    n = x_tr.shape[0]
    bs = min(cfg.batch_size, n)

    best_val, best_step, best_state, stale, step = -1.0, -1, None, 0, 0
    t0 = time.time()

    while step < cfg.max_steps:
        model.train()
        step += 1
        pick = torch.randint(0, n, (bs,), generator=g)
        xb, yb = x_tr[pick].to(device), y_tr[pick].to(device)
        opt.zero_grad()
        criterion(model(xb), yb).backward()
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            va = evaluate(model, x_val, y_val, device)
            if va > best_val:
                best_val, best_step, stale = va, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

    secs = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    active, dead = representation_stats(model, x_val, device)
    return RunResult(best_val, best_step, step, step >= cfg.max_steps,
                     active, dead, secs)


def stratified_split(y: torch.Tensor, val_per_class: int,
                     seed: int = 1234) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split indices into (train, val), balanced per class, disjoint by construction.

    Raises rather than silently shrinking when a class cannot supply the request:
    a quietly-truncated split makes the real budget differ from the claimed one,
    which is invisible downstream and invalidates the comparison.
    """
    g = torch.Generator().manual_seed(seed)
    train_parts, val_parts = [], []
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        if idx.numel() <= val_per_class:
            raise ValueError(
                "class {} has only {} examples, cannot hold out {} for validation. "
                "Lower val_per_class or supply an explicit validation set.".format(
                    int(c), idx.numel(), val_per_class))
        val_parts.append(idx[:val_per_class])
        train_parts.append(idx[val_per_class:])
    return torch.cat(train_parts), torch.cat(val_parts)
