"""
Protocol v2: the experiment that decides what the v1 result meant.

v1 finding (2026-07-27): kWTA lost to the dense baseline at every data budget,
and the deficit grew monotonically as data shrank -- +0.01 pts at 60k labels,
-2.29 at 3k, -5.38 at 600. All three seeds agreed in sign at the two smallest
budgets. That is the OPPOSITE of the sparsity hypothesis.

But v1 held EPOCHS fixed while the dataset shrank, so the 600-label condition ran
~50 optimizer steps against ~4,690 at 60k. kWTA updates roughly a fifth of the
units per step, so it needs more steps to reach the same place -- and the harness
gave it fewest steps exactly where it needed most. The v1 deficit correlates with
log(gradient steps) at r = 0.92, which is what you would expect if the whole
effect were an optimization artifact.

So v1 cannot distinguish:
    (H_data)  sparsity genuinely hurts generalization from few examples
    (H_optim) sparsity converges slower and was scored before it finished

This script separates them by removing the confound:

  * EVERY condition gets the SAME number of gradient steps (STEPS), regardless of
    how many examples it has. Small budgets simply revisit their examples more.
  * Model selection is best-on-validation, not last-epoch, so neither model can be
    scored mid-convergence. This is the single change that makes H_optim testable.
  * Stratified sampling, subset-computed normalization, a frozen val split
    (see data/mnist_v2.py).
  * Dead-unit diagnostics on every kWTA layer, which independently test the
    capacity-collapse mechanism predicted in PLAN.md section 2.2.

PREDICTION, registered before running (do not edit after seeing results):
  If H_optim is right, the deficit collapses toward zero at all budgets, and dead-
  unit fractions are low. If H_data is right, the deficit at 600 labels survives
  at something like its v1 magnitude. If dead-unit fractions are high (>30%), the
  cause is capacity collapse from missing duty-cycle boosting, and the fix is the
  boosting rewrite (PLAN.md section 2.2) rather than abandoning sparsity.

The model code is UNCHANGED -- this imports the same frozen `baseline.py` and the
same `build_dense_model` / `build_kwta_model`. Only the protocol moved.

Run with:
    python -m experiments.train_v2
    QUICK=1 python -m experiments.train_v2     # smoke test, ~1 minute
"""

import copy
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from architecture.model import build_dense_model, build_kwta_model
from architecture.modules.kwta import KWinnersTakeAll
from data.mnist_v2 import (
    assert_disjoint,
    load_mnist_tensors,
    make_batch,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUICK = os.environ.get("QUICK") == "1"

# --- protocol v2 config -----------------------------------------------------
# SHOTS_PER_CLASS mirrors v1's total budgets (600 / 3k / 6k / 30k) so the two
# protocols are directly comparable. The top point is 4,800/class rather than
# 6,000: MNIST's rarest digit is 5 (5,421 examples), the frozen val split removes
# ~450 of them, and stratified sampling is capped by the rarest class.
SHOTS_PER_CLASS = [60, 300, 600, 3000, 4800] if not QUICK else [60, 300]
SEEDS = [0, 1, 2] if not QUICK else [0]
STEPS = 2000 if not QUICK else 100      # IDENTICAL for every data budget
EVAL_EVERY = 100 if not QUICK else 50
BATCH_SIZE = 64                          # same as v1
LR = 1e-3                                # same as v1
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "protocol", "shots_per_class", "n_train_examples", "seed", "model",
    "steps", "best_val_acc", "best_val_step", "test_acc",
    "n_params", "dead_unit_frac", "train_seconds",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class WinCounter:
    """
    Counts how often each unit survives kWTA, to measure capacity collapse.

    A unit that never wins across an entire evaluation pass receives no gradient
    and is effectively deleted from the network. If that fraction is large, kWTA
    is not producing a sparse distributed code -- it is producing a smaller dense
    one, which would confound "data efficiency" with "model size" exactly as
    architecture/skills.md warns.
    """

    def __init__(self, model: nn.Module):
        self.counts = {}
        self.handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, KWinnersTakeAll):
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_mod, _inp, out):
            o = out.detach()
            # Count per CHANNEL for conv layers (weights are shared across space,
            # so a channel is the unit whose capacity can actually be lost) and
            # per unit for linear layers.
            c = (o != 0).sum(dim=(0, 2, 3)) if o.dim() == 4 else (o != 0).sum(dim=0)
            self.counts[name] = c if name not in self.counts else self.counts[name] + c
        return fn

    def dead_fraction(self) -> float:
        if not self.counts:
            return float("nan")
        dead = sum((v == 0).sum().item() for v in self.counts.values())
        total = sum(v.numel() for v in self.counts.values())
        return dead / total

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def evaluate(model, x_u8, y, idx, mean, std) -> float:
    model.eval()
    correct = 0
    for i in range(0, idx.numel(), 512):
        b = idx[i:i + 512]
        xb, yb = make_batch(x_u8, y, b, mean, std, DEVICE)
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / idx.numel()


def train_one_run(model, x_u8, y_tr, sub_idx, val_idx, x_te, y_te, test_idx,
                  mean, std, seed):
    """
    Trains for exactly STEPS optimizer steps, selects the best-validation
    checkpoint, and returns test accuracy at that checkpoint.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Dedicated batch-sampling generator: both models get the IDENTICAL batch
    # sequence for a given seed, independent of how much RNG model init consumed.
    g = torch.Generator().manual_seed(10_000 + seed)
    n = sub_idx.numel()

    best_val, best_step, best_state = -1.0, -1, None
    t0 = time.time()

    for step in range(1, STEPS + 1):
        model.train()
        pick = sub_idx[torch.randint(0, n, (min(BATCH_SIZE, n),), generator=g)]
        xb, yb = make_batch(x_u8, y_tr, pick, mean, std, DEVICE)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            va = evaluate(model, x_u8, y_tr, val_idx, mean, std)
            if va > best_val:
                best_val, best_step = va, step
                best_state = copy.deepcopy(model.state_dict())

    train_seconds = time.time() - t0
    model.load_state_dict(best_state)

    # Dead-unit measurement on the validation set, at the selected checkpoint.
    counter = WinCounter(model)
    evaluate(model, x_u8, y_tr, val_idx, mean, std)
    dead = counter.dead_fraction()
    counter.remove()

    test_acc = evaluate(model, x_te, y_te, test_idx, mean, std)
    return test_acc, best_val, best_step, dead, train_seconds


def main():
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_protocol_v2")
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol="v2", shots_per_class=SHOTS_PER_CLASS, seeds=SEEDS,
               steps=STEPS, eval_every=EVAL_EVERY, batch_size=BATCH_SIZE, lr=LR,
               device=str(DEVICE), quick=QUICK)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2))
    print(f"\nwriting to {run_dir}\n")

    x_tr, y_tr, x_te, y_te = load_mnist_tensors()
    pool_idx, val_idx = train_val_split(x_tr.shape[0])
    test_idx = torch.arange(x_te.shape[0])

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    for shots in SHOTS_PER_CLASS:
        for seed in SEEDS:
            sub_idx = stratified_subset(y_tr, pool_idx, shots, seed)
            assert_disjoint(sub_idx, val_idx)  # hard leakage check, every run
            mean, std = subset_norm_stats(x_tr, sub_idx)

            accs = {}
            for model_name, build_fn in [("dense_baseline", build_dense_model),
                                         ("kwta_sparse", build_kwta_model)]:
                set_seed(seed)  # identical init for both models
                model = build_fn()
                n_params = sum(p.numel() for p in model.parameters())

                acc, bval, bstep, dead, secs = train_one_run(
                    model, x_tr, y_tr, sub_idx, val_idx, x_te, y_te, test_idx,
                    mean, std, seed)
                accs[model_name] = acc

                row = dict(protocol="v2", shots_per_class=shots,
                           n_train_examples=sub_idx.numel(), seed=seed,
                           model=model_name, steps=STEPS, best_val_acc=round(bval, 4),
                           best_val_step=bstep, test_acc=round(acc, 4),
                           n_params=n_params,
                           dead_unit_frac=("" if dead != dead else round(dead, 4)),
                           train_seconds=round(secs, 1))
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                dead_s = "  -" if dead != dead else f"{dead*100:5.1f}%"
                print(f"shots={shots:<5} seed={seed} {model_name:<14} "
                      f"test={acc:.4f}  val={bval:.4f}@{bstep:<5} "
                      f"dead={dead_s}  {secs:5.1f}s")

            d = (accs["kwta_sparse"] - accs["dense_baseline"]) * 100
            print(f"{'':>28}paired delta = {d:+.2f} pts\n")

    print(f"Saved to {csv_path}")
    print("\nCompare against v1 (fixed epochs): -5.38 pts @600, -2.29 @3000, "
          "-0.64 @6000, -0.12 @30000, +0.01 @60000.")
    print("If the deltas above are near zero, v1 measured optimization budget, "
          "not data efficiency.")


if __name__ == "__main__":
    main()
