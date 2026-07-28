"""
Protocol v2 -- the experiment that decides what the v1 result meant.

================================ THE QUESTION ================================

v1 finding (2026-07-27, commit 26ba704, experiments/results.csv): kWTA lost to
the dense baseline at every data budget, and lost MORE as data shrank:

    60,000 labels  +0.01 pts     30,000  -0.12     6,000  -0.64
     3,000 labels  -2.29 pts        600  -5.38     (all 3 seeds agree in sign
                                                    at the two smallest budgets)

That is the opposite of the sparsity hypothesis. But v1 held EPOCHS fixed while
the dataset shrank, so the 600-label condition ran ~50 optimizer steps against
~4,690 at 60k. kWTA masks ~80% of units, so it updates roughly a fifth of the
weights per step and needs more steps to converge -- and the harness gave it
fewest steps exactly where it needed most. The v1 deficit correlates with
log(gradient steps) at r = 0.92.

So v1 cannot distinguish two hypotheses that demand opposite next moves:

    H_optim : kWTA converges slower and was scored before it finished.
    H_data  : kWTA genuinely generalizes worse from few examples.

=============================== THE MECHANISM ================================

The confound is removed by never scoring a model mid-convergence. Instead of a
fixed epoch count (v1) or even a fixed step count, each run trains until its
validation accuracy stops improving (patience-based early stopping, capped at
MAX_STEPS), and is scored at its best-validation checkpoint. Whatever each model
needs, it gets.

That turns v1's confound into a measurement: `best_val_step` is how long each
model took to peak, so the convergence-speed difference that contaminated v1 is
now a logged number rather than an unmodelled nuisance.

Three further protocol fixes (see data/mnist_v2.py), each closing a hole that
data/skills.md already warned about:
  * a frozen 5,000-example validation split, so test is touched once per run
  * stratified sampling, so class balance stops adding variance at small budgets
  * normalization computed from the training subset, not the full dataset

MODELS ARE UNCHANGED. This imports the same frozen baseline.py and the same
build_dense_model / build_kwta_model as v1. Only the protocol moved. Per root
skills.md ("make the smallest change that tests one mechanism at a time"), the
kWTA rewrite -- competition axis and duty-cycle boosting, PLAN.md section 2 --
is deliberately held back for the NEXT experiment, so this run attributes its
result to the protocol change alone.

=============================== DIAGNOSTICS =================================

Accuracy alone cannot say whether a mechanism failed or the hypothesis linking
the mechanism to accuracy failed (root skills.md). Three direct measurements:

  1. `best_val_step`   -- convergence speed. Directly tests H_optim.
  2. `dead_unit_frac`  -- share of units that never win kWTA across a full val
                          pass. Such units receive no gradient and are
                          effectively deleted, shrinking the network. Tests the
                          capacity-collapse prediction in PLAN.md 2.2.
  3. `overlap_same` /  -- mean pairwise Jaccard overlap of the penultimate
     `overlap_diff`       layer's active-unit sets, split by same-class vs
                          different-class pairs. This is kWTA's ACTUAL claim
                          (pattern separation), measured rather than inferred.

========================= PRE-REGISTERED PREDICTION =========================

Written before the run. Do not edit after seeing results (experiments/skills.md).

  * |delta| < 1.0 pt at 600 labels           -> H_optim. v1 measured optimization
                                                budget, not data efficiency.
                                                Sparsity is neutral so far;
                                                proceed to the axis/boosting
                                                rewrite as PLAN.md Stage 1.
  * delta < -3.0 pts at 600, all seeds agree -> H_data. Sparsity genuinely hurts
                                                at low data; pattern separation
                                                is falsified in this form and
                                                episodic memory (Stage 2) becomes
                                                the priority.
  * anything between                         -> underpowered. Add seeds before
                                                concluding.

  Independent of the above:
  * dead_unit_frac > 0.30       -> capacity collapse is implicated. Fix the
                                   missing duty-cycle boosting BEFORE judging
                                   the idea; the current result is about a
                                   shrinking network, not about sparsity.
  * kWTA best_val_step > 2x dense's -> confirms convergence speed was the v1
                                   mechanism, whatever the accuracy shows.
  * overlap_same not below dense's -> the module is not separating patterns at
                                   all, so the implementation is wrong before
                                   the hypothesis is even testable.

Run with:
    QUICK=1 python -m experiments.train_v2     # smoke test, ~1-2 min
    python -m experiments.train_v2             # full run, expect 60-90 min on CPU

Conditions run smallest-budget-first and results append as they complete, so the
600- and 3,000-label answers -- which ARE the experiment; the large budgets are
controls -- arrive within the first few minutes.
"""

import copy
import csv
import json
import os
import random
import subprocess
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

# ----------------------------- protocol v2 config ----------------------------
# Budgets mirror v1's totals (600 / 3k / 6k / 30k) so the protocols are directly
# comparable. The top point is 4,800/class not 6,000: MNIST's rarest digit is 5
# (5,421 examples), the frozen val split removes ~450, and stratified sampling is
# capped by the rarest class. Ascending order = cheapest, most informative first.
SHOTS_PER_CLASS = [60, 300, 600, 3000, 4800] if not QUICK else [60, 300]
SEEDS = [0, 1, 2] if not QUICK else [0]

MAX_STEPS = 4000 if not QUICK else 150   # cap; early stopping usually ends sooner
EVAL_EVERY = 100 if not QUICK else 50
PATIENCE = 12 if not QUICK else 2        # evals without val improvement -> stop
BATCH_SIZE = 64                          # same as v1
LR = 1e-3                                # same as v1
OVERLAP_N_PER_CLASS = 60                 # val examples per class for the overlap metric
# -----------------------------------------------------------------------------

# v1 paired deltas (pts), keyed by total labeled examples, for the final summary.
V1_DELTA = {600: -5.38, 3000: -2.29, 6000: -0.64, 30000: -0.12, 60000: +0.01}

FIELDNAMES = [
    "protocol", "git_commit", "shots_per_class", "n_train_examples", "seed",
    "model", "steps_run", "hit_step_cap", "best_val_acc", "best_val_step",
    "test_acc", "n_params", "dead_unit_frac", "active_units",
    "overlap_same", "overlap_diff", "train_seconds",
]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else "unknown"
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class WinCounter:
    """
    Counts how often each unit survives kWTA, to detect capacity collapse.

    A unit that never wins across a full evaluation pass receives no gradient, so
    its weights never move, so it keeps losing. Effective capacity shrinks over
    training and the network ends up smaller than the baseline -- which silently
    converts a data-efficiency experiment into a model-size experiment, the exact
    confound architecture/skills.md warns about.
    """

    def __init__(self, model: nn.Module):
        self.counts, self.handles = {}, []
        for name, mod in model.named_modules():
            if isinstance(mod, KWinnersTakeAll):
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_m, _i, out):
            o = out.detach()
            # Per CHANNEL for conv layers -- weights are shared across space, so
            # the channel is the unit whose capacity can actually be lost.
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


@torch.no_grad()
def code_overlap(model, x_u8, y, idx, mean, std):
    """
    Mean pairwise Jaccard overlap of penultimate-layer active-unit sets, split by
    same-class vs different-class pairs.

    This is the mechanism's own claim, measured directly. If sparsity works as
    theorized, same-class overlap should stay high (a class shares a code) while
    different-class overlap falls (codes are disjoint), so different examples stop
    overwriting each other's weights.

    Hooking `classifier[2]` works for both models: it is the ReLU in the dense net
    and the Sequential(ReLU, kWTA) in the sparse one -- i.e. the representation
    the final linear layer actually reads.
    """
    acts = []
    h = model.classifier[2].register_forward_hook(
        lambda _m, _i, o: acts.append((o.detach() != 0).float().cpu()))
    model.eval()
    for i in range(0, idx.numel(), 256):
        b = idx[i:i + 256]
        xb, _ = make_batch(x_u8, y, b, mean, std, DEVICE)
        model(xb)
    h.remove()

    M = torch.cat(acts)                     # [n, 128] binary
    labs = y[idx].cpu()
    sizes = M.sum(1)
    inter = M @ M.t()
    union = (sizes[:, None] + sizes[None, :] - inter).clamp(min=1.0)
    J = inter / union

    n = M.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    same = (labs[:, None] == labs[None, :]) & off
    diff = (labs[:, None] != labs[None, :])
    return J[same].mean().item(), J[diff].mean().item(), sizes.mean().item()


def train_one_run(model, x_u8, y_tr, sub_idx, val_idx, x_te, y_te, test_idx,
                  mean, std, seed):
    """
    Trains until validation accuracy stops improving (PATIENCE evals without a
    new best), capped at MAX_STEPS, and scores the best-validation checkpoint.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Dedicated batch-sampling generator, so dense and kWTA see the IDENTICAL
    # batch sequence for a given seed regardless of how much RNG init consumed.
    g = torch.Generator().manual_seed(10_000 + seed)
    n = sub_idx.numel()
    bs = min(BATCH_SIZE, n)

    best_val, best_step, best_state, stale = -1.0, -1, None, 0
    step = 0
    t0 = time.time()

    while step < MAX_STEPS:
        model.train()
        step += 1
        pick = sub_idx[torch.randint(0, n, (bs,), generator=g)]
        xb, yb = make_batch(x_u8, y_tr, pick, mean, std, DEVICE)
        opt.zero_grad()
        criterion(model(xb), yb).backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            va = evaluate(model, x_u8, y_tr, val_idx, mean, std)
            if va > best_val:
                best_val, best_step, stale = va, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

    secs = time.time() - t0
    model.load_state_dict(best_state)

    counter = WinCounter(model)
    evaluate(model, x_u8, y_tr, val_idx, mean, std)
    dead = counter.dead_fraction()
    counter.remove()

    ov_idx = stratified_subset(y_tr, val_idx, OVERLAP_N_PER_CLASS, seed=999)
    ov_same, ov_diff, active = code_overlap(model, x_u8, y_tr, ov_idx, mean, std)

    test_acc = evaluate(model, x_te, y_te, test_idx, mean, std)
    return dict(test_acc=test_acc, best_val=best_val, best_step=best_step,
                steps_run=step, hit_cap=(step >= MAX_STEPS), dead=dead,
                ov_same=ov_same, ov_diff=ov_diff, active=active, secs=secs)


def preflight(y_tr, pool_idx, val_idx):
    """
    Build and check EVERY subset before training anything.

    Stratified sampling is capped by the rarest class, and a budget that cannot be
    honored raises. Discovering that at the last condition would waste an hour, so
    fail at t=0 instead. Also runs the leakage assertion on every subset.
    """
    print("preflight:")
    subsets = {}
    for shots in SHOTS_PER_CLASS:
        for seed in SEEDS:
            idx = stratified_subset(y_tr, pool_idx, shots, seed)
            assert_disjoint(idx, val_idx)
            counts = torch.bincount(y_tr[idx], minlength=10)
            assert torch.all(counts == shots), f"unbalanced subset at {shots}: {counts}"
            subsets[(shots, seed)] = idx
        print(f"  {shots:>5}/class -> {subsets[(shots, SEEDS[0])].numel():>6} examples"
              f"   balanced, disjoint from val  [{len(SEEDS)} seeds]")
    print(f"  val split: {val_idx.numel()} examples, frozen (seed 1234)\n")
    return subsets


def summarize(rows):
    """Paired per-seed deltas, versus v1's, with the pre-registered verdict."""
    by = {}
    for r in rows:
        by.setdefault((int(r["n_train_examples"]), int(r["seed"])), {})[r["model"]] = r

    print("\n" + "=" * 78)
    print("PROTOCOL v2 SUMMARY -- paired delta = kWTA - dense, percentage points")
    print("=" * 78)
    print(f"{'labels':>7} {'v2 delta':>12} {'seeds':>7} {'v1 delta':>10} "
          f"{'steps d/k':>12} {'dead':>7} {'overlap s/d':>14}")

    verdict_delta = None
    for n in sorted({int(r['n_train_examples']) for r in rows}):
        ds = [by[(n, s)] for s in sorted({int(r['seed']) for r in rows})
              if (n, s) in by and len(by[(n, s)]) == 2]
        if not ds:
            continue
        deltas = [(d["kwta_sparse"]["test_acc"] - d["dense_baseline"]["test_acc"]) * 100
                  for d in ds]
        m = sum(deltas) / len(deltas)
        signs = "".join("+" if x > 0 else "-" for x in deltas)
        sd = sum(d["dense_baseline"]["best_val_step"] for d in ds) / len(ds)
        sk = sum(d["kwta_sparse"]["best_val_step"] for d in ds) / len(ds)
        dead = sum(d["kwta_sparse"]["dead_unit_frac"] for d in ds) / len(ds)
        os_ = sum(d["kwta_sparse"]["overlap_same"] for d in ds) / len(ds)
        od = sum(d["kwta_sparse"]["overlap_diff"] for d in ds) / len(ds)
        v1 = V1_DELTA.get(n)  # 48,000 has no v1 counterpart (v1's top point was 60,000)
        v1s = f"{v1:+.2f}" if v1 is not None else "--"
        print(f"{n:>7} {m:>+11.2f} {signs:>7} {v1s:>10} "
              f"{sd:>5.0f}/{sk:<6.0f} {dead*100:>6.1f}% {os_:>6.2f}/{od:<6.2f}")
        if n == 600:
            verdict_delta = (m, signs, dead, sk, sd)

    if verdict_delta is None:
        print("\n(600-label condition absent -- no verdict)")
        return
    m, signs, dead, sk, sd = verdict_delta
    print("\n" + "-" * 78)
    print(f"PRE-REGISTERED VERDICT at 600 labels (v1 was -5.38):  delta = {m:+.2f} pts")
    if abs(m) < 1.0:
        print("  -> H_optim. v1 measured optimization budget, not data efficiency.")
        print("     Sparsity is NEUTRAL so far. Next: PLAN.md Stage 1 (competition")
        print("     axis + duty-cycle boosting), which is where an effect could come from.")
    elif m < -3.0 and signs.count("-") == len(signs):
        print("  -> H_data. Sparsity genuinely hurts at low data; pattern separation")
        print("     is falsified in this form. Next: PLAN.md Stage 2 (episodic memory).")
    else:
        print("  -> AMBIGUOUS. Add seeds before concluding anything.")
    if dead == dead and dead > 0.30:
        print(f"  !! dead units {dead*100:.0f}% > 30%: capacity collapse is implicated.")
        print("     Fix duty-cycle boosting BEFORE judging the idea -- the result above")
        print("     is about a shrinking network, not about sparsity.")
    if sd > 0 and sk > 2 * sd:
        print(f"  !! kWTA peaked at {sk:.0f} steps vs dense {sd:.0f} (>2x): confirms")
        print("     convergence speed was the v1 mechanism.")
    print("-" * 78)


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    commit = git_commit()
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_protocol_v2")
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol="v2", git_commit=commit, shots_per_class=SHOTS_PER_CLASS,
               seeds=SEEDS, max_steps=MAX_STEPS, eval_every=EVAL_EVERY,
               patience=PATIENCE, batch_size=BATCH_SIZE, lr=LR,
               selection="best-on-validation", device=str(DEVICE), quick=QUICK)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2), "\n")
    print(f"writing to {run_dir}\n")

    x_tr, y_tr, x_te, y_te = load_mnist_tensors()
    pool_idx, val_idx = train_val_split(x_tr.shape[0])
    test_idx = torch.arange(x_te.shape[0])
    subsets = preflight(y_tr, pool_idx, val_idx)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    rows = []
    for shots in SHOTS_PER_CLASS:
        for seed in SEEDS:
            sub_idx = subsets[(shots, seed)]
            mean, std = subset_norm_stats(x_tr, sub_idx)
            accs = {}

            for model_name, build_fn in [("dense_baseline", build_dense_model),
                                         ("kwta_sparse", build_kwta_model)]:
                set_seed(seed)  # identical init for both models
                model = build_fn()
                r = train_one_run(model, x_tr, y_tr, sub_idx, val_idx,
                                  x_te, y_te, test_idx, mean, std, seed)
                accs[model_name] = r["test_acc"]

                row = dict(
                    protocol="v2", git_commit=commit, shots_per_class=shots,
                    n_train_examples=sub_idx.numel(), seed=seed, model=model_name,
                    steps_run=r["steps_run"], hit_step_cap=int(r["hit_cap"]),
                    best_val_acc=round(r["best_val"], 4), best_val_step=r["best_step"],
                    test_acc=round(r["test_acc"], 4),
                    n_params=sum(p.numel() for p in model.parameters()),
                    dead_unit_frac=(None if r["dead"] != r["dead"] else round(r["dead"], 4)),
                    active_units=round(r["active"], 1),
                    overlap_same=round(r["ov_same"], 4),
                    overlap_diff=round(r["ov_diff"], 4),
                    train_seconds=round(r["secs"], 1))
                rows.append(row)
                with open(csv_path, "a", newline="") as f:  # append as we go
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                dead_s = "   -" if r["dead"] != r["dead"] else f"{r['dead']*100:4.1f}%"
                cap = " CAP" if r["hit_cap"] else ""
                print(f"  {shots:>5}/cls seed={seed} {model_name:<14} "
                      f"test={r['test_acc']:.4f}  val={r['best_val']:.4f}"
                      f"@{r['best_step']:<4}/{r['steps_run']:<4}{cap}  "
                      f"dead={dead_s}  act={r['active']:5.1f}  "
                      f"ovl={r['ov_same']:.2f}/{r['ov_diff']:.2f}  {r['secs']:5.1f}s")

            d = (accs["kwta_sparse"] - accs["dense_baseline"]) * 100
            print(f"  {'':>16}--> paired delta {d:+.2f} pts\n")

    summarize(rows)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
