"""
Stage 1b: how sparse should the prior be, and how far down does it help?

============================== WHAT STAGE 1 FOUND =============================

The competition axis was the whole story, and it is now a real result.

  600 labels (5 seeds, paired)        delta vs dense    p        separation
    kwta_channel                          +1.43      0.0003 ***    0.181
    kwta_global   (v1/v2 module)          -1.58      0.0124 *      0.133
    kwta_channel_boost                    -8.61      0.0078 **     0.177
    kwta_global_boost                     -7.62      0.196         0.157
    dense                                     --        --         0.060

  3,000 labels: kwta_channel +0.19 (n.s.). The effect is confined to low data,
  which is exactly how a PRIOR should behave.

  Data multiplier: dense needs 1.78x the labels (about 1,070) to reach what
  kwta_channel reaches with 600.

Two beliefs from the previous round were wrong, and the per-layer breakdown is
what corrected them:

  1. "Half the network is dead" was misleading. The dead units are 94-99% in the
     FC layer (56-63% of its 128 units); conv1 is at 0-2.5% and conv2 at 3-13%.
     The aggregate looked catastrophic only because FC holds 128 of the 176
     counted units.
  2. Dead units were NOT the binding constraint. `dim` only changes the conv
     blocks -- for a 2-D input both rules are identical -- so the axis change
     cannot have touched the FC layer where nearly all the dead units live, yet
     it is what produced the +1.43. And kwta_channel actually kills MORE conv2
     channels than global (12.5% vs 5.6%) while winning.

  The pre-registered rule said "dead > 30% -> fix boosting before judging the
  idea." That rule was followed and it falsified its own premise: boosting DID
  reduce dead units (46.8% -> 43.0%) and DID raise separation (0.133 -> 0.157),
  and it destroyed accuracy anyway (-7.62 pts, with a 25.8-point spread across
  seeds). See experiments/diagnose_boost.py for the leading explanation.

So the mechanism that matters is real lateral inhibition among feature detectors
sharing a receptive field -- not sparsity per se, and not capacity recovery.

================================= THIS RUN ===================================

Two questions the +1.43 immediately raises, and one variable changes: k.

  1. HOW SPARSE? k = 0.2 was picked arbitrarily in v1 and never examined. If
     sparsity is a low-data prior, the best k should be a real optimum, and it may
     well move with the data budget.
  2. HOW FAR DOWN? The advantage was +1.43 at 600 labels and +0.19 at 3,000. If
     the prior story is right the advantage must GROW as labels get scarcer, so
     the grid extends down to 10 shots/class. This is the strongest available test
     of the interpretation, and it is cheap -- less data, same step budget.

Arms: dense, plus kwta_channel at k in {0.05, 0.1, 0.2, 0.4}. Boosting is off
everywhere; Stage 1 showed it is harmful as implemented and it is diagnosed
separately rather than swept here.

Per-layer units kept at each k (SmallCNN: conv1 C=16, conv2 C=32, fc 128):
    k=0.05 ->  1 / 2 / 6      k=0.20 ->  3 /  6 / 26
    k=0.10 ->  2 / 3 / 13     k=0.40 ->  6 / 13 / 51

========================= PRE-REGISTERED PREDICTION =========================

Written before the run; do not edit afterwards.

  * ADVANTAGE GROWS as labels fall (>= +1.4 pts at 600, larger at 300 and 100)
      -> the prior interpretation holds. The claim becomes "N x fewer labels" with
         the data multiplier as the headline, and the next step is CIFAR-10
         (PLAN.md Stage 4), where the effect is expected to shrink.
  * ADVANTAGE PEAKS AT 600 and falls at 100/300 labels
      -> it is not a data-scarcity prior but something specific to that budget.
         Treat +1.43 as a local effect and do not generalize it.
  * BEST k SHIFTS SPARSER as labels fall -> the prior's strength should scale with
      scarcity, which would be a clean, quotable relationship and argues for
      making k a function of the budget.
  * BEST k FLAT ACROSS BUDGETS -> k is an architecture constant, not a knob to
      tie to the data budget. Simpler, and worth knowing.
  * ADVANTAGE VANISHES AT EVERY k BELOW 600 LABELS -> the effect does not survive
      the regime it was supposed to be built for. Record it and go to PLAN.md
      Stage 2 (episodic memory), which consumes these codes rather than competing.

  Watch: at k=0.05 conv1 keeps 1 of 16 channels per position. If accuracy
  collapses there, the useful range is bounded below and the sweep has found its
  edge rather than failed.

Run with:
    QUICK=1 python -m experiments.train_stage1b     # smoke, ~2 min
    python -m experiments.train_stage1b             # full, 100 runs, ~70-90 min
"""

import csv
import json
import os
import time

import torch

import experiments.train_v2 as tv2
from architecture.model import build_dense_model, build_kwta_v2_model
from data.mnist_v2 import (
    load_mnist_tensors,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

QUICK = os.environ.get("QUICK") == "1"

# Same protocol constants as v2/Stage 1, set on train_v2 so the training code
# path is identical and the arms stay comparable across experiments.
tv2.MAX_STEPS = 4000 if not QUICK else 150
tv2.EVAL_EVERY = 100 if not QUICK else 50
tv2.PATIENCE = 12 if not QUICK else 2

SHOTS_PER_CLASS = [10, 30, 60, 300] if not QUICK else [10]
SEEDS = [0, 1, 2, 3, 4] if not QUICK else [0]
K_VALUES = [0.05, 0.10, 0.20, 0.40]

ARMS = [("dense", lambda: build_dense_model())] + [
    (f"kwta_ch_k{k:g}", (lambda kk: lambda: build_kwta_v2_model(dim="channel", boost=0.0, k=kk))(k))
    for k in K_VALUES
]

# Stage 1 reference (5 seeds, same protocol, k=0.2).
STAGE1_REF = {600: +1.43, 3000: +0.19}

FIELDNAMES = ["protocol", "git_commit", "arm", "k", "shots_per_class",
              "n_train_examples", "seed", "steps_run", "hit_step_cap",
              "best_val_acc", "best_val_step", "test_acc", "n_params",
              "dead_unit_frac", "active_units", "overlap_same", "overlap_diff",
              "separation", "train_seconds"]


def data_multiplier(dense_curve, target_acc, at_labels):
    """
    How many labels the dense baseline needs to reach `target_acc`, expressed as a
    multiple of `at_labels`.

    This is the headline metric PLAN.md section 3 asked for: "2.3x fewer labels
    for the same accuracy" is a claim a reader can hold, where "+1.43 points at
    600 labels" is not. Log-linear interpolation along the dense curve; returns
    None when the target sits outside the measured range, because extrapolating
    this number would be exactly the kind of overclaim the project should avoid.
    """
    pts = sorted(dense_curve.items())                      # [(labels, acc), ...]
    if len(pts) < 2:
        return None
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a1 <= a0:
            continue
        if a0 <= target_acc <= a1:
            import math
            f = (target_acc - a0) / (a1 - a0)
            needed = 10 ** (math.log10(n0) + f * (math.log10(n1) - math.log10(n0)))
            return needed / at_labels
    return None


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    commit = tv2.git_commit()
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_stage1b_k_sweep")
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol="stage1b", git_commit=commit, arms=[a for a, _ in ARMS],
               k_values=K_VALUES, dim="channel", boost=0.0,
               shots_per_class=SHOTS_PER_CLASS, seeds=SEEDS,
               max_steps=tv2.MAX_STEPS, eval_every=tv2.EVAL_EVERY,
               patience=tv2.PATIENCE, batch_size=tv2.BATCH_SIZE, lr=tv2.LR,
               selection="best-on-validation", device=str(tv2.DEVICE), quick=QUICK)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2), "\n")
    print(f"writing to {run_dir}")
    print(f"{len(ARMS) * len(SHOTS_PER_CLASS) * len(SEEDS)} runs planned\n")

    x_tr, y_tr, x_te, y_te = load_mnist_tensors()
    pool_idx, val_idx = train_val_split(x_tr.shape[0])
    test_idx = torch.arange(x_te.shape[0])

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    rows = []
    for shots in SHOTS_PER_CLASS:
        for seed in SEEDS:
            sub_idx = stratified_subset(y_tr, pool_idx, shots, seed)
            mean, std = subset_norm_stats(x_tr, sub_idx)
            dense_acc = None

            for arm, build_fn in ARMS:
                tv2.set_seed(seed)
                model = build_fn()
                r = tv2.train_one_run(model, x_tr, y_tr, sub_idx, val_idx,
                                      x_te, y_te, test_idx, mean, std, seed)
                if arm == "dense":
                    dense_acc = r["test_acc"]

                row = dict(
                    protocol="stage1b", git_commit=commit, arm=arm,
                    k=("" if arm == "dense" else arm.split("_k")[1]),
                    shots_per_class=shots, n_train_examples=sub_idx.numel(), seed=seed,
                    steps_run=r["steps_run"], hit_step_cap=int(r["hit_cap"]),
                    best_val_acc=round(r["best_val"], 4), best_val_step=r["best_step"],
                    test_acc=round(r["test_acc"], 4),
                    n_params=sum(p.numel() for p in model.parameters()),
                    dead_unit_frac=(None if r["dead"] != r["dead"] else round(r["dead"], 4)),
                    active_units=round(r["active"], 1),
                    overlap_same=round(r["ov_same"], 4),
                    overlap_diff=round(r["ov_diff"], 4),
                    separation=round(r["ov_same"] - r["ov_diff"], 4),
                    train_seconds=round(r["secs"], 1))
                rows.append(row)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                d = (r["test_acc"] - dense_acc) * 100
                dead_s = "   -" if r["dead"] != r["dead"] else f"{r['dead'] * 100:4.1f}%"
                print(f"  {shots:>4}/cls s={seed} {arm:<15} test={r['test_acc']:.4f} "
                      f"({d:+5.2f})  dead={dead_s}  sep={r['ov_same'] - r['ov_diff']:.3f}  "
                      f"act={r['active']:4.1f}  {r['secs']:5.1f}s")
            print()

    summarize(rows)
    print(f"\nSaved to {csv_path}")


def summarize(rows):
    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")

    budgets = sorted({r["n_train_examples"] for r in rows})
    dense_curve = {n: mean([r["test_acc"] for r in rows
                            if r["n_train_examples"] == n and r["arm"] == "dense"])
                   for n in budgets}

    print("\n" + "=" * 94)
    print("STAGE 1b SUMMARY -- kwta_channel k sweep, paired delta vs dense")
    print("=" * 94)

    best = {}
    for n in budgets:
        sub = [r for r in rows if r["n_train_examples"] == n]
        dense = {r["seed"]: r["test_acc"] for r in sub if r["arm"] == "dense"}
        ref = STAGE1_REF.get(n)
        print(f"\n{n} labels   dense = {dense_curve[n] * 100:.2f}%"
              + (f"   (Stage 1 k=0.2 reference: {ref:+.2f} pts)" if ref else ""))
        print(f"  {'arm':<16}{'test acc':>10}{'delta':>9}{'signs':>8}{'dead':>8}{'sep':>8}{'act':>7}{'x labels':>10}")
        for arm, _ in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            acc = mean([r["test_acc"] for r in a])
            deltas = [(r["test_acc"] - dense[r["seed"]]) * 100 for r in a if r["seed"] in dense]
            m = mean(deltas) if arm != "dense" else 0.0
            signs = "".join("+" if d > 0 else "-" for d in deltas) if arm != "dense" else ""
            mult = data_multiplier(dense_curve, acc, n) if arm != "dense" else None
            if arm != "dense" and m > best.get(n, (-99, None))[0]:
                best[n] = (m, arm, mult)
            print(f"  {arm:<16}{acc:>10.4f}{m:>+9.2f}{signs:>8}"
                  f"{mean([r['dead_unit_frac'] for r in a]) * 100 if arm != 'dense' else float('nan'):>7.1f}%"
                  f"{mean([r['separation'] for r in a]):>8.3f}"
                  f"{mean([r['active_units'] for r in a]):>7.1f}"
                  f"{(f'{mult:.2f}x' if mult else '--'):>10}")

    print("\n" + "-" * 94)
    print("BEST ARM BY BUDGET  (does the advantage grow as labels get scarcer?)")
    for n in budgets:
        if n in best:
            m, arm, mult = best[n]
            print(f"  {n:>6} labels: {arm:<16} {m:+.2f} pts"
                  + (f"   dense needs {mult:.2f}x the labels to match" if mult else ""))
    trend = [best[n][0] for n in budgets if n in best]
    if len(trend) >= 2:
        print(f"\n  advantage from smallest to largest budget: "
              f"{' -> '.join(f'{t:+.2f}' for t in trend)}")
        print("  DECREASING left-to-right = the prior interpretation holds (advantage grows")
        print("  as labels get scarcer). INCREASING = the +1.43 was specific to 600 labels.")
    ks = [best[n][1] for n in budgets if n in best]
    print(f"  best k by budget: {' -> '.join(ks)}")
    print("  shifting sparser as labels fall = tie k to the data budget; flat = k is a constant.")
    print("-" * 94)


if __name__ == "__main__":
    main()
