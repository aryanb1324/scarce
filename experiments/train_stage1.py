"""
Stage 1: why is half the kWTA network dead, and does fixing it buy accuracy?

================================ WHERE WE ARE ================================

Protocol v2 (commit 9e7874f) settled the v1 question and raised a sharper one.

Settled: 70-85% of v1's kWTA deficit was an optimization artifact. Training to
convergence instead of a fixed 5 epochs moved the paired delta at 600 labels from
-5.38 pts to -1.58 (n.s., t=-2.38, p=0.14) and at 3,000 labels from -2.29 to
-0.35. v1 was measuring optimization budget, as suspected.

Raised, and this is the interesting part -- v2's two diagnostics disagree:

  THE MECHANISM WORKS. Same-class vs different-class code separation, kWTA
  relative to dense:  2.24x at 600 labels, 1.21x at 3,000, 1.07x at 6,000,
  0.99x at 30,000, 0.88x at 48,000. Monotonically largest exactly where data is
  scarcest -- the hypothesis's own mechanistic prediction, confirmed.

  THE NETWORK IS HALF DEAD. 44-49% of units never won kWTA across a full
  validation pass, at every budget, stable rather than transient.

So kWTA is buying genuinely better-separated codes with roughly half its
capacity, and the two effects cancel. That is why accuracy sits at "no
difference" while the mechanism it depends on is measurably working.

Independent corroboration that sparsity itself is not the problem: the DENSE
baseline drifts toward sparse codes on its own as data grows (77.3 -> 45.9 active
units of 128 from 600 to 48,000 labels) and its separation rises 0.057 -> 0.260.
Given enough data, the dense net discovers what kWTA imposes. Sparsity looks like
a useful PRIOR for the low-data regime -- and a prior is only worth having if it
doesn't cost half the network to hold.

================================= THE DESIGN =================================

Two candidate causes of the dead units, crossed 2x2 so the effect is
attributable, plus the dense control. Five arms:

                       boost=0            boost=1
    dim=global      kwta_global       kwta_global_boost      (global = v1/v2 anchor)
    dim=channel     kwta_channel      kwta_channel_boost
    plus            dense

  * AXIS. v1 takes one global top-k over flattened C x H x W, so a channel can
    lose everywhere and die. Channel-wise competition keeps exactly k channels at
    EVERY spatial position, so a weak channel still wins where it is locally
    best. The axis may fix the dead units by itself.
  * BOOSTING. Duty-cycle boosting inflates the ranking score of chronic losers so
    they re-enter the competition.

`kwta_global` (boost=0) is bit-identical to the v1 module -- asserted in
tests/test_kwta_v2.py -- so it re-derives the v2 numbers under this run's seeds
and acts as the anchor.

Budgets are 600 and 3,000 labels only: v2 showed everything above 6,000 is a flat
zero, so those points cost time and carry no information. FIVE seeds, not three,
because v2's -1.58 at 600 labels was underpowered (p=0.14) and the residual is
exactly what this experiment has to resolve.

Training code is imported from train_v2 rather than reimplemented, so the code
path is identical and the arms stay comparable to the v2 results.

========================= PRE-REGISTERED PREDICTION =========================

Written before the run; do not edit afterwards (experiments/skills.md).

  ATTRIBUTION -- which fix recovers the dead capacity:
  * dead drops in kwta_channel (boost=0)  -> the competition AXIS was the cause.
  * dead drops only with boost=1          -> the DUTY CYCLE was the cause.
  * both needed                           -> they are independent failures.
  * neither drops below ~20%              -> the cause is something else and the
                                             per-layer breakdown says where.

  PAYOFF -- does recovered capacity become accuracy, at 600 labels:
  * best arm >= +1.0 pt over dense, seeds agreeing -> sparsity is a real
      low-data prior once implemented properly. Sweep k next, then take the
      winning config to CIFAR-10 (PLAN.md Stage 4).
  * best arm within +-1.0 pt of dense, dead now low, separation still >= v2's ->
      the codes were never the bottleneck. Sparsity is neutral on MNIST; go to
      PLAN.md Stage 2 (episodic memory), which is where the large low-data win is
      expected and which CONSUMES these sparse codes rather than competing with
      them.
  * best arm still clearly below dense with dead units low -> forced sparsity
      genuinely costs accuracy here. Record it as a negative result and move to
      Stage 2.

  Watch also: separation must not collapse when boosting is on. Boosting spreads
  activity across units, which could undo the very pattern separation that makes
  kWTA worth having. If separation falls to dense levels, boosting has traded
  away the mechanism to save the capacity, and k needs re-tuning rather than
  boosting turned up.

Run with:
    QUICK=1 python -m experiments.train_stage1     # smoke, ~2 min
    python -m experiments.train_stage1             # full, ~50-70 min
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

# Protocol constants are set ON train_v2 so train_one_run uses the identical code
# path as the v2 run. Same values as v2 -- only the arms and seed count change.
tv2.MAX_STEPS = 4000 if not QUICK else 150
tv2.EVAL_EVERY = 100 if not QUICK else 50
tv2.PATIENCE = 12 if not QUICK else 2

SHOTS_PER_CLASS = [60, 300] if not QUICK else [60]
SEEDS = [0, 1, 2, 3, 4] if not QUICK else [0]

ARMS = [
    ("dense", lambda: build_dense_model()),
    ("kwta_global", lambda: build_kwta_v2_model(dim="global", boost=0.0)),
    ("kwta_global_boost", lambda: build_kwta_v2_model(dim="global", boost=1.0)),
    ("kwta_channel", lambda: build_kwta_v2_model(dim="channel", boost=0.0)),
    ("kwta_channel_boost", lambda: build_kwta_v2_model(dim="channel", boost=1.0)),
]

# v2 reference (3 seeds, same protocol) for the anchor check.
V2_REF = {600: {"delta": -1.58, "dead": 0.477, "sep": 0.127, "sep_dense": 0.057},
          3000: {"delta": -0.35, "dead": 0.489, "sep": 0.137, "sep_dense": 0.113}}

FIELDNAMES = ["protocol", "git_commit", "arm", "dim", "boost", "shots_per_class",
              "n_train_examples", "seed", "steps_run", "hit_step_cap",
              "best_val_acc", "best_val_step", "test_acc", "n_params",
              "dead_unit_frac", "dead_by_layer", "active_units",
              "overlap_same", "overlap_diff", "separation", "train_seconds"]


# Module paths of the three kWTA sites inside SmallCNN, for readable output.
LAYER_NAMES = {"features.1.1": "conv1", "features.4.1": "conv2", "classifier.2.1": "fc"}


def dead_by_layer(model, x_u8, y, idx, mean, std):
    """
    Per-layer dead-unit fractions. v2 logged only the aggregate (44-49%), which
    hides which layer is dying -- and that matters a lot: a dead conv1 channel
    removes a feature detector the whole network depends on, while a dead FC unit
    costs only a slice of the classifier head. The aggregate is also dominated by
    the FC layer, which has 128 of the 176 counted units.
    """
    c = tv2.WinCounter(model)
    tv2.evaluate(model, x_u8, y, idx, mean, std)
    out = {LAYER_NAMES.get(n, n): round((v == 0).float().mean().item(), 3)
           for n, v in c.counts.items()}
    c.remove()
    return out


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    commit = tv2.git_commit()
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_stage1_axis_boost")
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol="stage1", git_commit=commit,
               arms=[a for a, _ in ARMS], shots_per_class=SHOTS_PER_CLASS,
               seeds=SEEDS, max_steps=tv2.MAX_STEPS, eval_every=tv2.EVAL_EVERY,
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
            cell = {}

            for arm, build_fn in ARMS:
                tv2.set_seed(seed)               # identical init across arms
                model = build_fn()
                r = tv2.train_one_run(model, x_tr, y_tr, sub_idx, val_idx,
                                      x_te, y_te, test_idx, mean, std, seed)
                # train_one_run reloads the best checkpoint into `model`, so the
                # per-layer measurement below is taken at the scored checkpoint.
                per_layer = dead_by_layer(model, x_tr, y_tr, val_idx, mean, std)
                cell[arm] = r["test_acc"]

                row = dict(
                    protocol="stage1", git_commit=commit, arm=arm,
                    dim=("-" if arm == "dense" else ("channel" if "channel" in arm else "global")),
                    boost=(1.0 if arm.endswith("boost") else 0.0),
                    shots_per_class=shots, n_train_examples=sub_idx.numel(), seed=seed,
                    steps_run=r["steps_run"], hit_step_cap=int(r["hit_cap"]),
                    best_val_acc=round(r["best_val"], 4), best_val_step=r["best_step"],
                    test_acc=round(r["test_acc"], 4),
                    n_params=sum(p.numel() for p in model.parameters()),
                    dead_unit_frac=(None if r["dead"] != r["dead"] else round(r["dead"], 4)),
                    dead_by_layer=json.dumps(per_layer) if per_layer else "",
                    active_units=round(r["active"], 1),
                    overlap_same=round(r["ov_same"], 4),
                    overlap_diff=round(r["ov_diff"], 4),
                    separation=round(r["ov_same"] - r["ov_diff"], 4),
                    train_seconds=round(r["secs"], 1))
                rows.append(row)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                d = (r["test_acc"] - cell["dense"]) * 100
                dead_s = "   -" if r["dead"] != r["dead"] else f"{r['dead'] * 100:4.1f}%"
                print(f"  {shots:>4}/cls s={seed} {arm:<19} test={r['test_acc']:.4f} "
                      f"({d:+5.2f})  dead={dead_s}  sep={r['ov_same'] - r['ov_diff']:.3f}  "
                      f"act={r['active']:4.1f}  {r['secs']:5.1f}s")
                if per_layer:
                    print(f"  {'':>26}per-layer dead: {per_layer}")
            print()

    summarize(rows)
    print(f"\nSaved to {csv_path}")


def summarize(rows):
    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 92)
    print("STAGE 1 SUMMARY -- paired delta vs dense, percentage points")
    print("=" * 92)

    for n in sorted({r["n_train_examples"] for r in rows}):
        sub = [r for r in rows if r["n_train_examples"] == n]
        seeds = sorted({r["seed"] for r in sub})
        dense = {r["seed"]: r["test_acc"] for r in sub if r["arm"] == "dense"}
        ref = V2_REF.get(n, {})
        print(f"\n{n} labels   (v2 3-seed reference: kWTA delta {ref.get('delta', float('nan')):+.2f}, "
              f"dead {ref.get('dead', float('nan')) * 100:.0f}%, sep {ref.get('sep', float('nan')):.3f} "
              f"vs dense {ref.get('sep_dense', float('nan')):.3f})")
        print(f"  {'arm':<20}{'test acc':>10}{'delta':>9}{'signs':>8}{'dead':>8}{'sep':>8}{'act':>7}")
        for arm, _ in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            deltas = [(r["test_acc"] - dense[r["seed"]]) * 100 for r in a if r["seed"] in dense]
            signs = "".join("+" if d > 0 else "-" for d in deltas) if arm != "dense" else ""
            m = mean(deltas) if arm != "dense" else 0.0
            print(f"  {arm:<20}{mean([r['test_acc'] for r in a]):>10.4f}"
                  f"{m:>+9.2f}{signs:>8}"
                  f"{mean([r['dead_unit_frac'] for r in a]) * 100 if arm != 'dense' else float('nan'):>7.1f}%"
                  f"{mean([r['separation'] for r in a]):>8.3f}"
                  f"{mean([r['active_units'] for r in a]):>7.1f}")
        print(f"  (n={len(seeds)} seeds)")

    print("\n" + "-" * 92)
    print("READ IT LIKE THIS (pre-registered):")
    print("  * dead drops for kwta_channel at boost=0   -> the competition AXIS was the cause")
    print("  * dead drops only when boost=1             -> the DUTY CYCLE was the cause")
    print("  * best arm >= +1.0 pt over dense at 600    -> sparsity is a real low-data prior;")
    print("                                                sweep k, then CIFAR-10")
    print("  * best arm ~= dense with dead now low      -> codes were never the bottleneck;")
    print("                                                go to PLAN.md Stage 2 (episodic memory)")
    print("  * separation collapsing under boost        -> boosting traded away the mechanism;")
    print("                                                retune k rather than boosting harder")
    print("-" * 92)


if __name__ == "__main__":
    main()
