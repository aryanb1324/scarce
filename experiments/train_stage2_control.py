"""
Stage 2: is the competition doing the work, or is it just structured sparsity?

============================= WHAT STAGE 1b FOUND ============================

The k-sweep confirmed the effect and destroyed the explanation.

  paired delta vs dense (pts), channel-wise, 5 seeds

    labels    k=0.05    k=0.10    k=0.20    k=0.40      dense acc
       100     -1.98     -1.46     -0.79     -2.09        81.02%
       300     -0.75     +1.06    +1.55 *    -1.03        89.75%
       600     -0.08     +0.84    +1.43 ***  -0.69        92.91%
     3,000     -0.80     +0.06     +0.19     -0.05        96.90%

CONFIRMED: the effect is real and reproduces. k=0.2 wins at 300 labels (+1.55,
5/5 seeds, p=0.024, 1.40x label multiplier) and at 600 (+1.43, 5/5, p=0.0003,
1.78x).

FALSIFIED, and both were pre-registered:

  * "The advantage grows as labels get scarcer." It does not. It is non-monotonic
    -- -0.79 at 100 labels, +1.55 at 300, +1.43 at 600, +0.19 at 3,000. There is
    an optimum data budget, not a prior that strengthens with scarcity. (Caveat:
    at 100 labels the paired sd is 1.47 against 0.26 at 600, so resolving a
    1-point effect there needs ~17 seeds, not 5. The reversal is suggestive, not
    established -- see POWER mode below.)

  * "Best k shifts sparser as labels fall." It does not move at all. k* = 0.20 at
    every budget across a 30x range. k is an architecture constant, so the
    evidence-scaled / annealed-sparsity idea is dead before any code was written
    for it. This is exactly what the pre-registration was for.

  * **Pattern separation is not the mechanism.** The sparsest arm has the highest
    separation at every budget (k=0.05: 0.227 -> 0.298) and the worst accuracy.
    Across all 16 kWTA cells, separation vs paired delta is r = +0.19, p = 0.48.
    The "sparse codes -> pattern separation -> data efficiency" account survived
    v2 and looked confirmed; measuring it across k killed it.

An interior optimum in k with no representational correlate is the signature of a
benefit/cost tradeoff -- most likely regularization strength against usable
capacity (dead units run 72% at k=0.05 down to 35% at k=0.4). Which raises the
question this experiment exists to answer.

================================= THE QUESTION ===============================

Does the *input-dependent competition* matter, or would any sparsification with
the same structure work equally well?

Four arms, identical in every respect except how the surviving units are chosen:

  dense              no sparsification (reference)
  kwta_channel       top-k channels at each position -- competitive, input-dependent
  randk_channel      k RANDOM channels at each position -- same k, same axis, same
                     exact count, same always-on train/eval behaviour. The ONLY
                     difference from kwta_channel is that selection ignores the input.
  dropout_matched    conventional Dropout2d/Dropout at p = 1-k -- train-only,
                     rescaled, stochastic count. The "isn't this just spatial
                     dropout?" objection, answered with a measurement.

Budgets 300 and 600 only: those are where the effect lives, and 100 and 3,000 add
cost without information (100 needs a separate power run, 3,000 is a flat zero).
EIGHT seeds, up from five, because 300 labels has sd = 0.97 and five seeds only
just resolved it.

========================= PRE-REGISTERED PREDICTION =========================

Written before the run. Do not edit afterwards.

  * kwta >= randk + 1.0 pt, seeds agreeing
      -> THE COMPETITION IS THE MECHANISM. Input-dependent selection is doing real
         work beyond the sparsity structure. The lateral-inhibition framing is
         earned, and the next question is what the winners encode that random
         subsets do not.

  * kwta ~= randk (within ~0.5 pt)
      -> THE COMPETITION IS DECORATIVE. What helps is structured sparsity at the
         right level, and the honest description is "a well-tuned structured
         regularizer with an interior optimum," not lateral inhibition. The
         brain-inspired framing does no work and should be dropped from the paper
         rather than defended. This is a real result and it is publishable; it is
         just a different paper.

  * randk >> kwta
      -> the competition is actively harmful and the benefit is stochasticity.
         That would make this a dropout variant; report it as such.

  * dropout_matched ~= kwta
      -> whatever the mechanism, it is reachable with a standard tool and the
         contribution is the protocol and the measurement, not the module.

  Watch: randk is stochastic at eval where kwta is deterministic. Over 10k test
  examples that variance is far below the seed variance, but if randk's per-seed
  spread is much larger than kwta's, discount a small gap between them.

POWER MODE
----------
`POWER=1` runs dense vs kwta_channel k=0.2 at 100 labels with 20 seeds and nothing
else, to settle whether the reversal at 100 labels is real or is the 5-seed noise
floor (sd 1.47 there needs ~17 seeds to resolve 1 point at 80% power). ~35 min.
Do this one first -- it is cheap and it decides whether the non-monotonicity goes
in the paper as a finding or as an unresolved caveat.

Run with:
    QUICK=1 python -m experiments.train_stage2_control    # smoke, ~2 min
    POWER=1 python -m experiments.train_stage2_control    # 100-label power, ~35 min
    python -m experiments.train_stage2_control            # 64 runs, ~55 min
"""

import csv
import json
import os
import time

import torch

import experiments.train_v2 as tv2
from architecture.model import (
    build_dense_model,
    build_dropout_model,
    build_kwta_v2_model,
    build_randk_model,
)
from data.mnist_v2 import (
    load_mnist_tensors,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

QUICK = os.environ.get("QUICK") == "1"
POWER = os.environ.get("POWER") == "1"

tv2.MAX_STEPS = 4000 if not QUICK else 150
tv2.EVAL_EVERY = 100 if not QUICK else 50
tv2.PATIENCE = 12 if not QUICK else 2

K = 0.2  # fixed at the Stage 1b optimum; k is an architecture constant

if POWER:
    SHOTS_PER_CLASS, SEEDS = [10], list(range(20))
    ARMS = [("dense", lambda: build_dense_model()),
            ("kwta_channel", lambda: build_kwta_v2_model(dim="channel", boost=0.0, k=K))]
else:
    SHOTS_PER_CLASS = [30, 60] if not QUICK else [30]
    SEEDS = list(range(8)) if not QUICK else [0]
    ARMS = [
        ("dense", lambda: build_dense_model()),
        ("kwta_channel", lambda: build_kwta_v2_model(dim="channel", boost=0.0, k=K)),
        ("randk_channel", lambda: build_randk_model(k=K)),
        ("dropout_matched", lambda: build_dropout_model(k=K)),
    ]

# Stage 1b reference, 5 seeds, same protocol.
REF = {300: +1.55, 600: +1.43, 100: -0.79}

FIELDNAMES = ["protocol", "git_commit", "arm", "k", "shots_per_class",
              "n_train_examples", "seed", "steps_run", "hit_step_cap",
              "best_val_acc", "best_val_step", "test_acc", "n_params",
              "dead_unit_frac", "active_units", "overlap_same", "overlap_diff",
              "separation", "train_seconds"]


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    commit = tv2.git_commit()
    tag = "stage2_power" if POWER else "stage2_control"
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol=tag, git_commit=commit, arms=[a for a, _ in ARMS], k=K,
               shots_per_class=SHOTS_PER_CLASS, seeds=SEEDS,
               max_steps=tv2.MAX_STEPS, eval_every=tv2.EVAL_EVERY,
               patience=tv2.PATIENCE, batch_size=tv2.BATCH_SIZE, lr=tv2.LR,
               selection="best-on-validation", device=str(tv2.DEVICE),
               quick=QUICK, power=POWER)
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
                    protocol=tag, git_commit=commit, arm=arm,
                    k=("" if arm == "dense" else K),
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
                print(f"  {shots:>4}/cls s={seed} {arm:<17} test={r['test_acc']:.4f} "
                      f"({(r['test_acc'] - dense_acc) * 100:+5.2f})  "
                      f"sep={r['ov_same'] - r['ov_diff']:.3f}  act={r['active']:4.1f}  "
                      f"{r['secs']:5.1f}s")
            print()

    summarize(rows)
    print(f"\nSaved to {csv_path}")


def summarize(rows):
    import math

    def stats(v):
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        return m, sd, (sd / math.sqrt(n) if n else 0.0)

    print("\n" + "=" * 88)
    print("STAGE 2 SUMMARY -- paired delta vs dense, percentage points")
    print("=" * 88)

    verdict = {}
    for n in sorted({r["n_train_examples"] for r in rows}):
        sub = [r for r in rows if r["n_train_examples"] == n]
        dense = {r["seed"]: r["test_acc"] for r in sub if r["arm"] == "dense"}
        ref = REF.get(n)
        print(f"\n{n} labels" + (f"   (Stage 1b 5-seed reference for kwta: {ref:+.2f})" if ref else ""))
        print(f"  {'arm':<18}{'test acc':>10}{'delta':>9}{'sd':>7}{'se':>7}{'signs':>11}{'sep':>8}")
        for arm, _ in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            accs = [r["test_acc"] for r in a]
            deltas = [(r["test_acc"] - dense[r["seed"]]) * 100 for r in a if r["seed"] in dense]
            m, sd, se = stats(deltas) if arm != "dense" else (0.0, 0.0, 0.0)
            signs = "".join("+" if d > 0 else "-" for d in deltas) if arm != "dense" else ""
            seps = [r["separation"] for r in a]
            print(f"  {arm:<18}{sum(accs) / len(accs):>10.4f}{m:>+9.2f}{sd:>7.2f}{se:>7.2f}"
                  f"{signs:>11}{sum(seps) / len(seps):>8.3f}")
            verdict.setdefault(n, {})[arm] = m

    print("\n" + "-" * 88)
    print("THE COMPARISON THIS RUN EXISTS FOR: kwta_channel vs randk_channel")
    for n, v in verdict.items():
        if "randk_channel" not in v:
            continue
        gap = v["kwta_channel"] - v["randk_channel"]
        print(f"\n  {n} labels: kwta {v['kwta_channel']:+.2f}   randk {v['randk_channel']:+.2f}"
              f"   dropout {v.get('dropout_matched', float('nan')):+.2f}   gap = {gap:+.2f} pts")
        if gap >= 1.0:
            print("    -> COMPETITION IS THE MECHANISM. Input-dependent selection does real work")
            print("       beyond the sparsity structure; the lateral-inhibition framing is earned.")
        elif abs(gap) < 0.5:
            print("    -> COMPETITION IS DECORATIVE. Structured sparsity at the right level is the")
            print("       whole effect. Describe it as a well-tuned structured regularizer and drop")
            print("       the brain-inspired framing rather than defending it. Still a real result.")
        elif gap <= -1.0:
            print("    -> RANDOM BEATS COMPETITION. The benefit is stochasticity; this is a dropout")
            print("       variant and should be reported as one.")
        else:
            print("    -> INCONCLUSIVE at this n. Add seeds before writing anything down.")
    print("-" * 88)


if __name__ == "__main__":
    main()
