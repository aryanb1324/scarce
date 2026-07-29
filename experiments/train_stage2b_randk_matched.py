"""
Stage 2b: the random-k control, this time actually sparsity-matched.

=========================== WHY STAGE 2 DOESN'T COUNT ==========================

Stage 2 asked the only question left about channel-wise kWTA -- is the
input-dependent competition doing the work, or would any structured
sparsification do? -- and answered it with `randk_channel`: same k, same axis,
same nominal surviving count, winners drawn uniformly at random. kWTA beat it by
+12.9 pts. That number is confounded and must not be cited as-is.

The control matched kWTA on NOMINAL count and not on surviving signal. Measured
in the trained network, FC layer, nominal k = 0.2 x 128 = 26 units:

    arm               nominal selected     actually nonzero
    kwta_channel                    26                 26.0
    randk_channel                   26        7.6 (300 labels), 7.4 (600)

After ReLU roughly 70% of activations are exactly 0. `topk` by value lands on
the largest entries, all of which are nonzero, so kWTA's effective count equals
its nominal count. Uniform random selection lands on zeros in proportion to how
many there are and throws away most of its own budget. So `randk_channel` is not
"same sparsity, random selection" -- it is "3.4x sparser AND random selection
AND unrescaled," and the -11 pt collapse mixes at least three causes. The +12.9
gap is an UPPER BOUND on the competition effect, not a measurement of it.

`randk_matched` (architecture/modules/randk_matched.py) restricts the random
draw to the NONZERO entries at each site. Both rules then keep exactly
min(k, n_nonzero) live units per site -- kWTA because every nonzero outranks
every zero after ReLU, the control by construction -- so the arms differ in
exactly one property: whether the surviving subset is chosen by magnitude or at
random. That is the comparison Stage 2 was supposed to make.

================================== THE ARMS ===================================

  dense            no sparsification (the paired reference; cancels in the
                   head-to-head, but anchors both arms to the recorded runs)
  kwta_channel     k=0.2, dim="channel", boost=0.0 -- the +1.58 / +1.66 arm
  randk_matched    NEW. k random winners drawn from the live units. Live-unit
                   count identical to kwta_channel, selection input-independent.
  randk_nominal    the OLD confounded RandomKChannel at k=0.2, unchanged, kept
                   as the bridge to the recorded Stage 2 result. If it does not
                   reproduce its ~-11 pt delta and its ~7.5 live FC units, the
                   pipeline changed and nothing else in this run is comparable.

Budgets 300 and 600 labels (30 and 60 shots/class) -- where the effect lives.
EIGHT seeds, matching the Stage 2 / power-run standard. 64 runs.

MAX_STEPS 4000, PATIENCE 12, LR 1e-3, BATCH_SIZE 64 -- unchanged from every
recorded v2+ run, deliberately, because comparability depends on it.

============================== THE MEASUREMENT ================================

Rule 5 says every mechanism ships a measurement of the thing it claims to do,
and this run's entire claim is "the control is now matched." So the match is
PROVEN, not assumed: `eff_conv1` / `eff_conv2` / `eff_fc` log the mean number of
nonzero units per competition site at each of the three activation layers,
measured on a full validation pass through the best-validation checkpoint.
Reported per layer, not aggregated, per rule 8 -- the three layers have widths
16 / 32 / 128 and a single pooled number would be dominated by the FC layer,
which is exactly the error that once turned "47% of units are dead" into "94% of
them are in one layer."

Nominal k per layer: conv1 3/16, conv2 6/32, fc 26/128.

========================= PRE-REGISTERED PREDICTION ==========================

Written before the run. Do not edit afterwards, whatever the numbers say.

PRIMARY -- the head-to-head, paired per seed. Because both arms' deltas are
measured against the same dense run at the same seed, dense cancels exactly and
delta_kwta[i] - delta_randk_matched[i] is a paired difference in its own right.

  * kwta >= randk_matched + 1.0 pt, seeds agreeing
      -> THE COMPETITION IS THE MECHANISM, now on a control that is clean.
         Input-dependent selection does real work beyond the sparsity structure
         at matched live-unit count. The lateral-inhibition framing is earned,
         and the next question is what the winners encode that an equally-sized
         random live subset does not.

  * |kwta - randk_matched| < 0.5 pt
      -> THE COMPETITION IS DECORATIVE. The effect is structured sparsity at a
         matched signal level, and the earlier +12.9 was ENTIRELY the sparsity
         mismatch -- i.e. it measured "how much signal survives," not "how the
         survivors are chosen." The honest description is "a well-tuned
         structured regularizer with an interior optimum in k," the
         brain-inspired framing does no work, and it should be dropped from the
         paper rather than defended. This is a real, publishable result and the
         reason the control was worth running.

  * anything between (0.5 <= gap < 1.0, or gap < -0.5)
      -> INCONCLUSIVE. State the seeds needed rather than picking a side: at the
         paired sd observed here, n for 80% power at the observed gap. Do not
         round an inconclusive gap toward whichever branch is more convenient.

  * randk_matched >= kwta + 1.0 pt
      -> COMPETITION IS ACTIVELY HARMFUL and the benefit is stochasticity at the
         right sparsity level. This would make the mechanism a dropout variant
         and it should be reported as one.

SECONDARY -- the effective-count match, which licenses the primary comparison.

  * eff_fc(randk_matched) == eff_fc(kwta_channel) ~= 26.0, and eff_conv1 /
    eff_conv2 equal to within measurement noise
      -> the control is matched; the primary branch above is interpretable.
  * eff_fc(randk_matched) differs from eff_fc(kwta_channel) by more than 0.5
      -> the fix did not work and NOTHING in this run may be attributed to
         competition. Report the mismatch, do not report the gap.
  * eff_fc(randk_nominal) ~= 7.5, well below 26
      -> the Stage 2 confound reproduces, and the diagnosis in
         architecture/skills.md is confirmed rather than merely argued.

SECONDARY -- what the two randk arms say about each other.

  * randk_matched >> randk_nominal (say +8 pts or more)
      -> most of the Stage 2 collapse was the missing signal, not the missing
         competition. This is the arithmetic complement of the primary branch
         and is stated separately so it is reported either way.

Watch: both randk arms are stochastic at eval where kwta is deterministic. Over
10k test examples that variance sits far below seed variance, but if either
randk arm's per-seed spread is much larger than kwta's, discount a small gap.

Run with:
    QUICK=1 python3 -m experiments.train_stage2b_randk_matched   # smoke, ~2 min
    python3 -m experiments.train_stage2b_randk_matched           # 64 runs, ~55-70 min
"""

from __future__ import annotations

import csv
import json
import math
import os
import time

import torch

import experiments.train_v2 as tv2
from architecture.model import (
    build_dense_model,
    build_kwta_v2_model,
    build_randk_matched_model,
    build_randk_model,
)
from data.mnist_v2 import (
    load_mnist_tensors,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

QUICK = os.environ.get("QUICK") == "1"

# Frozen protocol constants -- identical to every recorded v2+ run.
tv2.MAX_STEPS = 4000 if not QUICK else 150
tv2.EVAL_EVERY = 100 if not QUICK else 50
tv2.PATIENCE = 12 if not QUICK else 2

K = 0.2  # the Stage 1b optimum; k is an architecture constant

SHOTS_PER_CLASS = [30, 60] if not QUICK else [30]
SEEDS = list(range(8)) if not QUICK else [0]

ARMS = [
    ("dense", lambda: build_dense_model()),
    ("kwta_channel", lambda: build_kwta_v2_model(dim="channel", boost=0.0, k=K)),
    ("randk_matched", lambda: build_randk_matched_model(k=K)),
    ("randk_nominal", lambda: build_randk_model(k=K)),
]

# The activation sites, and the nominal k each one selects at k = 0.2.
SITES = [("features.1", "eff_conv1", 3),
         ("features.4", "eff_conv2", 6),
         ("classifier.2", "eff_fc", 26)]

# 8-seed reference deltas vs dense from the Stage 2 / power runs, same protocol.
REF_KWTA = {300: +1.58, 600: +1.66}
REF_RANDK_NOMINAL_EFF_FC = 7.6

FIELDNAMES = ["protocol", "git_commit", "arm", "k", "shots_per_class",
              "n_train_examples", "seed", "steps_run", "hit_step_cap",
              "best_val_acc", "best_val_step", "test_acc", "n_params",
              "dead_unit_frac", "active_units", "eff_conv1", "eff_conv2",
              "eff_fc", "overlap_same", "overlap_diff", "separation",
              "train_seconds"]


# --------------------------------------------------------------------------
# effective live-unit count, per layer (rule 5: measure the claim; rule 8: per
# structurally-different component, never pooled)
# --------------------------------------------------------------------------

class EffectiveCounter:
    """
    Mean number of NONZERO units per competition site, per activation layer.

    For a conv layer a "site" is one (sample, h, w) position and the count runs
    over channels -- the same axis the competition runs over. For the FC layer a
    site is one sample. This is the number that must match between kwta_channel
    and randk_matched for the control to be valid, and the number that did NOT
    match in Stage 2 (26.0 vs 7.6 at the FC layer).

    Registered AFTER training, on the best-validation checkpoint, so it measures
    the network that was actually scored.
    """

    def __init__(self, model):
        self.sums, self.counts, self.handles = {}, {}, []
        mods = dict(model.named_modules())
        for name, col, _nominal in SITES:
            if name in mods:
                self.handles.append(mods[name].register_forward_hook(self._hook(col)))

    def _hook(self, col):
        def fn(_m, _i, out):
            o = out.detach()
            live = (o != 0).sum(dim=1).float()   # per site, over the competing axis
            self.sums[col] = self.sums.get(col, 0.0) + live.sum().item()
            self.counts[col] = self.counts.get(col, 0) + live.numel()
        return fn

    def result(self):
        return {col: (self.sums[col] / self.counts[col] if self.counts.get(col) else float("nan"))
                for _n, col, _k in SITES}

    def remove(self):
        for h in self.handles:
            h.remove()


def measure_effective_counts(model, x_u8, y, idx, mean, std):
    c = EffectiveCounter(model)
    tv2.evaluate(model, x_u8, y, idx, mean, std)
    out = c.result()
    c.remove()
    return out


# --------------------------------------------------------------------------
# paired statistics (no scipy in this environment)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_stats(diffs):
    """mean, sd, se, t, two-sided p, sign string -- for one paired vector."""
    n = len(diffs)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), se=float("nan"),
                    t=float("nan"), p=float("nan"), signs="")
    m = sum(diffs) / n
    if n < 2:
        return dict(n=n, mean=m, sd=0.0, se=0.0, t=float("nan"), p=float("nan"),
                    signs="+" if m > 0 else "-")
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    if se == 0.0:
        t = float("inf") if m != 0 else 0.0
        p = 0.0 if m != 0 else 1.0
    else:
        t = m / se
        df = n - 1
        p = _betai(df / 2.0, 0.5, df / (df + t * t))
    signs = "".join("+" if d > 0 else ("-" if d < 0 else "0") for d in diffs)
    return dict(n=n, mean=m, sd=sd, se=se, t=t, p=p, signs=signs)


def seeds_needed(gap: float, sd: float) -> float:
    """Rough n for 80% power at alpha=0.05, two-sided, paired."""
    if gap == 0.0 or sd <= 0.0:
        return float("inf")
    return math.ceil((2.8 * sd / abs(gap)) ** 2)


# --------------------------------------------------------------------------

def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    commit = tv2.git_commit()
    tag = "stage2b_randk_matched"
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol=tag, git_commit=commit, arms=[a for a, _ in ARMS], k=K,
               shots_per_class=SHOTS_PER_CLASS, seeds=SEEDS,
               max_steps=tv2.MAX_STEPS, eval_every=tv2.EVAL_EVERY,
               patience=tv2.PATIENCE, batch_size=tv2.BATCH_SIZE, lr=tv2.LR,
               selection="best-on-validation", device=str(tv2.DEVICE),
               quick=QUICK,
               measures=["eff_conv1 (3/16)", "eff_conv2 (6/32)", "eff_fc (26/128)"])
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
                # best-val checkpoint is already loaded by train_one_run
                eff = measure_effective_counts(model, x_tr, y_tr, val_idx, mean, std)
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
                    eff_conv1=round(eff["eff_conv1"], 3),
                    eff_conv2=round(eff["eff_conv2"], 3),
                    eff_fc=round(eff["eff_fc"], 3),
                    overlap_same=round(r["ov_same"], 4),
                    overlap_diff=round(r["ov_diff"], 4),
                    separation=round(r["ov_same"] - r["ov_diff"], 4),
                    train_seconds=round(r["secs"], 1))
                rows.append(row)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
                print(f"  {shots:>4}/cls s={seed} {arm:<15} test={r['test_acc']:.4f} "
                      f"({(r['test_acc'] - dense_acc) * 100:+6.2f})  "
                      f"eff={eff['eff_conv1']:4.2f}/{eff['eff_conv2']:4.2f}/"
                      f"{eff['eff_fc']:5.2f}  sep={r['ov_same'] - r['ov_diff']:.3f}  "
                      f"{r['secs']:5.1f}s")
            print()

    summarize(rows)
    print(f"\nSaved to {csv_path}")


def summarize(rows):
    arm_names = [a for a, _ in ARMS]
    budgets = sorted({r["n_train_examples"] for r in rows})

    print("\n" + "=" * 96)
    print("STAGE 2b -- EFFECTIVE LIVE-UNIT COUNT PER LAYER (the claim being proven)")
    print("nominal at k=0.2:  conv1 3/16   conv2 6/32   fc 26/128")
    print("=" * 96)
    for n in budgets:
        sub = [r for r in rows if r["n_train_examples"] == n]
        print(f"\n{n} labels")
        print(f"  {'arm':<16}{'eff_conv1':>11}{'eff_conv2':>11}{'eff_fc':>10}"
              f"{'active_units':>14}")
        for arm in arm_names:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            print(f"  {arm:<16}"
                  f"{sum(r['eff_conv1'] for r in a) / len(a):>11.2f}"
                  f"{sum(r['eff_conv2'] for r in a) / len(a):>11.2f}"
                  f"{sum(r['eff_fc'] for r in a) / len(a):>10.2f}"
                  f"{sum(r['active_units'] for r in a) / len(a):>14.1f}")

    # -- the match verdict, which licenses everything below it -----------------
    print("\n" + "-" * 96)
    print("SECONDARY VERDICT -- is the control actually matched?")
    matched_ok = True
    for n in budgets:
        sub = [r for r in rows if r["n_train_examples"] == n]

        def mean_eff(arm, col):
            a = [r[col] for r in sub if r["arm"] == arm]
            return sum(a) / len(a) if a else float("nan")

        km, rm, rn = (mean_eff("kwta_channel", "eff_fc"),
                      mean_eff("randk_matched", "eff_fc"),
                      mean_eff("randk_nominal", "eff_fc"))
        d = abs(km - rm)
        print(f"\n  {n} labels  eff_fc:  kwta {km:.2f}   randk_matched {rm:.2f}"
              f"   randk_nominal {rn:.2f}   |kwta-matched| = {d:.3f}")
        if d > 0.5:
            matched_ok = False
            print("    -> NOT MATCHED. The fix did not work. Nothing in this run may be")
            print("       attributed to competition; report the mismatch, not the gap.")
        else:
            print("    -> MATCHED (within 0.5 live FC units). The primary comparison is licensed.")
        if rn == rn and rn < km - 0.5:
            print(f"    -> the Stage 2 confound reproduces ({rn:.1f} vs {km:.1f} live FC units,"
                  f" {km / rn if rn else float('nan'):.1f}x); "
                  f"reference was {REF_RANDK_NOMINAL_EFF_FC}.")

    # -- paired deltas vs dense ----------------------------------------------
    print("\n" + "=" * 96)
    print("PAIRED DELTA vs DENSE, percentage points (same seed, same subset, dense cancels)")
    print("=" * 96)
    per_budget = {}
    for n in budgets:
        sub = [r for r in rows if r["n_train_examples"] == n]
        dense = {r["seed"]: r["test_acc"] for r in sub if r["arm"] == "dense"}
        ref = REF_KWTA.get(n)
        print(f"\n{n} labels" + (f"   (8-seed reference for kwta: {ref:+.2f})" if ref else ""))
        print(f"  {'arm':<16}{'test acc':>10}{'delta':>9}{'sd':>7}{'se':>7}{'t':>8}"
              f"{'p':>10}{'signs':>11}{'sep':>8}")
        per_budget[n] = {}
        for arm in arm_names:
            a = sorted([r for r in sub if r["arm"] == arm], key=lambda r: r["seed"])
            if not a:
                continue
            accs = [r["test_acc"] for r in a]
            deltas = [(r["test_acc"] - dense[r["seed"]]) * 100 for r in a if r["seed"] in dense]
            per_budget[n][arm] = {r["seed"]: r["test_acc"] for r in a}
            if arm == "dense":
                print(f"  {arm:<16}{sum(accs) / len(accs):>10.4f}{0.0:>+9.2f}"
                      f"{'':>7}{'':>7}{'':>8}{'':>10}{'':>11}"
                      f"{sum(r['separation'] for r in a) / len(a):>8.3f}")
                continue
            s = paired_stats(deltas)
            print(f"  {arm:<16}{sum(accs) / len(accs):>10.4f}{s['mean']:>+9.2f}"
                  f"{s['sd']:>7.2f}{s['se']:>7.2f}{s['t']:>8.2f}{s['p']:>10.4f}"
                  f"{s['signs']:>11}{sum(r['separation'] for r in a) / len(a):>8.3f}")

    # -- the primary comparison ----------------------------------------------
    print("\n" + "=" * 96)
    print("PRIMARY -- kwta_channel vs randk_matched, PAIRED per seed")
    print("(both deltas are vs the same dense run, so dense cancels exactly:")
    print(" diff[i] = acc_kwta[i] - acc_randk_matched[i])")
    print("=" * 96)
    for n in budgets:
        acc = per_budget.get(n, {})
        if "kwta_channel" not in acc or "randk_matched" not in acc:
            continue
        seeds = sorted(set(acc["kwta_channel"]) & set(acc["randk_matched"]))
        diffs = [(acc["kwta_channel"][s] - acc["randk_matched"][s]) * 100 for s in seeds]
        st = paired_stats(diffs)
        nom = None
        if "randk_nominal" in acc:
            ns = sorted(set(acc["kwta_channel"]) & set(acc["randk_nominal"]))
            nom = paired_stats([(acc["kwta_channel"][s] - acc["randk_nominal"][s]) * 100
                                for s in ns])
        print(f"\n  {n} labels   n={st['n']}")
        print(f"    gap = {st['mean']:+.2f} pts   sd {st['sd']:.2f}   se {st['se']:.2f}"
              f"   t {st['t']:.2f}   p {st['p']:.4f}   signs {st['signs']}")
        if nom:
            print(f"    (vs the OLD confounded control: gap = {nom['mean']:+.2f}, "
                  f"p {nom['p']:.4f}, signs {nom['signs']})")
        g = st["mean"]
        if not matched_ok:
            print("    -> WITHHELD. The control is not matched at this run; see above.")
        elif g >= 1.0:
            print("    -> COMPETITION IS THE MECHANISM, on a clean control. Input-dependent")
            print("       selection does real work beyond the sparsity structure at matched")
            print("       live-unit count. The lateral-inhibition framing is earned.")
        elif abs(g) < 0.5:
            print("    -> COMPETITION IS DECORATIVE. The effect is structured sparsity at a")
            print("       matched signal level; the earlier +12.9 was ENTIRELY the sparsity")
            print("       mismatch. Describe it as a well-tuned structured regularizer and")
            print("       DROP the brain-inspired framing rather than defending it.")
        elif g <= -1.0:
            print("    -> COMPETITION IS ACTIVELY HARMFUL; the benefit is stochasticity at the")
            print("       right sparsity level. Report this as a dropout variant.")
        else:
            print(f"    -> INCONCLUSIVE at n={st['n']}. Need ~{seeds_needed(g, st['sd']):.0f} "
                  f"seeds for 80% power at this gap and sd. Do not round toward a branch.")
        if "randk_nominal" in acc and "randk_matched" in acc:
            rs = sorted(set(acc["randk_matched"]) & set(acc["randk_nominal"]))
            rr = paired_stats([(acc["randk_matched"][s] - acc["randk_nominal"][s]) * 100
                               for s in rs])
            print(f"    randk_matched - randk_nominal = {rr['mean']:+.2f} pts "
                  f"(p {rr['p']:.4f}, signs {rr['signs']})")
            if rr["mean"] >= 8.0:
                print("       -> most of the Stage 2 collapse was the MISSING SIGNAL, not the")
                print("          missing competition.")
    print("-" * 96)


if __name__ == "__main__":
    main()
