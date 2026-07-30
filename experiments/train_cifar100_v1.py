"""
CIFAR-100 v1 -- the harder gate. Does the effect survive 100 fine classes?

=============================== WHY THIS EXISTS ==============================

`train_cifar_v1.py` moved the project's central claim off MNIST and onto CIFAR-10.
CIFAR-10 is harder than MNIST but still only ten coarse classes a tiny CNN can
pull apart from colour and gross shape. CIFAR-100 keeps the same 32x32 natural
images and splits them into 100 fine classes with only 500 training images each,
so at a few shots per class the task is genuinely hard: a from-scratch tiny CNN
sits nowhere near saturation, and the low-data regime is real rather than
manufactured. If "a structural sparsity prior buys data efficiency where labels
are scarce" is a property of the mechanism, this is where it should show; if it
was a property of near-saturated toy datasets, this is where it should die.

This is a NEW versioned entry point (protocol rule 6: one variable per experiment,
new protocol -> new script, never an in-place edit). The variable changed relative
to train_cifar_v1.py is THE DATASET (CIFAR-10 -> CIFAR-100, and with it the number
of classes and the per-class budget). Everything else is held at the recorded
values:

    MAX_STEPS 4000, PATIENCE 12, EVAL_EVERY 100, BATCH_SIZE 64, LR 1e-3,
    train-to-convergence with best-on-validation selection, paired per-seed
    deltas, identical batch sequence across arms at a given seed.

The model comes from `scarce.nets.build_net`, not `architecture/baseline.py`.
baseline.py is frozen AND hardcoded to 1x28x28, so it physically cannot take a
3-channel input; `build_net` keeps the structure the research validated (two conv
blocks, one 128-unit hidden FC layer, an activation factory swapped in at every
activation site) and adapts only the input stem, flatten width, and the 100-way
head. Both arms share one architecture and differ ONLY in the activation, so a
delta cannot be a model-size difference -- parameter counts are logged to prove it.

================================== THE ARMS ==================================

    dense           nn.ReLU                                  (reference)
    kwta_channel    ReLU -> KWinnersTakeAllV2(k=0.2, dim="channel", boost=0.0)
    dropout         ReLU -> StructuredDropout(p=0.8)         (matched surviving
                                                              fraction; the
                                                              "isn't this just
                                                              spatial dropout?"
                                                              control)

k = 0.20, dim = "channel", boost = 0.0 is the exact configuration that won on
MNIST and was carried unchanged to CIFAR-10. k is NOT re-swept here: re-sweeping
k in the same run that changes the dataset would confound the two. If the effect
vanishes at k = 0.2 here, the follow-up experiment is a k sweep on CIFAR-100 -- a
separate script.

    TODO(pretrained arm): a pretrained-backbone arm is being added separately.
    When it lands, slot it into ARMS below as an additional entry (name e.g.
    "pretrained") and add its factory/builder. Do NOT import it here yet -- this
    script must run and be interpretable without it, and the pre-registration
    below deliberately does not depend on it.

============================== THE BUDGET CHOICE =============================

Budgets are PER-CLASS shot counts, because with 100 classes a fixed total-label
budget would put a fraction of a shot in some classes. The three budgets and their
total-label meaning:

      5 shots/class   ->    500 labels
     10 shots/class   ->  1,000 labels
     25 shots/class   ->  2,500 labels

The pool ceiling is 400/class (500 train minus the 100/class frozen val holdout),
so all three are comfortably inside budget. These are small on purpose: 500-2,500
labels spread over 100 classes is the scarce end, which is where the claim lives.

============================= THE ACCURACY REGIME ===========================

Stated in advance, because it changes how every branch below must be read. A
from-scratch two-conv-block CNN on CIFAR-100 at 5-25 shots per class will be at
VERY LOW absolute accuracy -- expect roughly 5-15% test accuracy, well above the
1% chance rate but nowhere near usable. There is no accuracy-matched comparison to
MNIST or CIFAR-10 available: this network cannot reach those datasets' accuracies
on CIFAR-100 at any budget in this sweep. So a per-point delta of +1.0 pt here is
a LARGE relative effect on a small base, and "did not replicate at matched shots"
and "did not replicate in a near-chance regime" are not separable by this run --
neither may be stated without the other.

This near-chance, high-class-count regime is EXACTLY where a pretrained backbone
should dominate a from-scratch net, which is the motivation for the pretrained arm
flagged above: the honest expectation is that no from-scratch activation trick
closes the gap that transfer learning would, and this run measures how far a
mechanism-only change can move the needle before that arm is added.

========================= PRE-REGISTERED PREDICTION =========================

Written and committed to before the run. Do not edit after seeing results
(experiments/skills.md, and CLAUDE.md rule 4).

REFERENCE -- channel-wise kWTA k=0.2 vs dense, same protocol:
    MNIST, 8 seeds:        +1.58 at 300 labels,   +1.66 at 600 labels
    CIFAR-10 (this net):   being measured concurrently by train_cifar_v1.py

The primary comparison is `kwta_channel - dense` at 5 and 10 shots/class
(500 and 1,000 CIFAR-100 labels).

PRIOR: the MNIST/CIFAR-10 effect, where it exists, is a ~1-2 pt regularization
benefit that is largest in a moderate-accuracy regime and shrinks toward null both
where data is very scarce (paired sd blows up) and where data is plentiful (the
baseline learns the structure itself). CIFAR-100 at 5-25 shots/class is a
near-chance regime with high paired variance, so the honest prior is a SMALL,
NOISY effect trending toward null -- most probably branch B or branch D. This
prediction is registered as expecting NON-replication, and branch A firing would
be the surprising, claim-strengthening outcome.

  BRANCH A -- REPLICATES.
    delta >= +1.0 pt at 5 and/or 10 shots/class, with >=80% sign consistency and a
    paired se small enough that the seeds run resolve 1.0 pt at 80% power.
    -> The effect is a property of the mechanism, robust to a 10x harder task with
       100 fine classes. The central claim survives its hardest test to date and is
       no longer plausibly a toy-dataset artifact. Next: sweep k on CIFAR-100, and
       compare against the pretrained arm to see whether the mechanism buys
       anything transfer learning does not already give for free.

  BRANCH B -- VANISHES.
    |delta| < 0.5 pt at BOTH 5 and 10 shots/class, or signs split near 50/50, with
    enough seeds to resolve 1.0 pt.
    -> The MNIST/CIFAR-10 effect does not generalize to a hard, many-class,
       near-chance regime. It was a property of moderate-accuracy datasets where
       sparsification acts as a cheap regularizer. Report as the headline finding,
       not a footnote: the gate is allowed to close, and the whole point of running
       it is that it can.

  BRANCH C -- REVERSES.
    delta <= -1.0 pt at 5 and/or 10 shots/class, with >=80% sign consistency.
    -> The mechanism actively costs accuracy. Most likely reading: with 100 classes
       and near-chance accuracy, throwing away 80% of channel activations discards
       signal the net cannot spare, so channel-wise competition is anti-generic in
       the hard regime and the brain-inspired framing should be dropped, not scoped.

  BRANCH D -- UNDERPOWERED.
    Anything else, or a measured paired sd large enough that the seeds run cannot
    resolve 1.0 pt at 80% power.
    -> Report the noise floor and the seeds required; conclude nothing. n = 3 is a
       direction check, never a significance test (CLAUDE.md rule 3). At a
       near-chance base the paired sd is expected to be large, so D is a genuinely
       likely and legitimate pilot outcome, NOT a soft 'no effect'.

  SECONDARY, registered but not decisive:
    * dropout ~= kwta_channel (within 0.5 pt) -> whatever helps is reachable with a
      standard tool, and the contribution is the protocol, not the module.
    * dead-unit fraction, reported PER COMPONENT (conv1, conv2, fc) -- rule 8.
      Registered EXPECTATION: comparable to or higher than CIFAR-10, since 100-way
      classification drives an even wider spread of channel activations.
    * separation (same- minus different-class Jaccard overlap of the penultimate
      code) is logged but NOT a hypothesis: Stage 1b falsified separation as the
      causal account on MNIST (r = +0.19, p = 0.48). Logged so the CIFAR-100
      correlation can be checked later, not argued from. NOTE: with 100 classes the
      different-class pair pool dominates, so absolute overlap numbers are not
      comparable to the 10-class runs -- only the sign and trend are.

================================== COMPUTE ==================================

Device policy is identical to train_cifar_v1.py and set by the same measurement:
`torch.topk` along dim=1 (the operation channel-wise kWTA is built on) is ~20x
slower on MPS than CPU, making the kWTA arm ~4x slower overall on MPS. Since every
arm must share a device for the comparison to be paired and the kWTA arm dominates
cost, the default is CPU. Set AN_DEVICE=mps to override; the device is logged.

Run with:
    QUICK=1 python3 -m experiments.train_cifar100_v1     # smoke test
    PILOT=1 python3 -m experiments.train_cifar100_v1     # 3 seeds x 5/10 shots
            python3 -m experiments.train_cifar100_v1     # 5 seeds x 3 budgets

Conditions run cheapest-budget-first and every row is appended to the CSV as it
completes, so an interrupt keeps the finished work.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import subprocess
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from architecture.modules.kwta_v2 import KWinnersTakeAllV2
from architecture.modules.randk import StructuredDropout
from scarce.nets import build_net
from data.cifar100 import (
    assert_disjoint,
    load_cifar100_tensors,
    make_batch,
    preflight,
    subset_norm_stats,
    train_val_split,
)

QUICK = os.environ.get("QUICK") == "1"
PILOT = os.environ.get("PILOT") == "1"

# ---------------------------- protocol constants -----------------------------
# FROZEN at the recorded MNIST/CIFAR-10 values. Comparability across datasets is
# the entire point of this run; changing any of these makes the CIFAR-100 numbers
# incomparable to the earlier ones and the experiment answers nothing.
MAX_STEPS = 4000 if not QUICK else 150
EVAL_EVERY = 100 if not QUICK else 50
PATIENCE = 12 if not QUICK else 2
BATCH_SIZE = 64
LR = 1e-3
# -----------------------------------------------------------------------------

K = 0.2                       # MNIST optimum, carried to CIFAR-10; not re-swept here
OVERLAP_N_PER_CLASS = 10      # val images per class for the code-overlap metric.
                              # Smaller than CIFAR-10's 60: with 100 classes,
                              # 60/class -> 6,000 images -> a 6,000^2 Jaccard matrix.
                              # 10/class -> 1,000 images keeps the metric cheap.
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 100

if QUICK:
    SHOTS_PER_CLASS, SEEDS = [5], [0]
elif PILOT:
    SHOTS_PER_CLASS, SEEDS = [5, 10], [0, 1, 2]
else:
    SHOTS_PER_CLASS, SEEDS = [5, 10, 25], [0, 1, 2, 3, 4]

# Total-label meaning of each per-class budget, for the summary header.
TOTAL_LABELS = {s: s * NUM_CLASSES for s in (5, 10, 25)}

# MNIST reference, same protocol, same arm, 8 seeds (label-count budgets, not
# directly comparable to CIFAR-100's per-class budgets -- printed for context only).
MNIST_DELTA_REF = {300: +1.58, 600: +1.66}


def _kwta_factory():
    return lambda: nn.Sequential(nn.ReLU(),
                                 KWinnersTakeAllV2(k=K, dim="channel", boost=0.0))


def _dropout_factory():
    return lambda: nn.Sequential(nn.ReLU(), StructuredDropout(p=1.0 - K))


ARMS: List[Tuple[str, object]] = [
    ("dense", nn.ReLU),
    ("kwta_channel", _kwta_factory()),
    ("dropout", _dropout_factory()),
    # TODO(pretrained arm): append ("pretrained", <factory-or-builder>) here once
    # it is available. Do not import it above until then.
]
if QUICK:
    ARMS = ARMS[:2]

FIELDNAMES = [
    "protocol", "git_commit", "device", "arm", "k", "shots_per_class",
    "n_train_examples", "seed", "steps_run", "hit_step_cap", "best_val_acc",
    "best_val_step", "test_acc", "n_params", "active_units",
    "dead_frac_conv1", "dead_frac_conv2", "dead_frac_fc", "dead_frac_all",
    "overlap_same", "overlap_diff", "separation", "train_seconds",
]


# --------------------------------- plumbing ----------------------------------

def pick_device() -> torch.device:
    """CPU by default, and the docstring records the measurement that decided it.

    MPS is available on this machine and is genuinely faster for the dense arm,
    but `topk` along dim=1 -- the operation channel-wise kWTA is built on -- is
    ~20x slower there than on CPU, which makes the sparse arm 4x slower overall.
    Both arms must share a device or the comparison stops being paired, so the
    slow arm picks the device.
    """
    override = os.environ.get("AN_DEVICE")
    mps = torch.backends.mps.is_available()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("device: cpu  (mps available={}; not used -- topk(dim=1) is ~20x slower "
          "on mps and the kWTA arm dominates cost. AN_DEVICE=mps overrides.)"
          .format(mps))
    return torch.device("cpu")


DEVICE = pick_device()


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


def activation_sites(model) -> List[Tuple[str, nn.Module]]:
    """The three structurally different activation sites, named.

    Rule 8: any diagnostic aggregated over structurally different components gets
    reported per component. A single dead-unit number weighted by layer width is
    dominated by the widest layer and points at the wrong cause -- a "47% of units
    are dead" figure in this project turned out to be 94% one layer.
    """
    sites = []
    conv_i = 0
    for m in model.features:
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            if isinstance(m, nn.Conv2d):
                conv_i += 1
            continue
        sites.append(("conv{}".format(conv_i), m))
    sites.append(("fc", model.penultimate))
    return sites


class DeadUnitCounter:
    """Counts, per activation site, the units that are never nonzero over a pass.

    Measured for EVERY arm, not just the sparse one. A dying-ReLU baseline is a
    real phenomenon and the dense arm's number is the only thing that makes the
    sparse arm's interpretable.

    A unit is a CHANNEL for 4-D outputs (weights are shared across space, so the
    channel is the thing whose capacity can actually be lost) and a unit for 2-D.
    """

    def __init__(self, model):
        self.counts: Dict[str, torch.Tensor] = {}
        self.handles = []
        for name, mod in activation_sites(model):
            self.handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_m, _i, out):
            o = out.detach()
            c = (o != 0).sum(dim=(0, 2, 3)) if o.dim() == 4 else (o != 0).sum(dim=0)
            c = c.cpu()
            self.counts[name] = c if name not in self.counts else self.counts[name] + c
        return fn

    def per_site(self) -> Dict[str, float]:
        return {k: (v == 0).float().mean().item() for k, v in self.counts.items()}

    def overall(self) -> float:
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
    """Mean pairwise Jaccard overlap of penultimate active-unit sets, same- vs
    different-class, plus mean active-unit count.

    Logged, not argued from: Stage 1b falsified separation as the causal account on
    MNIST. Here so the same correlation can be checked on CIFAR-100 later. With 100
    classes the different-class pool dominates, so the absolute numbers are not
    comparable to the 10-class runs -- only the sign and trend are.
    """
    acts = []
    h = model.penultimate.register_forward_hook(
        lambda _m, _i, o: acts.append((o.detach() != 0).float().cpu()))
    model.eval()
    for i in range(0, idx.numel(), 256):
        b = idx[i:i + 256]
        xb, _ = make_batch(x_u8, y, b, mean, std, DEVICE)
        model(xb)
    h.remove()

    M = torch.cat(acts)
    M = M.flatten(1) if M.dim() > 2 else M
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


def train_one_run(model, x_tr, y_tr, sub_idx, val_idx, x_te, y_te, test_idx,
                  mean, std, seed, ov_idx):
    """Train to convergence, score the best-validation checkpoint.

    Identical in structure to `experiments/train_cifar_v1.train_one_run`; it is not
    imported because that function is bound to that script's DEVICE, its
    `make_batch`, and its constants. Duplicated deliberately rather than editing a
    script whose results are recorded (protocol rule 6).
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Dedicated batch generator, independent of model-init RNG, so every arm at a
    # given seed sees the IDENTICAL batch sequence. Without this the comparison is
    # not paired and the per-seed delta carries batch-order noise for free.
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
        xb, yb = make_batch(x_tr, y_tr, pick, mean, std, DEVICE)
        opt.zero_grad()
        criterion(model(xb), yb).backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            va = evaluate(model, x_tr, y_tr, val_idx, mean, std)
            if va > best_val:
                best_val, best_step, stale = va, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

    secs = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    counter = DeadUnitCounter(model)
    evaluate(model, x_tr, y_tr, val_idx, mean, std)
    dead_sites, dead_all = counter.per_site(), counter.overall()
    counter.remove()

    ov_same, ov_diff, active = code_overlap(model, x_tr, y_tr, ov_idx, mean, std)
    test_acc = evaluate(model, x_te, y_te, test_idx, mean, std)

    return dict(test_acc=test_acc, best_val=best_val, best_step=best_step,
                steps_run=step, hit_cap=(step >= MAX_STEPS), dead_sites=dead_sites,
                dead_all=dead_all, ov_same=ov_same, ov_diff=ov_diff,
                active=active, secs=secs)


# -------------------------------- reporting ----------------------------------

def _stats(v):
    n = len(v)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    se = (sd / math.sqrt(n)) if n > 1 else float("nan")
    return m, sd, se


def seeds_needed(sd: float, effect: float = 1.0) -> float:
    """Paired-sample n for 80% power at alpha=0.05 two-sided, for `effect` points.

    The MNIST noise floor was budget-dependent, so it must be measured per budget,
    never assumed. On CIFAR-100 at a near-chance base the paired sd is expected to
    be large, so this number matters more here than anywhere prior.
    """
    if sd != sd or effect <= 0:
        return float("nan")
    return math.ceil(7.849 * (sd / effect) ** 2)


def summarize(rows):
    print("\n" + "=" * 92)
    print("CIFAR-100 v1 SUMMARY -- paired per-seed delta vs dense, percentage points")
    print("=" * 92)

    verdict = {}
    for shots in sorted({r["shots_per_class"] for r in rows}):
        sub = [r for r in rows if r["shots_per_class"] == shots]
        dense = {r["seed"]: r["test_acc"] for r in sub if r["arm"] == "dense"}
        total = TOTAL_LABELS.get(shots, shots * NUM_CLASSES)
        print("\n{} shots/class ({} labels total)".format(shots, total))
        print("  {:<16}{:>10}{:>9}{:>7}{:>7}{:>10}{:>9}{:>8}".format(
            "arm", "test acc", "delta", "sd", "se", "signs", "n->1pt", "sep"))
        for arm, _ in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            accs = [r["test_acc"] for r in a]
            deltas = [(r["test_acc"] - dense[r["seed"]]) * 100
                      for r in a if r["seed"] in dense]
            if arm == "dense":
                m, sd, se, signs, need = 0.0, 0.0, 0.0, "", float("nan")
            else:
                m, sd, se = _stats(deltas)
                signs = "".join("+" if d > 0 else "-" for d in deltas)
                need = seeds_needed(sd)
            seps = [r["separation"] for r in a]
            print("  {:<16}{:>10.4f}{:>+9.2f}{:>7.2f}{:>7.2f}{:>10}{:>9}{:>8.3f}"
                  .format(arm, sum(accs) / len(accs), m, sd, se, signs,
                          "-" if need != need else int(need),
                          sum(seps) / len(seps)))
            if arm != "dense":
                verdict.setdefault(shots, {})[arm] = (m, sd, se, signs, len(deltas))

        print("  dead units per component (rule 8):")
        for arm, _ in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            print("    {:<16} conv1 {:>5.1f}%  conv2 {:>5.1f}%  fc {:>5.1f}%   "
                  "active(fc) {:>5.1f}/128".format(
                      arm,
                      100 * sum(r["dead_frac_conv1"] for r in a) / len(a),
                      100 * sum(r["dead_frac_conv2"] for r in a) / len(a),
                      100 * sum(r["dead_frac_fc"] for r in a) / len(a),
                      sum(r["active_units"] for r in a) / len(a)))

    # ---------------- the pre-registered verdict ----------------
    print("\n" + "-" * 92)
    print("PRE-REGISTERED VERDICT -- kwta_channel vs dense at 5 / 10 shots/class")
    print("(MNIST reference, same protocol, 8 seeds: +1.58 at 300, +1.66 at 600 labels;")
    print(" NOTE: those are label-count budgets, not directly comparable to per-class shots.)")

    key = [s for s in (5, 10) if s in verdict and "kwta_channel" in verdict[s]]
    if not key:
        print("  no 5/10-shot kwta cell present -- no verdict.")
        print("-" * 92)
        return

    fired = None
    for s in key:
        m, sd, se, signs, cnt = verdict[s]["kwta_channel"]
        need = seeds_needed(sd)
        cons = max(signs.count("+"), signs.count("-")) / max(1, len(signs))
        print("\n  {} shots/class ({} labels): delta {:+.2f} pts  sd {:.2f}  se {:.2f}  "
              "signs {}  ({:.0f}% consistent)".format(
                  s, TOTAL_LABELS.get(s, s * NUM_CLASSES), m, sd, se, signs, 100 * cons))
        print("            noise floor: {} seeds needed to resolve 1.0 pt at 80% "
              "power; {} run.".format("?" if need != need else int(need), cnt))
        powered = (need == need) and (cnt >= need)
        if m >= 1.0 and cons >= 0.8 and powered:
            fired = fired or "A"
        elif m <= -1.0 and cons >= 0.8 and powered:
            fired = "C" if fired in (None, "C") else fired

    ms = [verdict[s]["kwta_channel"][0] for s in key]
    sds = [verdict[s]["kwta_channel"][1] for s in key]
    cnts = [verdict[s]["kwta_channel"][4] for s in key]
    all_powered = all((seeds_needed(sd) == seeds_needed(sd)) and c >= seeds_needed(sd)
                      for sd, c in zip(sds, cnts))
    if fired is None and all(abs(m) < 0.5 for m in ms) and len(key) == 2 and all_powered:
        fired = "B"
    if fired is None:
        fired = "D"

    print("\n  BRANCH {} FIRED.".format(fired))
    if fired == "A":
        print("  REPLICATES. The effect survives a 10x harder task with 100 fine classes.")
        print("  The central claim is no longer plausibly a toy-dataset artifact. Next:")
        print("  sweep k on CIFAR-100, and compare against the pretrained arm.")
    elif fired == "B":
        print("  VANISHES. The effect does not generalize to a hard, many-class, near-")
        print("  chance regime. It was a property of moderate-accuracy datasets where")
        print("  sparsification acts as a cheap regularizer. This is the headline finding,")
        print("  not a footnote -- the gate is allowed to close.")
    elif fired == "C":
        print("  REVERSES. The mechanism actively costs accuracy: with 100 classes at a")
        print("  near-chance base, discarding 80% of channel activations throws away signal")
        print("  the net cannot spare. The brain-inspired framing should be dropped.")
    else:
        print("  UNDERPOWERED. Report the noise floor and the seeds required; conclude")
        print("  nothing. n = 3 is a direction check, never a significance test.")
        print("  NOTE: at a near-chance base a large paired sd is expected, so D is a")
        print("  legitimate pilot outcome, NOT a soft 'no effect'.")
    print("\n  Caveat registered in advance and still binding: CIFAR-100 at 5-25 shots/class")
    print("  is a near-chance (~5-15%) accuracy regime. 'Did not replicate at matched shots'")
    print("  and 'did not replicate in a near-chance regime' are not separable by this run.")
    print("  This is exactly the regime where a pretrained backbone should dominate.")
    print("-" * 92)


# ----------------------------------- main ------------------------------------

def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    commit = git_commit()
    tag = "cifar100_v1" + ("_quick" if QUICK else "_pilot" if PILOT else "")
    run_dir = os.path.join("experiments", "results",
                           time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(protocol=tag, dataset="cifar100", git_commit=commit,
               arms=[a for a, _ in ARMS], k=K, input_shape=list(INPUT_SHAPE),
               num_classes=NUM_CLASSES, shots_per_class=SHOTS_PER_CLASS, seeds=SEEDS,
               total_labels=[s * NUM_CLASSES for s in SHOTS_PER_CLASS],
               max_steps=MAX_STEPS, eval_every=EVAL_EVERY, patience=PATIENCE,
               batch_size=BATCH_SIZE, lr=LR, selection="best-on-validation",
               model_builder="scarce.nets.build_net", device=str(DEVICE),
               quick=QUICK, pilot=PILOT)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2), "\n")
    print("writing to {}".format(run_dir))
    print("{} runs planned\n".format(len(ARMS) * len(SHOTS_PER_CLASS) * len(SEEDS)))

    x_tr, y_tr, x_te, y_te = load_cifar100_tensors()
    pool_idx, val_idx = train_val_split(y_tr)
    assert_disjoint(pool_idx, val_idx)          # the check, actually called
    test_idx = torch.arange(x_te.shape[0])
    subsets = preflight(y_tr, pool_idx, val_idx, SHOTS_PER_CLASS, SEEDS)

    # Fixed evaluation subset for the overlap metric, drawn from validation only.
    ov_idx = _overlap_indices(y_tr, val_idx)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    rows = []
    for shots in SHOTS_PER_CLASS:
        for seed in SEEDS:
            sub_idx = subsets[(shots, seed)]
            mean, std = subset_norm_stats(x_tr, sub_idx)
            dense_acc = None
            for arm, factory in ARMS:
                set_seed(seed)                   # identical init across arms
                model = build_net(INPUT_SHAPE, NUM_CLASSES, factory)
                r = train_one_run(model, x_tr, y_tr, sub_idx, val_idx,
                                  x_te, y_te, test_idx, mean, std, seed, ov_idx)
                if arm == "dense":
                    dense_acc = r["test_acc"]
                d = r["dead_sites"]
                row = dict(
                    protocol=tag, git_commit=commit, device=str(DEVICE), arm=arm,
                    k=("" if arm == "dense" else K), shots_per_class=shots,
                    n_train_examples=sub_idx.numel(), seed=seed,
                    steps_run=r["steps_run"], hit_step_cap=int(r["hit_cap"]),
                    best_val_acc=round(r["best_val"], 4),
                    best_val_step=r["best_step"], test_acc=round(r["test_acc"], 4),
                    n_params=sum(p.numel() for p in model.parameters()),
                    active_units=round(r["active"], 1),
                    dead_frac_conv1=round(d.get("conv1", float("nan")), 4),
                    dead_frac_conv2=round(d.get("conv2", float("nan")), 4),
                    dead_frac_fc=round(d.get("fc", float("nan")), 4),
                    dead_frac_all=round(r["dead_all"], 4),
                    overlap_same=round(r["ov_same"], 4),
                    overlap_diff=round(r["ov_diff"], 4),
                    separation=round(r["ov_same"] - r["ov_diff"], 4),
                    train_seconds=round(r["secs"], 1))
                rows.append(row)
                with open(csv_path, "a", newline="") as f:   # append as we go
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                delta = "" if dense_acc is None else "({:+5.2f})".format(
                    (r["test_acc"] - dense_acc) * 100)
                print("  {:>4}/cls s={} {:<14} test={:.4f} {:<8} "
                      "val={:.4f}@{}/{}{}  dead c1/c2/fc={:.0f}/{:.0f}/{:.0f}%  "
                      "act={:4.1f}  {:5.1f}s".format(
                          shots, seed, arm, r["test_acc"], delta, r["best_val"],
                          r["best_step"], r["steps_run"],
                          " CAP" if r["hit_cap"] else "",
                          100 * d.get("conv1", float("nan")),
                          100 * d.get("conv2", float("nan")),
                          100 * d.get("fc", float("nan")),
                          r["active"], r["secs"]))
            print()

    summarize(rows)
    print("\nSaved to {}".format(csv_path))


def _overlap_indices(y_tr, val_idx):
    """A fixed, class-balanced slice of VALIDATION for the overlap metric.

    Drawn from val, never from train or test: it is a diagnostic on data the model
    selected against, which is the same choice the earlier runs made, so the
    numbers are comparable across datasets.
    """
    g = torch.Generator().manual_seed(999)
    labs = y_tr[val_idx]
    parts = []
    for c in range(NUM_CLASSES):
        cls = val_idx[labs == c]
        parts.append(cls[torch.randperm(cls.numel(), generator=g)[:OVERLAP_N_PER_CLASS]])
    return torch.cat(parts)


if __name__ == "__main__":
    main()
