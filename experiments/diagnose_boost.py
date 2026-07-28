"""
Why did duty-cycle boosting destroy accuracy? A 5-minute diagnostic.

Stage 1 result: boosting did exactly what it was designed to do and made the model
much worse anyway.

    600 labels        dead units      separation     accuracy vs dense
    kwta_channel        43.5%           0.181            +1.43
    + boost=1.0         41.5%           0.177            -8.61   (spread 10.4 pts)
    kwta_global         46.8%           0.133            -1.58
    + boost=1.0         43.0%           0.157            -7.62   (spread 25.8 pts)

Fewer dead units, comparable-or-better separation, catastrophic accuracy, and a
seed spread of 10-26 points where the unboosted arms sit inside 0.3-0.8. That
signature -- enormous variance rather than smooth degradation -- points at a
train/eval inconsistency, not at a bad idea.

LEADING HYPOTHESIS: boosting is applied only in train() mode (as Numenta's is, and
as `_boosting` enforces), so during training the winner set is chosen from
boosted scores while at evaluation it is chosen from raw activations. With the
exponent clamped at 3 the ranking can be distorted by up to 20x, so the two winner
sets can diverge badly -- and the network is then evaluated on a representation it
was never trained to produce. Best-on-validation selection cannot rescue this,
because every checkpoint has the same mismatch.

This script measures the mismatch directly instead of assuming it: after training,
it runs the same validation batch through the model twice -- once in train() mode,
once in eval() -- and reports the Jaccard agreement between the two winner sets,
per layer. Boosting is the only thing that differs between the modes (SmallCNN has
no BatchNorm or dropout), so any disagreement is attributable to it.

READ IT AS:
  * agreement falls steeply with beta, tracking the accuracy loss
      -> the mismatch is the cause. Fixes worth trying, cheapest first:
         (a) much smaller beta (0.1-0.25) so ranking is nudged, not overturned;
         (b) anneal beta -> 0 over training so train and eval converge by the end;
         (c) apply boosting in eval too -- consistent, but makes inference depend
             on training-time statistics, which is a real cost.
  * agreement stays high (>0.9) while accuracy still collapses
      -> the mismatch is NOT the cause, and boosting is damaging the learned
         features themselves. Drop it; the axis result stands on its own.

This is a diagnostic, not an experiment: 1 seed, no claims. Its output decides
whether boosting is worth another run at all.

Run with:
    python -m experiments.diagnose_boost      # ~5 minutes
"""

import os

import torch

import experiments.train_v2 as tv2
from architecture.model import build_kwta_v2_model
from architecture.modules.kwta import KWinnersTakeAll
from data.mnist_v2 import (
    load_mnist_tensors,
    make_batch,
    stratified_subset,
    subset_norm_stats,
    train_val_split,
)

BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]
SHOTS = 60          # 600 labels -- the budget where the axis effect lives
SEED = 0

LAYER_NAMES = {"features.1.1": "conv1", "features.4.1": "conv2", "classifier.2.1": "fc"}


@torch.no_grad()
def winner_agreement(model, xb):
    """
    Jaccard overlap between the train-mode and eval-mode winner sets on the same
    input, per kWTA layer. 1.0 = boosting changes nothing at inference.
    """
    caught = {}

    def grab(name, mode):
        def fn(_m, _i, out):
            caught[(name, mode)] = (out.detach() != 0)
        return fn

    for mode in ("train", "eval"):
        getattr(model, mode)()
        handles = [mod.register_forward_hook(grab(name, mode))
                   for name, mod in model.named_modules()
                   if isinstance(mod, KWinnersTakeAll)]
        model(xb)
        for h in handles:
            h.remove()

    out = {}
    for name, _ in {k[0]: None for k in caught}.items():
        a, b = caught[(name, "train")], caught[(name, "eval")]
        inter = (a & b).sum().item()
        union = (a | b).sum().item()
        out[LAYER_NAMES.get(name, name)] = round(inter / max(union, 1), 3)
    return out


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tv2.MAX_STEPS, tv2.EVAL_EVERY, tv2.PATIENCE = 4000, 100, 12

    x_tr, y_tr, x_te, y_te = load_mnist_tensors()
    pool_idx, val_idx = train_val_split(x_tr.shape[0])
    test_idx = torch.arange(x_te.shape[0])
    sub_idx = stratified_subset(y_tr, pool_idx, SHOTS, SEED)
    mean, std = subset_norm_stats(x_tr, sub_idx)
    probe, _ = make_batch(x_tr, y_tr, val_idx[:256], mean, std, tv2.DEVICE)

    print(f"kwta_channel, k=0.2, {SHOTS * 10} labels, seed {SEED}")
    print(f"{'beta':>6}{'test acc':>10}{'dead':>8}{'sep':>8}   train/eval winner agreement")
    print("-" * 78)

    for beta in BETAS:
        tv2.set_seed(SEED)
        model = build_kwta_v2_model(dim="channel", boost=beta)
        r = tv2.train_one_run(model, x_tr, y_tr, sub_idx, val_idx, x_te, y_te,
                              test_idx, mean, std, SEED)
        agree = winner_agreement(model, probe)
        print(f"{beta:>6.2f}{r['test_acc']:>10.4f}{r['dead'] * 100:>7.1f}%"
              f"{r['ov_same'] - r['ov_diff']:>8.3f}   {agree}")

    print("-" * 78)
    print("Agreement falling with beta, tracking the accuracy loss -> train/eval")
    print("mismatch is the cause; retry with beta 0.1-0.25 or anneal it to 0.")
    print("Agreement staying high while accuracy collapses -> boosting is damaging")
    print("the features themselves; drop it, the axis result stands alone.")


if __name__ == "__main__":
    main()
