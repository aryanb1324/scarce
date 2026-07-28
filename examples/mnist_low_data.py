"""
Worked example, and the library's reproduction check.

Runs `aryanet.fit` on MNIST at 300 and 600 labels -- the budgets where the
research measured channel-wise kWTA at +1.58 and +1.66 points over dense (8
seeds, paired, experiments/results/20260728_151333_stage2_control).

It doubles as a regression test on the library layer. `aryanet` reimplements the
protocol against a shape-adaptive network rather than the frozen `SmallCNN`, so
if the library's numbers do not land near the recorded ones, the library is
wrong -- not the record.

Two honest caveats on what "reproduction" means here:

  * The research scored TEST accuracy at the best-validation checkpoint. `fit`
    selects on validation, so the deltas are comparable but not identical.
  * `build_net` pools conditionally where `SmallCNN` pools unconditionally. At
    28x28 that is a no-op, so the architectures match on this data.

    python -m examples.mnist_low_data           # ~30-40 min on CPU
    QUICK=1 python -m examples.mnist_low_data   # ~3 min, budget too short to
                                                #   converge -- expect kWTA to
                                                #   look bad, which is the
                                                #   fixed-epoch confound itself
"""

import os

import torch

import aryanet
from aryanet import TrainConfig
from data.mnist_v2 import load_mnist_tensors, stratified_subset, train_val_split

QUICK = os.environ.get("QUICK") == "1"

RECORDED = {300: +1.58, 600: +1.66}   # kwta_channel k=0.2, 8 seeds, paired
SEEDS = 5 if not QUICK else 2
CONFIG = TrainConfig() if not QUICK else TrainConfig(max_steps=300, eval_every=50,
                                                     patience=3)


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    x_tr, y_tr, x_te, y_te = load_mnist_tensors()
    pool_idx, val_idx = train_val_split(x_tr.shape[0])

    x_val, y_val = x_tr[val_idx].unsqueeze(1), y_tr[val_idx]
    x_test, y_test = x_te.unsqueeze(1), y_te

    for shots in (30, 60):
        n_labels = shots * 10
        sub = stratified_subset(y_tr, pool_idx, shots, seed=0)
        print("\n" + "#" * 74)
        print("# {} labels ({}/class)   recorded kwta delta: {:+.2f} pts".format(
            n_labels, shots, RECORDED[n_labels]))
        print("#" * 74 + "\n")

        result = aryanet.fit(
            x_tr[sub].unsqueeze(1), y_tr[sub],
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            seeds=SEEDS, config=CONFIG, verbose=True)

        kwta = next((a for a in result.arms if a.name == "kwta_channel_k0.2"), None)
        if kwta is not None:
            gap = kwta.mean - RECORDED[n_labels]
            print("\nREPRODUCTION at {} labels: library {:+.2f} vs recorded {:+.2f} "
                  "(diff {:+.2f} pts)".format(
                      n_labels, kwta.mean, RECORDED[n_labels], gap))
            if QUICK:
                print("  QUICK budget cannot converge; a negative delta here is the "
                      "fixed-epoch confound, not a contradiction.")


if __name__ == "__main__":
    main()
