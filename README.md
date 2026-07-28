# Brain-Inspired, Data-Efficient Architecture — Starter Project

## What this is

A minimal, runnable first experiment: a standard dense CNN (ReLU) vs. the
same CNN with a k-Winner-Take-All (kWTA) sparse activation swapped in,
trained on shrinking fractions of MNIST, to test whether forced sparsity
improves generalization when data is scarce.

This is deliberately small in scope — the point is to get one clean,
complete result end to end (idea → code → experiment → plot) before
adding more mechanisms or more complexity.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

```bash
# 1. Run the unit tests first — confirms the kWTA module behaves correctly
#    on toy inputs before you spend time on a real training run.
python -m pytest tests/ -v

# 2. Run the actual experiment (takes a few minutes on CPU for MNIST).
#    Downloads MNIST automatically on first run (needs internet).
python -m experiments.train

# 3. Plot accuracy vs. data fraction for both models.
python -m experiments.plot_results
```

Results land in `experiments/results.csv` and
`experiments/data_efficiency_curve.png`.

## What to look at

The plot shows test accuracy (y-axis) vs. fraction of training data used
(x-axis, log scale), one line per model, with error bars across 3 seeds.
If the kWTA line sits above the dense baseline line — especially at the
smaller data fractions (5%, 1%) — that's a real first signal worth
digging into further. If the lines are on top of each other, that's a
useful negative result too: it tells you kWTA alone (at this k value,
this depth, this dataset) isn't the effect, and you can move to the next
idea with real information instead of a guess.

## Where to go from here

- Try different values of `k` in `architecture/model.py`'s
  `build_kwta_model` (currently 0.2 = keep top 20% of units).
- Try a harder dataset (CIFAR-10) once MNIST results are in.
- Read `skills.md` (and the directory-level ones in `architecture/`,
  `data/`, `experiments/`, `tests/`) before making changes — and update
  them as you learn things. That's the whole point of this structure: it
  turns one-off lessons into permanent project knowledge instead of
  things you have to rediscover next week.
