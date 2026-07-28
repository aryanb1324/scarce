# Experiments Directory Skill File

## Purpose

This directory contains training scripts, run configs, logs, and results.
This is where architecture modules and data pipelines get combined into
an actual training run and compared against the baseline.

## When to Edit This Directory

Edit this directory when:
- Adding a new experiment config (new mechanism, new hyperparameter sweep)
- Changing the training loop itself (optimizer, scheduler, logging)
- Recording or analyzing results

Do not edit this directory when:
- The bug is inside a module's forward/backward pass (see
  `architecture/skills.md`)
- The bug is in dataset loading or preprocessing (see `data/skills.md`)

## Important Files

- `configs/baseline_*.yaml`: fixed baseline configs — see root skill
  file's "Dangerous Areas," do not edit casually
- `configs/`: one config per experiment, versioned, not overwritten
- `train.py` (or equivalent): the training entry point
- `results/`: logged metrics, ideally one subfolder per run with the
  config that produced it saved alongside the results

## Rules

- One config change = one experiment. Do not change multiple
  hyperparameters or mechanisms in a single config and call it one result.
- Every run must record: seed, data budget (exact dataset size / steps
  seen), full config, and git commit hash (or equivalent) so it's
  reproducible later.
- Every comparison between the new architecture and baseline must use
  identical data budgets, identical eval protocol, and multiple seeds
  before being treated as a real result.
- Never delete a run's config or logs, even for failed experiments — move
  to `results/archive/` if it's cluttering the main results view.
- Report variance across seeds, not just a best-seed number.

## Common Mistakes

- Mistake: Comparing a new mechanism's "best epoch" against the
  baseline's "final epoch" (or similarly mismatched comparison points).
  - Why it happens: Convenient to grab whatever number looks best.
  - How to avoid it: Fix the comparison protocol (e.g. always final epoch,
    or always best-on-val) before running any experiments, and never
    change it after seeing results.

- Mistake: A promising early result turns out to be a single lucky seed.
  - Why it happens: Running one seed is faster, and it's tempting to
    report it as a finding.
  - How to avoid it: Treat single-seed results as provisional; require at
    least 3 seeds before writing a result down as a conclusion.

## Debugging Playbook

When an experiment run misbehaves:

1. Confirm the config matches what you intended (log it, don't assume).
2. Check the loss curve for the whole run, not just the final metric.
3. Confirm the data budget actually matches the baseline's data budget.
4. If results look implausibly good, check for data leakage first (see
   `data/skills.md`) before believing the architecture is responsible.
5. Re-run with a different seed before trusting a surprising result.

## Validation

After adding or editing an experiment:
```bash
# fill in, e.g.:
# python experiments/train.py --config experiments/configs/smoke_test.yaml
```
Run a short smoke-test config first to catch crashes before committing to
a full-length run.

## Self-Improvement Rule

If a bug or lesson is specific to running/comparing experiments (protocol
mismatches, logging gaps, reproducibility issues), record it here.

## Lessons

### 2026-07-27 — Protocol v1 and v2, and why both exist

**v1** (`train.py`): fixed 5 epochs at every data fraction, final-epoch test
accuracy, no validation split, `results.csv` overwritten each run. Its numbers are
kept and are still the reference point for "what the naive protocol says," but
they cannot support a data-efficiency claim — see the root `skills.md` lesson on
fixed epochs.

**v2** (`train_v2.py`): every run trains until validation accuracy stops improving
(patience-based early stopping, capped at MAX_STEPS) and is scored at its
best-validation checkpoint, so neither model can be scored mid-convergence
whatever its data budget. Plus stratified sampling, subset-computed
normalization, a frozen val split, per-run result directory with config and
commit hash, and three diagnostics (convergence speed, dead units, code overlap).

Note the design choice: a *fixed step count* would also remove v1's confound, but
only by equalizing it — both models could still be scored before finishing.
Training each to convergence is what actually answers "how well can this model do
with this much data," and it converts the confound into a measurement, since
`best_val_step` is then the convergence-speed difference itself.

**Rule:** protocol changes get a version number and a new entry point; they never
edit the existing one in place. Two protocols coexisting is correct — silently
migrating one is what makes historical results incomparable.

### 2026-07-27 — Run the cheapest condition first

v1 ordered `DATA_FRACTIONS = [1.0, 0.5, ...]` and printed only after all 5 epochs
of a run finished, so the first line of output arrived after ~4,690 optimizer
steps — about ten minutes of staring at nothing, with no way to tell a slow run
from a hung one. The four `100.0%` lines that appear first are torchvision's
download progress bars, not results, which makes the silence more confusing.

**Rule:** order sweeps cheapest-condition-first and print incrementally (per
epoch, or per eval). Append each result to the CSV as it completes rather than
writing everything at the end, so an interrupted run keeps its completed work.
This is free — `set_seed` is called at the top of every (fraction, seed, model)
cell and the subset sampler uses its own seeded generator, so each cell is
independent of iteration order and reordering cannot change a result.

### 2026-07-27 — Pre-register the prediction in the script

`train_v2.py` states, in its module docstring and before any run, which outcome
supports which hypothesis and what the decision rule is. Written after seeing the
numbers, any such statement is a rationalization; written before, it is an
experiment.

**Rule:** the hypothesis, the comparison point, and the decision rule go in the
run's config or script docstring before it executes, and are not edited
afterwards.
