# Data Directory Skill File

## Purpose

This directory contains dataset loading, preprocessing, and the low-data
/ few-shot sampling logic. Since the whole project's claim rests on data
efficiency, this is one of the most sensitive parts of the codebase —
a subtle bug here can invalidate every result.

## When to Edit This Directory

Edit this directory when:
- Adding or changing a dataset loader
- Changing preprocessing/augmentation
- Changing how a reduced-data subset is sampled for a low-data experiment

Do not edit this directory when:
- The issue is actually in the model (see `architecture/skills.md`)
- The issue is in the training loop itself (see `experiments/skills.md`)

## Important Files

- `datasets/`: dataset classes / loaders
- `sampling.py` (or equivalent): logic for drawing reduced-size training
  subsets — this directly implements the "less data" side of the project
- `splits/`: fixed, version-controlled train/val/test split definitions

## Rules

- Train/val/test splits must be fixed and version-controlled, not
  regenerated randomly on each run — a shifting split makes results
  incomparable across experiments.
- When sampling a reduced-data subset, sample from the train split only,
  and record exactly which examples (or the seed used to pick them) went
  into each subset size, so a given "10% of training data" run is
  reproducible.
- Check explicitly for leakage between train/val/test after any change
  here — e.g. shared examples, near-duplicates, or preprocessing that
  peeks at statistics computed over the full dataset instead of train-only.
- Any normalization statistics (mean/std, tokenizer vocab, etc.) must be
  computed from the training subset being used in that experiment, not
  from the full dataset, when testing low-data conditions.

## Common Mistakes

- Mistake: Preprocessing statistics (normalization, vocab) are computed
  once from the full dataset and reused across all low-data experiments.
  - Why it happens: Convenient to compute once and cache.
  - How to avoid it: This leaks information from the excluded data into
    the low-data run, inflating apparent data efficiency. Recompute (or
    explicitly justify not recomputing) per subset size.

- Mistake: The "reduced" training subset accidentally overlaps with val
  or test after a refactor of the sampling logic.
  - Why it happens: Sampling logic changed without re-verifying split
    disjointness.
  - How to avoid it: Add an automated check (e.g. in `tests/`) that
    asserts zero overlap between splits, and run it after any change here.

## Debugging Playbook

When results look suspicious or a dataset bug is suspected:

1. Log and manually inspect a batch — actual shapes, actual label values.
2. Verify split disjointness programmatically (don't eyeball it).
3. Verify the low-data subset size matches what the config claims.
4. Check whether any statistic used in preprocessing was computed on data
   outside the intended training subset.

## Validation

After editing this directory, run:
```bash
# fill in, e.g.:
# python -m pytest tests/ -k "data"
```

## Self-Improvement Rule

If a leakage bug, sampling bug, or preprocessing lesson is found here,
record it — this is the highest-stakes directory for silent correctness
bugs in the whole project.

## Lessons

### 2026-07-27 — v1 shipped all three of this file's own warnings

Writing the rules down did not prevent them. `mnist.py` as first written:

1. **Hardcoded full-dataset normalization** — `Normalize((0.1307,), (0.3081,))`
   are full-MNIST-train statistics, applied unchanged in the 1% condition. This is
   verbatim the first entry under "Common Mistakes" above. Effect size on MNIST is
   small; on CIFAR it will not be.
2. **No validation split at all** — the test set was the only eval signal, so it
   was being consumed a little with every idea tried.
3. **Unstratified sampling** — a uniform permutation, so at 600 examples per-class
   counts varied enough across seeds to rival the effect being measured.

And `assert_no_overlap` existed but was never called from anywhere, so the
directory the file itself calls "the highest-stakes for silent correctness bugs"
had zero test coverage.

**Rule:** a rule in a skills file is not in force until something executes it.
Every constraint in this file gets a test in `tests/` that fails when the
constraint is violated — see `tests/test_sampling.py`, which asserts split
disjointness, exact class balance, seed reproducibility, that an oversized request
raises instead of silently shrinking the real data budget, that normalization
statistics come from the subset rather than the full set, and that the leakage
assertion itself actually fires.

**Rule:** a sampler must **raise** when it cannot honor the requested budget,
never truncate. A silently-shrunk subset makes the run's real data budget differ
from the config's claimed budget, which is invisible in the results and
invalidates the comparison.

**Constraint worth remembering:** stratified sampling is capped by the rarest
class. MNIST train's rarest digit is 5 (5,421 examples); after the frozen 5,000
validation split that leaves ~4,969, so ~4,800/class is the practical ceiling.

See `mnist_v2.py` for the corrected pipeline. `mnist.py` is left untouched so v1
results stay reproducible.
