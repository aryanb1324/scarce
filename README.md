# scarce

[![tests](https://github.com/aryanb1324/scarce/actions/workflows/ci.yml/badge.svg)](https://github.com/aryanb1324/scarce/actions/workflows/ci.yml)

**Find out which architecture actually helps on your low-data problem — instead of trusting one that helped on someone else's.**

```python
import scarce

result = scarce.fit(x_train, y_train, x_test=x_test, y_test=y_test)
print(result.report())
predictions = result.predict(x_new)
```

`fit` trains every candidate mechanism on *your* data under a protocol designed
not to fool you, measures the noise floor on *your* data, and names a winner only
if the effect clears it. When nothing does, it tells you to use a standard
network — and hands you one.

---

## The claim this library does *not* make

It does not ship a magic data-efficient architecture. It ships the measuring.

The sparse mechanism this project was built to investigate (channel-wise kWTA)
looked good on MNIST — and then **failed its own pre-registered gate on natural
images.** Both results, paired, same protocol:

| dataset | 300 labels | 600 labels |
|---|---|---|
| MNIST (8 seeds) | **+1.58** | **+1.66** |
| CIFAR-10 (3 seeds) | **−1.57** | **−1.61** |

On MNIST it helps by ~1.6 points. On CIFAR-10 it *costs* ~1.6 points, with every
seed agreeing in sign at both budgets. Ordinary `Dropout2d` does worse still on
CIFAR (−6.3 / −9.5). The most likely reading is that channel-wise competition
discards information natural images carry — colour, texture, low-contrast
background structure — that MNIST simply does not contain.

That outcome was written down as a falsifiable branch *before* the run, and it
fired. It is reported here as the headline rather than a footnote, because the
whole point of pre-registering a gate is that it is allowed to close.

Two caveats, both registered in advance and still binding: the CIFAR runs are
**3 seeds** — at 600 labels the reversal is resolved (−1.61 at sd 0.38, ~7× its
standard error), but at 300 labels sd is 0.88, so 3 seeds cannot resolve a 1-point
effect there and the direction is suggestive rather than settled. And CIFAR at
these budgets is a ~33–38% accuracy regime where MNIST was at 90%+, so *"reverses
at matched label count"* and *"reverses in a low-accuracy regime"* are not
separable by this run.

**This is exactly why the library searches instead of asserting.** A mechanism
that reverses sign between two datasets is not a prior you should ship as a
default — it is a hypothesis you should *test on your own data*. That is what
`fit` does, and it is why the tool remains useful even though the mechanism that
motivated it did not generalize.

(The mechanism isn't novel either — channel-wise kWTA is `local=True` in
`nupic.torch`.)

## Install

No clone needed — pip fetches it directly:

```bash
pip install git+https://github.com/aryanb1324/scarce.git
```

Two things to know:

- **Use a recent pip** (≥ 23, with setuptools ≥ 61). An old pip — e.g. the macOS
  system pip 21.x — silently builds a broken package named `UNKNOWN-0.0.0` because
  it can't read the project metadata. Run `pip install --upgrade pip` first if
  unsure.
- **On PyPI:** publishing is wired up (Trusted Publishing, fires on a GitHub
  Release), but **no release has been cut yet**. Until one is, plain
  `pip install scarce` will not get this package — use the `git+https` form
  above. Once the first release lands, `pip install scarce` works and this note
  should be deleted.

To hack on it instead, clone and install editable:

```bash
git clone https://github.com/aryanb1324/scarce.git && cd scarce
pip install -e .
```

Depends only on `torch` and `numpy` — the statistics are implemented directly, so
installing this doesn't drag in scipy. Installing adds only the `scarce` package to
your environment (nothing generic like `data` or `architecture`).

## What you get back

Real output, 600 MNIST labels, 5 seeds:

```
==========================================================================
SCARCE SEARCH -- paired delta vs dense, percentage points
==========================================================================
600 training / 5000 validation examples, 5 seeds

  arm                   val acc    delta     sd       p     signs
  dense                  0.9203       --     --      --(reference)
  kwta_channel_k0.2      0.9373    +1.70   0.63  0.0039     +++++
  kwta_channel_k0.1      0.9339    +1.36   1.07  0.0462     -++++
  dropout_k0.2           0.9320    +1.18   0.55  0.0086     +++++

COST (what each mechanism spends, not just what it buys)
  arm                   active units     dead frac     sec/run
  dense                         79.4         0.294        32.6
  kwta_channel_k0.2             26.0         0.597        56.7
  kwta_channel_k0.1             13.0         0.720        69.1
  dropout_k0.2                  41.3         0.002        94.0

--------------------------------------------------------------------------
WINNER: kwta_channel_k0.2
  +1.70 pts vs dense, p=0.0039 < 0.0167 (Bonferroni), 5/5 seeds agree.

NOISE FLOOR on this data: paired sd ~0.75 pts across 5 seeds,
  so 5 seeds can resolve effects down to ~0.94 pts at 80% power.

TEST ACCURACY (unbiased): 0.9470
--------------------------------------------------------------------------
```

Two things to notice.

**The cost column.** `kwta_channel` buys its 1.70 points with 60% of its units
permanently dead, and `k0.1` with 72%. A mechanism can hit its stated goal and
still be the wrong trade; you should be able to see the bill.

**`kwta_channel_k0.1` does not win despite p = 0.046.** Four arms are tested
against one reference, so the threshold is Bonferroni-corrected to 0.0167 and
that arm — which also has a seed disagreeing — correctly fails.

At **300** labels the same search returns **`dropout_k0.2`** instead. kWTA is
nominally ahead there (+1.91 vs +1.68) but the head-to-head paired difference is
p = 0.52 with two of five seeds favouring dropout — an unresolvable margin. The
rule breaks such ties toward the simpler, better-understood component rather than
selling you the exotic one on noise. It would be very easy for this library to
flatter its own mechanism, and that is precisely what the rule prevents.

## Why the protocol matters more than the mechanism

Every rule in `scarce/protocol.py` exists because breaking it produced a wrong
answer in this project's own history:

- **Train to convergence, never fixed epochs.** Holding epochs fixed while the
  training set shrinks makes dataset size and optimization budget the same
  variable. Here it manufactured a 5.4-point deficit out of a 1.6-point one *and
  inverted the sign of the conclusion*. Sparse nets update ~a fifth of the weights
  per step, so they converge slower — and the naive harness gave them the fewest
  steps exactly where they needed the most.
- **Paired seeds on identical batches.** Both arms see the same batches in the
  same order at a given seed. Reporting two independent means instead throws away
  most of the statistical power.
- **Normalization fitted on training data only.** Statistics computed over held-out
  data leak, and inflate apparent data efficiency.
- **Refuse to call a winner inside the noise.** A 5-seed run here reported a
  −0.79 pt effect that 20 seeds revealed as −0.06 ± 0.31. The first five seeds
  reproduced the original number exactly — it was seed luck. The decision rule
  reports how many seeds *would* have been needed rather than guessing.
- **Bonferroni-correct across arms.** Testing four mechanisms against one
  reference inflates false positives; the threshold is split accordingly.
- **Break statistical ties toward the simpler component.** Winning on a margin
  smaller than the noise is not winning. Arms whose head-to-head paired
  difference is unresolvable are reported as tied, and the simplest one is
  returned.

Reproduction: the library reproduces the recorded research numbers through a
completely separate, shape-adaptive implementation — **+1.70 vs +1.66** at 600
labels and **+1.91 vs +1.58** at 300, both within the paired standard error.
Run `python -m examples.mnist_low_data` to check it yourself.

### What the sparse mechanism is actually worth (on MNIST)

An early control suggested the input-dependent competition in kWTA beat a
same-sparsity random baseline by ~13 points. That control was confounded: it
matched kWTA on *nominal* selected count but kept ~3.4× fewer live units (random
selection lands on post-ReLU zeros; top-k can't). A properly sparsity-matched
control (Stage 2b, 8 seeds, paired) settles it:

- The competition does do **real work** — kWTA beats the matched random control by
  **+2.08 pts at 300 labels and +2.42 at 600, 8/8 seeds both** (p ≈ 0.003 / 0.0003).
- But it's **~2 points, not ~13**. About 84% of that old headline was a pure
  sparsity artifact, not competition.
- And **~2 pts is an upper bound**: the matched control hit the training-step cap
  (still improving when scored), so part of its deficit is undertraining, not
  random selection. A higher-cap rerun is the outstanding task.

**Read all of that as MNIST-scoped.** "The competition is doing something" and
"the competition is useful" are different claims, and CIFAR-10 answers the second
one *no*: the same mechanism that beats a random control by 2 points on MNIST
loses 1.6 points to a plain dense net on natural images.

The honest headline of this project is the *method* — four self-caught confounds
and one closed gate — not the size of the effect.

## API

```python
scarce.fit(
    x_train, y_train,          # (N,C,H,W) images or (N,F) tabular; numpy or torch
    x_val=None, y_val=None,    # omitted -> stratified split held out of train
    x_test=None, y_test=None,  # recommended: the only unbiased final number
    seeds=5,                   # below 5, the rule will usually decline to call one
    candidates=None,           # override the mechanism arms; must include 'dense'
    architectures=None,        # None: mechanisms only. "default"/"full"/<seq>: also
                               #   search capacity, depth, and a linear baseline
    budget=None,               # None: run every arm at `seeds`. "quick"/"standard"/
                               #   "thorough": two-stage screen-then-confirm search
    include_controls=False,    # add diagnostic arms (randk)
    config=TrainConfig(),      # max_steps, patience, lr, batch_size
)
```

`SearchResult` gives you `.winner`, `.model`, `.predict(x)`, `.arms` (per-arm
paired statistics), `.cost`, `.test_acc`, and `.report()`.

### Searching architecture, not just activation

By default `fit` varies only the activation on one fixed CNN. At a few hundred
labels, capacity is usually the *dominant* knob — a smaller network, or plain
logistic regression, often wins — so `architectures=` opens that axis:

```python
scarce.fit(x, y, architectures="default", budget="standard")
```

`"default"` is a curated 12-arm space: five architectures (linear, narrow CNN,
shallow CNN, the default CNN, wide CNN) crossed with the mechanisms. `"full"`
adds the pretrained backbone below.

### The pretrained baseline — probably the strongest arm here

For a few hundred *natural* images, the honestly-best move is usually not any
mechanism in this library: it is a pretrained backbone. `"full"` includes one, as
a frozen ResNet-18 linear probe (backbone weights frozen and run under `no_grad`;
only a fresh linear head trains):

```bash
pip install "scarce[pretrained]"     # needs torchvision
```

```python
scarce.fit(x, y, architectures="full", budget="standard")
```

Three deliberate choices, so it can't quietly flatter itself:

- **It is opt-in, not in `"default"`.** It pulls an optional dependency and a
  ~45 MB weights download; `"default"` stays lightweight, torchvision-free and
  offline.
- **It carries the *highest* complexity rank**, so on an unresolvable tie it
  *loses* to a small CNN or logistic regression. It has to strictly win to be
  recommended.
- **It is never paired with a mechanism** (`supports_mechanism=False`): a frozen
  backbone has no activation site for kWTA or dropout, so an arm named
  `pretrained/kwta` would misdescribe what was measured.

Cost, stated plainly: **~26 s/run vs ~0.6 s** for the small CNNs, because the
frozen backbone re-runs every training step. Caching its features once (they never
change) is the obvious optimization and is not implemented yet.

A tool that cannot lose to a pretrained baseline isn't measuring — it's
advertising. This is the arm that lets it lose.

Crossing axes multiplies training runs, so `budget` switches on a two-stage
**screen-then-confirm** search: rank every arm on a few cheap seeds, then confirm
only the survivors on *fresh, disjoint* seeds — and decide on those alone, so
selection can't contaminate the test. Run counts on the 12-arm space:

| budget | runs | conclusion strength |
|---|---|---|
| `quick` | 39 | 5-seed |
| `standard` | 68 | 8-seed |
| `thorough` | 160 | 20-seed |

(Naive all-arms-at-full-seeds would be 60 / 96 / 240.)

Adding a mechanism is one entry — an activation factory — in
`scarce/mechanisms.py`; adding an architecture is one entry in
`scarce/architectures.py`. Either is then measured against the reference under
the same protocol as everything else.

## Worked example

```bash
python -m examples.mnist_low_data          # reproduction check, ~30-40 min CPU
QUICK=1 python -m examples.mnist_low_data  # ~3 min
```

The `QUICK` run deliberately uses a budget too short to converge, so kWTA looks
bad. That's not a bug — it's a live demonstration of the fixed-epoch confound the
protocol exists to prevent.

## Development & verification

```bash
git clone https://github.com/aryanb1324/scarce.git && cd scarce
pip install -e ".[dev]"
pytest -q
```

The suite is **185 tests** and runs in ~55s on CPU — unit tests on toy tensors
plus end-to-end `fit()` runs; no dataset downloads, no network (the pretrained
tests build a real ResNet-18 with `weights=None`, so they never fetch weights).
CI runs it on Python 3.9 on every push (badge up top).

The install path itself is verified, not assumed. Each of these was run against a
clean environment:

- the whole tree byte-compiles on Python 3.9 (the target runtime);
- `python -m build` produces a wheel + sdist that pass `twine check`, and the
  wheel's `top_level.txt` is exactly `scarce` — installing it adds nothing generic
  like `data` or `architecture` to your environment;
- the built wheel installs into a fresh venv and both `fit()` paths run;
- `pip install git+https://github.com/aryanb1324/scarce.git` — the command above —
  installs cleanly from this public repo.

## Honest limitations

- **The built-in mechanisms are candidates, not recommendations.** kWTA helps on
  MNIST and *hurts* on CIFAR-10 (see the top of this README). Treat every arm as
  something to measure on your data, never as something known to help. The one
  arm with a strong prior in its favour is the pretrained backbone — and it is
  opt-in precisely so the tool doesn't assume your problem looks like ImageNet.
- **CIFAR-10 evidence is a 3-seed pilot, not a full sweep.** Resolved at 600
  labels, underpowered at 300. A 5-seed run is the outstanding task.
- **CIFAR-100 is scaffolded but unrun.** The pipeline and a pre-registered
  experiment exist (`experiments/train_cifar100_v1.py`) and its data-handling is
  unit-tested, but the end-to-end run has not completed — the dataset host was
  throttling to ~2.5 KB/s when it was attempted. No CIFAR-100 numbers are claimed.
- **The competition effect on MNIST is ~2 points and an upper bound.** A
  sparsity-matched control (Stage 2b) shows input-dependent selection beats a
  random-selection baseline by ~2 points, 8/8 seeds — not the ~13 an earlier
  confounded control suggested, and even the ~2 is inflated by that control
  hitting the step cap. MNIST only. See `architecture/skills.md`.
- **Search cost.** `arms × seeds` full training runs. Defaults are 4 × 5 = 20.
  That is the price of an answer you can trust; `TrainConfig` and `candidates`
  let you trade it down.
- **Small validation sets cap resolution.** With 200 val examples, accuracy is
  quantized to 0.5 pts and sub-point effects are unresolvable. The report warns.

## The research underneath

This library is the shippable part of an ongoing project on brain-inspired
structural priors for data efficiency. The experiments, the falsified
predictions, and the accumulated methodology live in:

- `PLAN.md` — audit and staged mechanism plan
- `skills.md` + one per directory — the lessons, each tied to the mistake that
  taught it. This is the project's central discipline.
- `experiments/results/<timestamp>_<name>/` — every run, with config and commit

Four pre-registered predictions have been falsified so far — including the
central one, when CIFAR-10 reversed the MNIST effect — and one recorded finding
was retracted by a power run after 20 seeds contradicted 5. Every one of those is
still in the repo with the run that produced it.

That is the process working, and it is the actual product. The mechanism was the
hypothesis; the protocol is what survived.

## License

MIT.
