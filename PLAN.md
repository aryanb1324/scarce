# scarce — Audit and Staged Plan

**Date:** 2026-07-27
**Status of baseline:** not yet measured in this session (sandbox has no PyTorch
and no network to PyPI or the MNIST mirrors — all of `pypi.org`,
`files.pythonhosted.org`, `github.com`, and `ossci-datasets.s3.amazonaws.com`
return 403/timeout). Aryan is running `pytest tests/ -v` and
`python -m experiments.train` locally; every number below is a prediction to be
checked against that run, not a measurement.

---

## 0. Executive summary

The scaffolding is good — the separation of `architecture/` / `data/` /
`experiments/` / `tests/` and the discipline encoded in the `skills.md` files is
better than most research repos start with. The problem is not the structure.

There are two independent problems:

1. **The harness will probably produce a null result regardless of whether
   sparsity helps.** Five confounds (§1) mean the current experiment measures
   "accuracy under a shared, badly-matched optimization budget," not data
   efficiency. The dense-vs-kWTA comparison *within* a fraction is fair; the
   curve *across* fractions is not interpretable, and both models are
   badly undertrained exactly where the interesting signal lives.
2. **The kWTA module, as implemented, is not the mechanism the docstring
   describes.** It does global top-k over flattened channel×space, has no
   duty-cycle boosting, and therefore behaves closer to "harsh saliency
   masking plus a dead-unit generator" than to cortical lateral inhibition
   (§2). This is the most likely reason the first experiment comes back flat.

The honest prior: **kWTA alone, on MNIST, at k=0.2, with this protocol, will
show no reliable effect.** That is not a reason to abandon the idea — it is a
reason to fix the harness first so that the negative result actually means
something, then test the mechanism in the form that has a chance of working.

The single highest-expected-value mechanism on the list is not kWTA at all — it
is the **complementary learning systems** memory (§4, Stage 2). Sparse codes are
best understood as the thing that *makes that memory work well*, not as a
standalone win.

---

## 1. Harness confounds — fix before trusting any number

Ordered by how much they distort the result.

### 1.1 Fixed epochs across data fractions (severity: critical)

`EPOCHS = 5`, `BATCH_SIZE = 64`, applied identically to every fraction.

| Fraction | Examples | Gradient steps in 5 epochs |
|---|---|---|
| 100% | 60,000 | ~4,690 |
| 10%  | 6,000  | ~470 |
| 1%   | 600    | ~50 |

At 1% the model takes ~50 Adam steps. It is not converged; it is barely
initialized. So the y-axis at low fractions measures *optimization budget*, not
*data efficiency*. Any mechanism that changes convergence speed (kWTA does —
hard masking slows early learning) gets scored as a data-efficiency effect.

**Fix:** decouple the two budgets. Hold **gradient steps** constant across
fractions (e.g. 3,000 steps for every condition, sampling with replacement), or
train to convergence with early stopping on a validation split. Report both
budgets in the CSV. Per root `skills.md` ("Do not silently change the data
budget … between the experimental model and the baseline") this is an
explicitly-labeled protocol change: keep the v1 results, call the new one
**protocol v2**, and never mix them in one plot.

### 1.2 No validation split (severity: high)

`get_mnist_datasets` returns train/test only. The test set is the sole eval
signal, and it will be looked at once per idea for the life of the project —
that is test-set overfitting by a thousand cuts, and it is invisible.

**Fix:** carve a fixed 5,000-example validation split out of MNIST train,
version-control the index list, and exclude those indices from
`sample_train_subset` (the current implementation samples from all 60,000, so a
val split carved after the fact would leak). Use val for early stopping and all
model selection. Touch test only for final reported numbers, and say in the
results how many times you touched it.

### 1.3 Normalization statistics leak from the full dataset (severity: medium)

`transforms.Normalize((0.1307,), (0.3081,))` are full-MNIST-train statistics,
applied unchanged in the 1% condition. `data/skills.md` lists this exact thing
under "Common Mistakes" — the repo's own rules already flag it.

The effect size on MNIST is small (mean/std of a 600-example digit sample is
close to the full-set value), but it is a self-consistency failure, it is
free to fix, and it will matter materially on CIFAR-10 later.

**Fix:** compute mean/std from the sampled training subset only; pass them into
the transform. Log the computed values per run.

### 1.4 Non-stratified subset sampling (severity: medium, rising as fraction falls)

`sample_train_subset` takes a uniform random permutation. At 1% (600 examples)
per-class counts land roughly in the 45–75 range; across seeds that class-balance
variance is comparable in size to the effect being measured, so it inflates the
error bars and can swamp a real difference.

**Fix:** stratified sampling — equal examples per class. This is also the
standard framing for few-shot work ("N-way K-shot"), which makes the results
comparable to outside literature. Keep the seed determining *which* examples,
not *how many* per class.

### 1.5 Three seeds, unpaired reporting (severity: medium)

Three seeds cannot resolve a 1-point difference against a 1–3 point seed std.
But there is free statistical power being left on the table: **both models see
the identical data subset for a given seed**, so the comparison is *paired*.

**Fix:** report the per-seed difference `acc(kwta) − acc(dense)` and its
confidence interval, not two independent means. Pairing typically halves the
effective noise. Runs at 1% take seconds — use 10 seeds at the low fractions
and 3 at 100%.

### 1.6 Smaller items

- **Results are overwritten.** `experiments/results.csv` is opened with `"w"`
  every run. Root `skills.md` says never delete a run's logs. Write to
  `experiments/results/<timestamp>_<name>/{results.csv,config.json}`.
- **No version control.** There is no `.git` in the repo, but `skills.md` requires
  logging a commit hash per run and version-controlling the splits. `git init`
  is step zero of protocol v2.
- **No param/FLOP logging**, which `architecture/skills.md` explicitly requires.
  Here the two models are architecturally identical so it is trivially satisfied
  — but the logging needs to exist *before* a mechanism that changes capacity
  arrives (Stage 2 does).
- **`assert_no_overlap` is dead code.** `data/skills.md` demands an automated
  split-disjointness check; `tests/` only covers kWTA. Add
  `tests/test_sampling.py`.
- **The fraction grid is aimed at the wrong end.** MNIST at 100% saturates near
  99% — there is no headroom to demonstrate anything. The informative range is
  below 1%. Replace `[1.0, 0.5, 0.1, 0.05, 0.01]` with a *shots-per-class* grid:
  `[1, 5, 10, 25, 100, 1000]` examples per class (i.e. 10 → 10,000 total).

---

## 2. The kWTA module is not doing what its docstring claims

### 2.1 Competition happens on the wrong axis

```python
flat = x.view(batch, -1)          # (B, 16, 28, 28) -> (B, 12544)
topk_vals, topk_idx = torch.topk(flat, k, dim=1)
```

For conv1 this is a **global top-k over channels and space jointly**: 12,544
units, k=0.2 keeps 2,508. Because a digit occupies a small fraction of the
frame, the surviving units are overwhelmingly determined by *where the strokes
are*, not by *which feature detector won*. The module is doing spatial saliency
masking. It is not lateral inhibition.

Cortical lateral inhibition — and every kWTA implementation that has shown a
data-efficiency effect (Ahmad & Scheinkman; Numenta's sparse nets) — runs the
competition **among feature detectors sharing a receptive field**: top-k across
the *channel* dimension at each spatial position. That is a competition among
16 units, not 12,544.

**Fix:** make the axis a parameter and ablate it.

```python
KWinnersTakeAll(k=0.2, dim="channel")   # top-k across C at each (h, w)  <- the real hypothesis
KWinnersTakeAll(k=0.2, dim="spatial")   # top-k across (h, w) per channel
KWinnersTakeAll(k=0.2, dim="global")    # current behaviour — keep as the ablation arm
```

For the fully-connected layer all three collapse to the same thing, so this
only changes the conv blocks — which is exactly the isolated comparison you
want.

### 2.2 No duty-cycle boosting → dead units (this is the big one)

A hard top-k with no compensation is a winner-take-all *ratchet*. A unit that
loses early receives no gradient, so its weights never move, so it keeps losing.
Effective capacity collapses over training, and the network ends up smaller than
the baseline — which turns the experiment into "does a smaller net generalize
better from less data," a different question entirely (and precisely the
confound `architecture/skills.md` warns about under "Common Mistakes").

The module's own docstring advertises a `boost_off` argument that was never
implemented.

**Fix:** implement duty-cycle boosting. Track a running average of how often
each unit wins; boost the pre-top-k score of chronically-losing units so they
re-enter the competition:

```
duty_i   <- (1 - α) * duty_i + α * won_i          # α ≈ 0.01, train mode only
boost_i  =  exp(β * (target_duty - duty_i))       # β ≈ 1.0, target_duty = k/N
score    =  x * boost                              # rank by score, mask by x
```

Boosting is applied only in `train()` mode and only for *ranking* — the values
that pass through are the unmasked originals. This is the mechanism that makes
kWTA a distributed sparse code rather than a small dense one, and I expect it to
account for most of whatever effect kWTA has.

### 2.3 The unit test gives false confidence

`test_exactly_k_units_nonzero_per_sample` feeds `torch.randn` — no zeros. In the
real network kWTA sits *after* ReLU, where a large share of activations are
exactly 0. `torch.topk` still returns k entries, so it happily "keeps" zeros and
the *actual* sparsity is data-dependent and lower than k, especially early in
training.

**Fix:** add a test on a ReLU'd input asserting the nonzero count is
`min(k, #positives)`, and document the intended semantics. Also add a
gradient-masking test — the current gradient test only asserts *some* gradient is
nonzero; it should assert gradient is **exactly zero for the losers** and
**exactly one** for the winners, which is the actual contract.

### 2.4 What is correct and should be kept

- Applying kWTA identically in train and eval is right (unlike dropout — the
  sparse code *is* the representation, not a regularizer to be averaged out).
- `ReLU → kWTA` ordering is right; selecting among non-negatives keeps the
  "winners" interpretation sensible.
- The factory-based `activation_fn` in `SmallCNN` is a genuinely good design
  choice — it is what makes the baseline and experimental model architecturally
  identical, and it should survive every stage below.
- Seeding is handled correctly: `set_seed(seed)` before each `build_fn()` gives
  both models identical initialization, because the kWTA wrapper consumes no
  RNG. Worth an explicit assertion so a future refactor cannot break it silently.

---

## 3. What the headline metric should be

"Test accuracy at 1% of data" is hard to reason about and easy to cherry-pick.
The claim the project actually wants to make is about *labels needed*, so
measure that directly:

> **Data multiplier** — how many labeled examples the dense baseline needs to
> reach the accuracy the new model reaches at budget *B*.

Read off horizontally from the two curves via interpolation. It turns the result
into one number a reader can hold: *"2.3× fewer labels for equal accuracy at the
10-shot point."* Report it with the paired-seed CI from §1.5.

Pre-register the protocol (metric, comparison point, seed count, stopping rule)
in the run config *before* running, and do not change it after seeing results.
`experiments/skills.md` already asks for this; make it mechanical.

---

## 4. Staged mechanism plan

One mechanism per experiment, per root `skills.md`. Each stage lists the
neuroscience motivation, the specific data-efficiency hypothesis, the ablation
that isolates it, and how it can fail.

### Stage 0 — Protocol v2 (no new mechanisms)

`git init`. Val split. Step-budget parity. Stratified shots-per-class sampling.
Subset-computed normalization. Per-run result directories with config + commit
hash. Param/FLOP logging. `tests/test_sampling.py`. Paired-delta reporting.

Then **re-run dense vs. kWTA unchanged** under v2. This is the reference point
for everything after, and it is the only way the eventual result — positive or
negative — means anything.

*Effort: ~1 focused session. Nothing below is trustworthy without it.*

### Stage 1 — Sparse, non-overlapping codes (the kWTA idea, done properly)

**Inspiration:** dentate gyrus pattern separation. The hippocampus expands
entorhinal input into a much larger, much sparser layer, so that similar
experiences land on nearly disjoint neuron sets and stop overwriting each other.

**Data-efficiency hypothesis:** with few examples, the dominant failure is
*interference* — gradient updates for example A degrade the representation of
example B. Sparse, low-overlap codes mean each example touches a small, mostly
disjoint set of weights, so interference falls and each example teaches more.

Sub-experiments, run in order, one variable each:

- **1a — competition axis.** `dim ∈ {channel, spatial, global}` at fixed k=0.2.
  Expect `channel` to be the only one that helps.
- **1b — duty-cycle boosting.** On/off at the winning axis. Expect the largest
  single delta here.
- **1c — k sweep.** `k ∈ {0.05, 0.1, 0.2, 0.4}`. Sparser should help more as
  labels get scarcer; if the optimal k is flat across data budgets, the
  interference story is wrong.
- **1d — expansion + sparsity.** The dentate gyrus story is *expand then
  sparsify*, not sparsify alone. Widen the FC layer 128 → 512 with k adjusted to
  keep the absolute active count constant. This changes parameter count, so it
  needs its own dense-512 control arm (this is exactly the confound
  `architecture/skills.md` warns about — budget for the extra control run).

**Measure the mechanism, not just the outcome.** Add
`experiments/analysis/overlap.py`: for a held-out batch, compute mean pairwise
Jaccard overlap of the active-unit sets, split into same-class and
different-class pairs. Track it per epoch alongside accuracy.

This matters because it decouples two failure modes. If overlap drops but
accuracy does not move, the mechanism works and the hypothesis linking it to
data efficiency is wrong — a genuinely informative result. If overlap does not
drop, the implementation is wrong. Without this metric a flat accuracy curve
tells you nothing about which of the two happened, and that ambiguity is what
kills research projects.

**How it fails:** hard masking slows convergence; under a step-budget-matched
protocol this shows up as a real loss. Watch the training curves, not just the
endpoint.

### Stage 2 — Complementary learning systems (highest expected value)

**Inspiration:** McClelland, McNaughton & O'Reilly's CLS theory. The brain runs
two learning systems: a fast, sparse, episodic hippocampus that memorizes single
experiences immediately, and a slow, distributed neocortex that extracts
statistical structure over many exposures. Humans learn from few examples partly
because the fast system does not need repetition.

**Data-efficiency hypothesis:** with 10 examples per class, a parametric softmax
head is estimating 128×10 weights from 100 points — hopeless. A non-parametric
memory over those same 100 points is *exact*. Blending them should dominate
either alone precisely in the low-label regime, and the advantage should vanish
as labels grow — a falsifiable, shape-specific prediction.

**Implementation** (`architecture/modules/episodic.py`):

- Store `(penultimate_embedding, label)` for every training example in the
  current budget. At 10-shot that is 100 vectors; memory cost is irrelevant.
- At inference: `logits_mem = Σ_j softmax(cos(z, k_j) / τ) · onehot(y_j)`.
- Blend: `logits = α · logits_parametric + (1 − α) · logits_mem`,
  with α selected on **validation** (never test).
- Refresh keys at the end of each epoch so they track the moving encoder.

**Why it composes with Stage 1:** cosine similarity in a dense 128-d space is
noisy; in a sparse, low-overlap code, nearest-neighbour lookup is far more
discriminative. Sparse codes are what make episodic retrieval *work*. Running
Stage 2 both with and without Stage 1's encoder is the cleanest evidence that
the sparsity mechanism earns its place.

**Discipline:** the memory holds only training-subset examples — no leak — but
its contents *are* part of the model, so log memory size and key dimension in
the params/FLOPs columns. And note the ablation is free: α=1 recovers the
baseline exactly, so the mechanism is trivially isolatable.

**How it fails:** if the encoder is bad (which at 10-shot it is), the embedding
space is bad, and kNN in a bad space is also bad. Mitigation is Stage 3 — an
encoder trained without labels.

### Stage 3 — Local Hebbian learning in the front end

**Inspiration:** cortex does not backpropagate a global error signal through ten
layers. Early sensory areas self-organize from unlabeled input via local
plasticity — Hebbian potentiation shaped by lateral inhibition.

**Data-efficiency hypothesis:** the honest version of "learning from less data"
is *less labeled* data. A child sees an enormous unlabeled visual stream before
learning ten object names. Training conv1/conv2 unsupervised on all 60,000
unlabeled MNIST images and then fitting only the classifier on 10 labels per
class separates the two budgets, and is a far better model of the biological
claim than shrinking both together.

**Concrete recipe:** Krotov & Hopfield's *Unsupervised learning by competing
hidden units* (PNAS 2019). A local Hebbian rule with a winner-take-all
competition among hidden units, learning MNIST features with no labels and no
backprop, competitive with supervised pretraining at low label counts.

The elegant part for this project: **the competition in Krotov–Hopfield's
learning rule is the same lateral inhibition as Stage 1's forward pass.** One
mechanism, two roles — inference-time sparsification and learning-time credit
assignment. If Stage 1 identifies which inhibition axis works, that finding
transfers directly here. That is a real architectural thesis, not a bag of
tricks, and it is what would make this project a coherent contribution rather
than a benchmark sweep.

**Reporting discipline:** every result must state *both* budgets — unlabeled
examples and labeled examples. A model that used 60,000 unlabeled images is not
comparable to one that used 100 images total, and presenting it as such would be
the single most damaging mistake available to this project. Add
`n_unlabeled_examples` as a mandatory CSV column at Stage 0 so it is impossible
to omit later.

**Prominently document** that this mechanism requires a non-standard training
procedure (custom update, no optimizer for those layers) — `architecture/skills.md`
asks for exactly this, because it silently breaks assumptions about optimizers
and mixed precision elsewhere.

**How it fails:** this is the highest-risk stage. Hebbian features are usually
somewhat worse than backprop features at equal label counts and only win at very
low ones. Do it last, gate it on Stage 1's axis result, and treat "no better than
supervised pretraining" as a publishable negative rather than a failure.

### Stage 4 — Scaling honesty (only after 1–3 show something)

Re-run the winning configuration on CIFAR-10 under identical protocol. Expect
the effect to shrink — MNIST is nearly linearly separable and flatters
regularizers of every kind. A mechanism that survives CIFAR-10 is real; one that
does not is an MNIST artifact, and finding that out early is worth more than a
year of MNIST results.

---

## 5. Suggested order of work

| # | Work | Depends on | Rough effort |
|---|---|---|---|
| 1 | Protocol v2 harness (§1, Stage 0) | — | 1 session |
| 2 | Re-run dense vs. kWTA under v2 | 1 | minutes of compute |
| 3 | kWTA axis + boosting rewrite (§2) | 1 | 1 session |
| 4 | Overlap metric (1d) | 3 | short |
| 5 | Stage 1 ablations 1a–1c | 3, 4 | compute-bound |
| 6 | Episodic memory (Stage 2) | 1 | 1 session |
| 7 | Stage 2 × Stage 1 interaction | 5, 6 | compute-bound |
| 8 | Krotov–Hopfield front end (Stage 3) | 5 | 2 sessions, high risk |
| 9 | CIFAR-10 replication (Stage 4) | 7 or 8 | compute-bound |

Items 1–3 are the ones that change whether any subsequent result is meaningful.
Nothing after them is worth starting first.

---

## 6. Calibration — what to actually expect

- **kWTA alone, properly implemented, on MNIST: a small effect at best**, likely
  1–3 points at ≤10 shots and nothing at 100%. Published sparse-net results are
  in that range. If Stage 1 comes back flat, that is the expected outcome, not a
  failure — and with the overlap metric in place it will be a *specific* result
  ("codes did separate, separation did not buy accuracy") rather than a shrug.
- **Episodic memory at ≤10 shots: a large effect**, plausibly 5–15 points. This
  is the well-understood part of the space; it is nearly free to implement and it
  is where the project's first real win most likely comes from.
- **Hebbian front end: high variance.** Could be the most interesting result in
  the project or a dead end. Its value is that it is the genuinely novel arm.
- **MNIST is a weak testbed for this claim.** Plan on moving to CIFAR-10, and
  treat every MNIST conclusion as provisional until it replicates there.

The reframe worth internalizing: the project's stated goal is data efficiency,
but the mechanism list is really about *three different bottlenecks* —
interference (Stage 1), missing episodic recall (Stage 2), and label dependence
(Stage 3). They are not competing hypotheses. The strongest version of this
architecture uses all three, and the reason to run them separately first is only
so you can attribute the effect afterwards.
