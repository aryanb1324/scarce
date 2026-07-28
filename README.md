# aryanet

**Find out which architecture actually helps on your low-data problem — instead of trusting one that helped on someone else's.**

```python
import aryanet

result = aryanet.fit(x_train, y_train, x_test=x_test, y_test=y_test)
print(result.report())
predictions = result.predict(x_new)
```

`fit` trains every candidate mechanism on *your* data under a protocol designed
not to fool you, measures the noise floor on *your* data, and names a winner only
if the effect clears it. When nothing does, it tells you to use a standard
network — and hands you one.

---

## The claim this library does *not* make

It does not ship a magic data-efficient architecture. Here is the actual evidence
behind the sparse mechanism it includes, measured on MNIST, paired, 8 seeds:

| labels | kWTA vs dense | verdict |
|---|---|---|
| 100 | −0.06 ± 0.31 (20 seeds) | nothing |
| 300 | **+1.58**, 8/8 seeds | real — but **tied by ordinary dropout** (−0.11, n.s.) |
| 600 | **+1.66**, 8/8 seeds | real, and beats dropout (+1.01) |
| 3,000 | +0.19 | nothing |

So: a ~1.6-point effect, in a narrow label window, on one dataset that saturates
near 99% and flatters regularizers of every kind, and at one of those two budgets
you could have gotten the same result from `nn.Dropout2d`. The mechanism itself
isn't novel either — channel-wise kWTA is `local=True` in `nupic.torch`.

**That is exactly why the library searches instead of asserting.** A fixed
"data-efficient architecture" shipped on this evidence would be an overclaim. A
tool that measures whether it helps *you* is honest, and more useful.

## Install

```bash
pip install -e .
```

Depends only on `torch` and `numpy` — the statistics are implemented directly, so
installing this doesn't drag in scipy.

## What you get back

Real output, 600 MNIST labels, 5 seeds:

```
==========================================================================
ARYANET SEARCH -- paired delta vs dense, percentage points
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

Every rule in `aryanet/protocol.py` exists because breaking it produced a wrong
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

## API

```python
aryanet.fit(
    x_train, y_train,          # (N,C,H,W) images or (N,F) tabular; numpy or torch
    x_val=None, y_val=None,    # omitted -> stratified split held out of train
    x_test=None, y_test=None,  # recommended: the only unbiased final number
    seeds=5,                   # below 5, the rule will usually decline to call one
    candidates=None,           # override the arms; must include a 'dense' reference
    include_controls=False,    # add diagnostic arms (randk)
    config=TrainConfig(),      # max_steps, patience, lr, batch_size
)
```

`SearchResult` gives you `.winner`, `.model`, `.predict(x)`, `.arms` (per-arm
paired statistics), `.cost`, `.test_acc`, and `.report()`.

Adding a mechanism is one entry — an activation factory — in
`aryanet/mechanisms.py`. It is then automatically measured against dense under
the same protocol as everything else.

## Worked example

```bash
python -m examples.mnist_low_data          # reproduction check, ~30-40 min CPU
QUICK=1 python -m examples.mnist_low_data  # ~3 min
```

The `QUICK` run deliberately uses a budget too short to converge, so kWTA looks
bad. That's not a bug — it's a live demonstration of the fixed-epoch confound the
protocol exists to prevent.

## Honest limitations

- **Validated on MNIST only.** The gating question for the whole project is
  CIFAR-10 under the identical protocol; it needs a GPU and has not been run.
  Until it is, treat the built-in mechanisms as *candidates worth measuring*, not
  as things known to help.
- **The mechanism story is unresolved.** Whether the input-dependent competition
  is doing the work, or whether any structured sparsity would do, is still open —
  the `randk` control that was supposed to answer it turned out to be confounded
  (it matches kWTA on nominal count but keeps 3.4× fewer live units, because
  random selection lands on post-ReLU zeros and top-k can't). See
  `architecture/skills.md`.
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

Three pre-registered predictions have been falsified so far, and one recorded
finding was retracted by a power run. That is the process working.

## License

MIT.
