# scarce — brain-inspired data-efficient architectures

Testing whether brain-inspired structural priors let a network learn from fewer
labels. Small CNN, MNIST, CPU-scale. The point is not to build a big model — it is
to get **trustworthy** measurements of small effects.

## Runtime — read this before writing any code

**Python 3.9** (`/Library/Developer/CommandLineTools/usr/bin/python3`). Code is
often drafted elsewhere and pasted in, so newer syntax fails at import:

- No PEP 604 unions in annotations (`float | int`) — use `Union[float, int]`, or
  put `from __future__ import annotations` right after the module docstring.
- No PEP 585 builtin generics (`list[str]`, `dict[str, int]`) — use `typing`.
- No `match` statements.
- No `Tensor.any(dim=(a, b))` / `.all(dim=(a, b))` — tuple `dim` works for
  `sum`/`mean` but not these. Use `.flatten(2).any(dim=2)`.

**Always import a new module on its own before launching a sweep.** An import
error aborts after MNIST loads but before training, wasting the whole run:

```bash
python3 -c "import architecture.modules.<new>" && echo ok
```

## Commands

```bash
python3 -m pytest tests/ -v                        # always before a sweep
QUICK=1 python3 -m experiments.<script>            # smoke test, ~2 min
python3 -m experiments.<script>                    # full run, 40-90 min
```

Sweeps take 40–90 minutes on CPU. Run them in the background and monitor the
results CSV rather than blocking; every script appends rows as they complete, so
partial results survive an interrupt.

## Non-negotiable protocol rules

These exist because violating them already produced a wrong conclusion once.

1. **Never hold epochs fixed while the training-set size varies.** That confounds
   architecture with optimization budget — it manufactured a 5.4-point deficit out
   of a 1.6-point one and inverted the sign of the conclusion. Train to
   convergence (patience early stopping) and score the best-validation checkpoint.
2. **Report the paired per-seed delta**, its sd/se, and sign consistency — never
   two independent means. Both arms see the identical subset per seed.
3. **Treat n = 3 seeds as a direction check, not a significance test.** At n = 3
   the p-value tracks luck in the spread. Use 5+ before writing a conclusion; more
   where the decision hinges on it. Check the noise floor per budget first.
4. **Pre-register the prediction in the script docstring before running**, and do
   not edit it afterwards. Report the branches that got falsified — three have so
   far, and each one saved weeks.
5. **Every mechanism ships a measurement of the thing it claims to do**, logged
   beside accuracy. Accuracy alone cannot distinguish a failed mechanism from a
   failed hypothesis.
6. **One variable per experiment.** New protocol → new versioned entry point, never
   an in-place edit of an existing one.
7. **Report benefit and cost separately.** A mechanism can achieve its stated
   mechanical goal and still lose (duty-cycle boosting did exactly that).
8. **Any diagnostic aggregated over structurally different components gets
   reported per component.** A single number weighted by component size points at
   the wrong cause — a "47% of units are dead" figure turned out to be 94% one
   layer.

## Frozen files — do not edit

Editing these breaks reproducibility of recorded results:

- `architecture/baseline.py` — the fixed comparison point for the whole project
- `architecture/model.py`: `build_dense_model`, `build_kwta_model` — used by the
  v1/v2 runs. Add new builders; never change these.
- `data/mnist.py`, `experiments/train.py` — the v1 pipeline
- `data/mnist_v2.py`: `VAL_SPLIT_SEED = 1234` — changing it invalidates every
  recorded v2+ result
- `experiments/results/**` — run logs are the project record

## Where things are

- `PLAN.md` — audit and staged mechanism plan
- `PAPER_SKETCH.md` — what the paper would claim, and what is still missing
- `skills.md` (root) + one per directory — accumulated lessons. **Read the relevant
  one before changing that directory, and add to it when something is learned.**
  That is the project's central discipline, not paperwork.
- `experiments/results/<timestamp>_<name>/` — one dir per run, with `config.json`
  (including the git commit) and `results.csv`

@skills.md
@architecture/skills.md
@data/skills.md
@experiments/skills.md
@tests/skills.md

## State as of commit `aba0fe0`

**The result:** channel-wise kWTA (k = 0.2) beats the dense baseline by +1.55 pts
at 300 labels and +1.43 at 600 (5/5 seeds, p = 0.0003; 1.78× label multiplier).
Global top-k over the flattened tensor *loses* by 1.58 at matched total sparsity —
the competition axis is what matters.

**What has been falsified** (all pre-registered):
- The advantage does not grow monotonically as labels get scarcer — it is
  non-monotonic, peaking around 300–600 labels.
- Best k does not shift with the data budget — k\* = 0.20 across a 30× range.
- **Pattern separation is not the mechanism.** The sparsest arm has the most
  separation at every budget and the worst accuracy (r = +0.19, p = 0.48).
- Duty-cycle boosting fails via train/eval mismatch, confirmed by measuring
  winner-set agreement. Dropped.

**Open question:** is the input-dependent competition doing the work, or would any
structured sparsification do? `experiments/train_stage2_control.py` answers it
with `randk_channel` — identical k, axis, and exact surviving count, but random
winners. If kWTA ≈ randk, the honest description is "a well-tuned structured
regularizer," not lateral inhibition, and the brain-inspired framing should be
dropped rather than defended.

## Update — Stage 2 + power run (8 seeds / 20 seeds)

**Retracted.** The 100-label reversal is gone. 20 seeds give **−0.06 ± 0.31**
(95% CI [−0.67, +0.54], 12+/8−) where 5 seeds gave −0.79. The first five seeds
reproduce −0.794 exactly, so it was seed luck, not a pipeline difference. The
sweep is ~0 / +1.58 / +1.66 / +0.19 at 100 / 300 / 600 / 3,000 — still
non-monotonic, but a **flat null** at the scarcest end rather than a reversal.
Paired sd at 100 labels is 1.375 (vs 0.50–0.87 at 600); resolving 0.79 pt there
needs ~24 seeds.

**kWTA replicates at 8 seeds:** +1.58 at 300, +1.66 at 600, 8/8 seeds both.

**The randk control is confounded — do not cite the +12.9 gap as-is.** It matches
kWTA on *nominal* count but not on surviving signal: 26.0 vs **7.6** live FC units
(3.4×), because random selection lands on post-ReLU zeros while top-k cannot. The
gap is an upper bound on the competition effect. A k-corrected rerun (randk at
k ≈ 0.68, or selection restricted to nonzeros) is the outstanding task.

**The uncomfortable arm is `dropout_matched`,** which the auto-verdict ignores: it
**ties kWTA at 300 labels** (−0.11, t = −0.95, 4+/4−) and loses only at 600
(+1.01, t = 5.90). Half the pre-registered "reachable with a standard tool" branch
has fired. Dropout also has 2.5× kWTA's code separation (0.42–0.45 vs 0.16–0.18)
while being no better — an independent replication of "separation is not the
mechanism."

**Known non-novelty:** channel-wise kWTA is `local=True` in `nupic.torch`, and
duty-cycle boosting is Numenta's. The defensible contribution is the protocol, the
optimization-budget confound, and the mechanism-level measurements — not the
module. Do not draft claims that rest on inventing the mechanism.

**The gate on everything:** CIFAR-10 under identical protocol. Needs a GPU; MNIST
saturates near 99% and flatters regularizers of every kind.
