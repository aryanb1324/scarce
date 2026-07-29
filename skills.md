# Project Skill File

## Project Overview

This project explores a new neural network architecture designed to reach
strong performance with substantially less training data than standard
architectures. The design draws inspiration from mechanisms in the human
brain (e.g. sparse activation, hierarchical/predictive processing,
few-shot generalization, local learning rules) rather than assuming these
are copied literally from neuroscience — the goal is a working,
benchmarkable model, not a biological simulation.

Core workflow: propose a mechanism → implement it as a module → test it in
isolation on a small controlled task → integrate into the full architecture
→ benchmark against a standard baseline on identical data budgets →
ablate → record results.

**First experiment (in progress):** dense ReLU baseline vs. a k-Winner-
Take-All (kWTA) sparse activation, both on a small CNN trained on MNIST,
compared across shrinking training-data fractions (100% down to 1%).
Hypothesis: forced sparsity (like lateral inhibition in cortex) reduces
interference between examples and improves generalization from less data.

## Tech Stack

- Language: Python 3.11+
- Framework: PyTorch + torchvision
- Experiment tracking: plain CSV logging for now (experiments/results.csv)
  — move to Weights & Biases or MLflow once experiments outgrow one script
- Config management: none yet — first experiment is a single script
  (experiments/train.py) with constants at the top. Move to a config
  system (Hydra or plain YAML) once you're running more than a couple of
  variants.
- Environment: any Python 3.11+ env manager (venv/conda/uv); CPU is fine
  for MNIST-scale experiments, no GPU required to get started

## Architecture

- `architecture/` holds the novel model code — the actual brain-inspired
  modules and how they compose into the full network.
- `data/` holds dataset loading, preprocessing, and the low-data sampling
  logic that is central to the data-efficiency claim.
- `experiments/` holds training scripts, configs, and results — this is
  where a "run" happens, using modules from `architecture/` and `data/`.
- `tests/` holds unit tests for individual modules (not full training
  runs) so that a new mechanism can be validated cheaply before you spend
  compute on a full training loop.

## Folder Structure

- `architecture/`: model definitions, custom layers/modules, the specific
  neuro-inspired mechanisms under test
- `data/`: dataset classes, preprocessing, low-data / few-shot sampling
  utilities
- `experiments/`: training scripts, run configs, logs, results, ablations
- `tests/`: unit tests for individual modules
- `skills.md` files in each of the above: directory-specific rules

## Commands

### Environment setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run unit tests (do this before any full training run)
```bash
python -m pytest tests/ -v
```

### Run the first experiment (dense vs. kWTA across data fractions)
```bash
python -m experiments.train
```

### Plot the results
```bash
python -m experiments.plot_results
```

## Coding Conventions

- Every new architectural mechanism gets its own module in `architecture/`
  with a short docstring stating: (1) what it does, (2) the neuroscience
  concept it's loosely inspired by, (3) why it should help with data
  efficiency specifically — not just "why it might help performance."
- Set and log a random seed for every run. Data-efficiency claims are
  fragile to seed variance; never report a single-seed result as a
  conclusion.
- Every new module gets a minimal unit test in `tests/` before it's wired
  into the full architecture — verify output shapes, gradient flow, and
  behavior on a toy input before spending a full training run on it.
- Keep the "standard baseline" architecture and its training recipe
  fixed and version-controlled, so every comparison is against the same
  target, not a moving one.

## Editing Rules

- Do not change the baseline model or its training config to make a
  comparison look better — if the baseline needs to change, that's a
  separate, explicitly-labeled experiment.
- Do not silently change the data budget (dataset size, epochs, or
  effective data seen) between the experimental model and the baseline —
  the data-efficiency comparison is only valid if this is controlled.
- Make the smallest change that tests one mechanism at a time. Avoid
  bundling multiple architectural changes into one experiment — you won't
  know which one caused the effect.
- Do not delete failed experiment logs or configs. Move them to an
  `experiments/archive/` or similar rather than deleting — negative
  results are information.

## Dangerous Areas

- `experiments/configs/baseline_*.yaml` (or equivalent) — this defines the
  comparison point for every claim in the project; changes here need to be
  deliberate and logged.
- Anything touching the data split / sampling logic in `data/` — a subtle
  leak between train/val/test here would invalidate the entire
  data-efficiency claim.
- Evaluation metric code — changing how a metric is computed mid-project
  makes historical results incomparable.

## Debugging Playbook

When a new mechanism doesn't work or a run produces strange results:

1. Reproduce on the smallest possible setup (tiny synthetic data, few
   steps) before debugging on the full run.
2. Check output shapes and gradient flow through the new module in
   isolation (this is what the `tests/` unit test should already cover —
   if it wasn't covered, add it now).
3. Check the loss curve shape, not just the final number — divergence,
   plateaus, or NaNs early on usually point to a specific cause
   (initialization, learning rate, or a broken forward pass).
4. Check whether the data pipeline is actually delivering what you think
   it is (log a batch and inspect it directly).
5. Isolate: disable the new mechanism and confirm the baseline still
   trains normally in the same harness, to rule out infrastructure bugs.
6. Once the root cause is found, update the relevant `skills.md` with what
   was learned.

## Self-Improvement Rule

Whenever you make progress, fix a bug, discover an architectural detail,
or receive a correction from the user, update the relevant skill file.

- If the lesson applies to the whole project (e.g. a data leakage bug, a
  seed-handling rule), update this root file.
- If the lesson is specific to one directory, update that directory's
  `skills.md`.
- Write specific, reusable rules — not vague reminders like "be careful
  with data splits." Say what happened and what rule prevents it.

## Environment

**The runtime is Python 3.9** (`/Library/Developer/CommandLineTools/usr/bin/python3`
on the Mac). Code for this project is often drafted in a newer environment, so
syntax that is legal where it was written can fail on import here.

Do not use, anywhere in this repo:

- PEP 604 unions in annotations — `float | int` → `Union[float, int]`, or put
  `from __future__ import annotations` directly after the module docstring, which
  makes every annotation in the file a lazy string and is the safer blanket fix.
- PEP 585 builtin generics — `list[str]`, `dict[str, int]` → `List[str]`, `Dict[...]`.
- `match` statements, `ExceptionGroup`, `tomllib`.

Also avoid `Tensor.any(dim=(a, b))` / `.all(dim=(a, b))` — tuple `dim` is newer
than tuple `dim` for `sum`/`mean`, which do work. Use `.flatten(2).any(dim=2)`.

**How to catch it:** a syntax error in a new module aborts the whole run at
import, after MNIST loads but before any training — so it costs a round trip, not
a sweep. Import the new module on its own first (`python -c "import
architecture.modules.X"`) before launching an hour-long experiment.

## Recent Lessons

### 2026-07-27 — Fixed epochs across data fractions is not a data-efficiency experiment

**What happened.** The first experiment (v1) held `EPOCHS = 5` constant while the
training set shrank from 60,000 to 600. That means the 1% condition ran ~50
optimizer steps and the 100% condition ran ~4,690. kWTA lost at every budget, and
the deficit grew monotonically as data shrank (+0.01 pts at 60k, −2.29 at 3k,
−5.38 at 600). The deficit correlates with log(gradient steps) at r = 0.92.

**Why it matters.** kWTA masks ~80% of units, so it updates roughly a fifth of the
weights per step and needs more steps to converge. The harness gave it fewest
steps exactly where it needed most. The result is therefore uninterpretable as a
data-efficiency finding: it cannot separate "sparsity hurts generalization from
few examples" from "sparsity converges slower and was scored before it finished."

**The rule.** *When the training-set size is the independent variable, the
optimization budget must be held constant — fixed gradient steps, or train to
convergence with best-on-validation selection. Never fixed epochs.* Fixed epochs
silently makes dataset size and optimization budget the same variable, so any
mechanism that changes convergence speed gets misread as a data-efficiency effect.

**Corollary.** Any result where the effect size tracks the number of gradient
steps should be treated as an optimization artifact until a step-matched re-run
says otherwise. Compute that correlation before interpreting a curve.

Applies project-wide. See `experiments/train_v2.py` for the corrected protocol
and `PLAN.md` for the full audit.

### 2026-07-27 — Report the paired difference, not two independent means

Both models see the identical training subset for a given seed, so the comparison
is paired and the per-seed difference has much lower variance than either arm.
Reporting `mean(kwta) ± sd` and `mean(dense) ± sd` separately discards that. At
6,000 labels the arms overlap heavily while the paired delta is a consistent
−0.64; at 3,000 the paired t is −11 on only 3 seeds.

**The rule.** *Report `acc(new) − acc(baseline)` per seed, its mean, and its SE.
Report sign consistency across seeds explicitly.* Three seeds can support a
conclusion when the comparison is paired and every seed agrees in sign; the same
three seeds support nothing when reported unpaired.

### 2026-07-28 — A mechanism can work perfectly and still not pay

Protocol v2's two diagnostics disagreed, and the disagreement was the result.
kWTA's code separation was 2.24x the dense baseline's at 600 labels, decaying
monotonically to nothing by 30,000 — the hypothesis's mechanistic prediction,
confirmed. Meanwhile 44–49% of its units were permanently dead. It bought better
codes with half the network, and the two effects cancelled to "no difference."

Accuracy alone would have read as "sparsity does nothing." With both diagnostics
it reads as "sparsity does exactly what it claims and the implementation charges
too much for it," which names the next experiment instead of ending the line of
inquiry.

**The rule.** *Measure the mechanism's benefit AND its cost, separately, in the
same run.* A null accuracy result is only interpretable when you can see which
side of that ledger moved. Every mechanism from here gets both: what it buys, and
what it spends (capacity, compute, parameters, or steps).

### 2026-07-28 — An aggregate metric hid WHERE the problem was, and that changed the diagnosis

The v2 dead-unit number was reported as one figure: 44–49% of units. It read as
network-wide capacity collapse, and the pre-registered rule said to fix that
before judging the mechanism. Stage 1 added a per-layer breakdown and the picture
inverted: **94–99% of the dead units are in the FC layer** (56–63% of its 128
units), while conv1 sits at 0–2.5% and conv2 at 3–13%. The aggregate looked
alarming only because FC holds 128 of the 176 counted units.

That reframing mattered, because `dim` only changes the conv blocks — for a 2-D
input both competition rules are identical — so the axis change could not have
touched the layer where nearly all the dead units live, yet it is what produced
the +1.43. Dead units were never the binding constraint.

**The rule.** *Any diagnostic aggregated over structurally different components
gets reported per component.* A single number weighted by component size will be
dominated by the biggest one and will point at the wrong cause.

### 2026-07-28 — Following a pre-registered rule that turns out to be wrong is the point

The registered rule was "dead > 30% → fix duty-cycle boosting before judging the
idea." It was followed. Boosting did reduce dead units (46.8% → 43.0%) and did
raise separation (0.133 → 0.157) — and it cost 7.6 points of accuracy, with a
25.8-point spread across seeds. The rule's premise was falsified by executing it.

Had the rule been quietly dropped after seeing that channel-wise already worked,
the project would have kept "boosting should help" as an untested belief.

**The rule.** *Run the pre-registered branch even when a different arm has already
made it look unnecessary.* Its value is in what it rules out, and a mechanism that
achieves its stated mechanical goal while hurting the outcome is a specific,
reusable finding — not a wasted run.

**Corollary — read variance, not just means.** The boosted arms' seed spreads
(10–26 points, against 0.3–0.8 unboosted) were the diagnostic signal. Smooth
degradation and wild instability have different causes; instability of that shape
points at a train/eval inconsistency rather than a worse-but-working mechanism.

### 2026-07-28 — The baseline was converging on the same solution by itself

The dense net's active-unit count fell 77.3 → 45.9 (of 128) as labels went 600 →
48,000, and its code separation rose 0.057 → 0.260, overtaking kWTA's by the
largest budget. Given enough data the baseline *discovers* the sparse, separated
code that kWTA imposes by construction.

**Why it matters.** It reframes the whole project. A brain-inspired mechanism
that hard-codes structure is a **prior**, and a prior only earns its place where
data is too scarce to learn that structure — which is precisely the regime this
project targets, so this is encouraging rather than deflating. But it also sets
the bar: the mechanism must be cheaper than learning the structure, or it loses
to just using more data.

**How to apply.** For every proposed mechanism, ask first: does the baseline
already learn this given enough data? If yes, measure the crossover point — that
is the mechanism's actual region of usefulness, and the honest claim.

### 2026-07-28 — Do not read p-values off three seeds

v2's paired tests gave p = 0.024 for a −0.19 pt effect at 6,000 labels (tiny,
uninteresting, "significant") and p = 0.14 for a −1.58 pt effect at 600 labels
(large, decision-relevant, "not significant"). With n = 3 the sd estimate is so
unstable that the p-value tracks luck in the spread rather than effect size.

**The rule.** *Treat n = 3 as a direction check, not a significance test.* Use
sign consistency and effect magnitude to decide what to run next; use 5+ seeds
before a number goes in a conclusion, and more where the decision hinges on it.

### 2026-07-28 — "5+ seeds" was not enough either, at the noisiest budget

The rule above said 5 seeds is enough for a conclusion. A 20-seed power run says
that depends entirely on the budget, and the rule as written produced a false
finding.

Stage 1b reported kWTA at **−0.79 vs dense at 100 labels** (5 seeds) and it was
written up as a falsification: "the advantage does not grow as labels get scarcer
— it reverses." Re-running the identical cell with 20 seeds:

    100 labels, paired:  −0.06 pts   sd 1.375   se 0.31   t = −0.20
                         95% CI [−0.67, +0.54]   signs 12+/8−

The first five seeds reproduce **−0.794**, against the recorded −0.79 — so this is
not a pipeline change or a different subset. Seeds 5–19 simply cancel it. The
"reversal" was one particular quintet of seeds.

The reason is that the paired sd is **not constant across the sweep**: 1.375 at
100 labels against 0.50–0.87 at 600. It grows as labels fall — precisely where the
scarcity claims live. At that sd, resolving the claimed 0.79 pt needs ~24 seeds;
five gives roughly 20% power.

**The rule.** *Measure the paired sd at a budget before interpreting any effect at
that budget, and state the seeds needed for the effect size you intend to claim.*
Seed counts are per-budget, not per-project. A useful reflex: an effect under
~2 × se is a coin flip regardless of how clean the sign split looks — 12+/8− here
looked like a trend and was nothing.

**What survives.** The non-monotonicity is still real, but weaker than recorded:
~0 at 100 labels, +1.58 at 300, +1.66 at 600, +0.19 at 3,000. A flat null at the
scarcest end, not a reversal. Correct the claim rather than keeping the stronger
version.

### 2026-07-28 — "Which arm won" needs the head-to-head test, not two deltas
### against the reference

Building `scarce` surfaced a gap that the experiment scripts have had all along.
Every run reports each arm's paired delta *against dense*, and the arm with the
biggest delta gets called the winner. That is not the same question.

At 300 MNIST labels, 5 seeds: `kwta_channel` +1.91 and `dropout` +1.68. Both beat
dense significantly. But the paired **head-to-head** difference is +0.23 with
sd 0.72 -- p = 0.52, and two of five seeds favour dropout. The arms are
indistinguishable, and "kwta wins at 300" was never supportable. At 600 labels the
same comparison gives +0.52, sd 0.27, p = 0.0124: genuinely resolved.

The head-to-head costs nothing. Both arms' deltas are measured against dense at
the same seed, so dense cancels and `delta_a[i] - delta_b[i]` is exactly the
paired difference -- no extra runs, just arithmetic that was not being done.

**The rule.** *Ranking arms by their deltas against a shared reference is not a
comparison between the arms. Before naming a winner among several arms, compute
the paired head-to-head difference against the runner-up and report its p.* An
arm leads the table by a margin smaller than the noise more often than intuition
suggests, especially when every arm beats the reference.

**Corollary — decide ties against your own hypothesis.** When the margin is
unresolvable, prefer the simpler or more standard arm. Here that means returning
`dropout` over the project's own mechanism at 300 labels. A rule that breaks ties
toward the thing you are hoping for will find support in noise indefinitely.

### 2026-07-27 — A mechanism needs a metric of its own, not just accuracy

kWTA's stated hypothesis is that sparse codes reduce interference between
examples. Accuracy alone cannot tell you whether the mechanism failed or the
hypothesis linking the mechanism to accuracy failed — and those call for opposite
next steps.

**The rule.** *Every mechanism ships with a direct measurement of the thing it
claims to do, logged alongside accuracy.* For kWTA: pairwise overlap of active-unit
sets (same-class vs different-class) and the fraction of units that never win.
For episodic memory: retrieval accuracy in isolation. Without it, a flat result is
a shrug instead of a finding.
