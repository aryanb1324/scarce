# Architecture Directory Skill File

## Purpose

This directory contains the novel model code: the custom modules
implementing brain-inspired mechanisms, and the full architecture that
composes them.

## When to Edit This Directory

Edit this directory when:
- Implementing a new mechanism or module
- Changing how existing modules compose into the full network
- Refactoring a module's internals without changing its interface

Do not edit this directory when:
- The issue is actually a data pipeline bug (see `data/skills.md`)
- The issue is a training-loop or hyperparameter problem, not the model
  itself (see `experiments/skills.md`)

## Important Files

- `modules/`: individual mechanisms, one file per mechanism
- `model.py` (or equivalent): composes modules into the full architecture
- `baseline.py` (or equivalent): the fixed standard-architecture baseline
  used for every comparison — see root skill file's "Dangerous Areas"

## Local Architecture

Each module should be self-contained: takes a standard tensor shape in,
returns a standard tensor shape out, with no hidden dependency on global
state. This makes it possible to swap mechanisms in and out of `model.py`
to test them individually, and to unit-test them in isolation in `tests/`.

## Rules

- Every module's docstring states: what it does, the neuroscience concept
  it's loosely inspired by, and the specific hypothesis for why it should
  improve data efficiency (not just accuracy).
- New modules must be tested standalone in `tests/` before being wired
  into `model.py`.
- Keep `baseline.py` untouched except in explicitly-labeled experiments —
  it's the fixed comparison point for the whole project.
- Prefer composability over one giant monolithic model class — a
  mechanism that can't be isolated can't be properly ablated later.
- Document any mechanism that requires a non-standard training procedure
  (e.g. local learning rules, custom backward passes) prominently, since
  this can silently break assumptions elsewhere (e.g. standard optimizers,
  mixed precision).

## Common Mistakes

- Mistake: A new mechanism changes effective parameter count or FLOPs
  substantially, confounding a "data efficiency" comparison with a
  "just a bigger/smaller model" comparison.
  - Why it happens: Easy to add a mechanism without checking its cost.
  - How to avoid it: Log parameter count and FLOPs for every architecture
    variant alongside every result.

- Mistake: A mechanism is tested only end-to-end, so when it fails to
  help, it's unclear whether the idea or the implementation is at fault.
  - Why it happens: Skipping the isolated unit test to save time.
  - How to avoid it: Always validate shape/gradient behavior on a toy
    input before a full training run.

## Debugging Playbook

When a module misbehaves:

1. Test it standalone with a toy input and check output shape.
2. Check gradients flow through it (no vanishing/exploding on a toy case).
3. Compare its behavior with the mechanism disabled (does the rest of the
   architecture still function normally?).
4. Only then debug it inside the full training loop.

## Validation

After editing this directory, run:
```bash
# fill in, e.g.:
# python -m pytest tests/ -k "architecture"
```

## Self-Improvement Rule

If a bug, correction, or design insight is specific to a module or to how
modules compose, record it here with the mechanism name and what was
learned.

## Mechanism Notes

### kWTA (`modules/kwta.py`) — the competition axis is a design decision, not a detail

The v1 implementation flattens `(B, C, H, W)` to `(B, C*H*W)` and takes a single
global top-k. For conv1 that is a competition among 12,544 units spanning
channels *and* space, so the survivors are determined mostly by where the digit's
strokes are — it is spatial saliency masking, not lateral inhibition.

Cortical lateral inhibition, and every sparse-net result that has shown a
data-efficiency effect, competes among **feature detectors sharing a receptive
field**: top-k across the channel dimension at each spatial position (a
competition among 16 units, not 12,544).

**Rule:** for any competition/normalization mechanism on a conv feature map, state
which axis it operates over and treat that axis as an ablation arm. `channel`,
`spatial`, and `global` are three different mechanisms wearing one name. For a
fully-connected layer all three collapse to the same thing, so the axis only
changes the conv blocks — which makes it a clean isolated comparison.

### kWTA — hard top-k without duty-cycle boosting is a ratchet

A unit that loses early receives no gradient, so its weights never move, so it
keeps losing. Effective capacity collapses over training and the network ends up
smaller than the baseline — which silently converts a data-efficiency experiment
into a model-size experiment (the exact confound in "Common Mistakes" above).

The docstring advertises a `boost_off` argument that was never implemented.

**Rule:** any hard selection mechanism (top-k, routing, gating, mixture-of-experts
style dispatch) needs an explicit participation-balancing term, and a logged
**dead-unit fraction** to prove it is working. Track a running duty cycle per unit
and boost chronically-losing units' *ranking* score — never the value that passes
through — in train mode only. `experiments/train_v2.py` measures the dead fraction
via forward hooks; run it before concluding anything about a masking mechanism.

### kWTA — the v1 predictions, now measured (protocol v2, commit 9e7874f)

Both predictions above were confirmed quantitatively, so treat them as evidence
rather than as reasoning:

- **Dead units: 44–49%** across every data budget, stable rather than transient.
  The ratchet is real and it is the module's dominant cost.
- **Pattern separation works.** Same-class minus different-class Jaccard overlap
  of the penultimate code, kWTA relative to dense: 2.24x at 600 labels, 1.21x at
  3,000, 1.07x at 6,000, 0.99x at 30,000, 0.88x at 48,000. Monotonic, and largest
  exactly where data is scarcest.
- **Effective sparsity is below k, and drifts.** The FC layer selects k = 26 units
  but only ~21.6 were nonzero at the largest budget — `topk` is selecting zeros,
  as the test-on-`randn` note below predicts.

So the mechanism is sound and the implementation overcharges for it. `kwta_v2.py`
makes the two candidate causes (competition axis, duty-cycle boosting)
independently switchable at **matched total sparsity** — channel-wise keeps
k*C channels at each of H*W positions, which is exactly the k*C*H*W the global
rule keeps — so an accuracy difference between arms cannot be explained by one
arm simply being less sparse. Always check that kind of matching before
attributing an effect to a mechanism.

### Unit tests on `randn` can hide activation-dependent behavior

`tests/test_kwta.py` feeds `torch.randn`, which has no zeros, so the "exactly k
units nonzero" test passes. In the real network kWTA sits after ReLU where many
activations are exactly 0; `torch.topk` returns k entries regardless, so it
"keeps" zeros and the true sparsity is data-dependent and lower than k.

**Rule:** test a module on inputs with the *distribution it will actually see*
(post-ReLU, post-normalization), not just on Gaussian noise. And assert the real
contract: gradient exactly zero for masked units, exactly one for survivors — not
merely that "some gradient is nonzero."
