# Tests Directory Skill File

## Purpose
Unit tests for individual architecture/data modules — fast, isolated,
run before any full training run.

## Rules
- Every new module in architecture/ or data/ gets a test here before
  being wired into a full training run.
- Tests should use tiny toy tensors, not real data — they check shape
  and gradient behavior, not model quality.

## Validation
Run all tests: python -m pytest tests/ -v

## Lessons

### 2026-07-28 — Seed every test that samples, or it fails on someone else's machine

`test_selection_ignores_the_input` used an unseeded `torch.randn` and failed on
roughly 3 runs in 5, blocking a sweep for no reason. The assertion that broke was
not even the one under test: it was the secondary check that kWTA keeps a channel
boosted ×100 more than 99% of the time, which came in at 98.93%. That is correct
behaviour — where the boosted channel's own pre-boost value is near zero, ×100
leaves it small and it legitimately loses the competition — so the threshold, not
the module, was wrong.

**Rule:** *any test that draws random data calls `torch.manual_seed` first.* An
unseeded statistical assertion is a test that fails on a schedule you do not
control, and the cost lands at the worst moment: right before a 55-minute run,
where it is indistinguishable from a real regression.

**Rule:** *a threshold on a sampled quantity needs slack for its tail, or it needs
to assert the distribution instead.* Prefer `> 0.98` with a seed, or assert the
rate directly, over a bound that happens to hold for the draw you tried once.

### 2026-07-28 — Hand-written fixtures for a statistical test encode your
### assumptions, and then the test confirms them

`test_an_unresolvable_margin_does_not_sell_the_fancier_arm` was written with
invented per-seed numbers meant to represent "kwta +1.9, dropout +1.7, tied."
The invented arms tracked each other far too tightly seed-for-seed
(differences [0.3, 0.1, 0.2, 0.2, 0.4]), so the *paired* difference had almost no
variance and came out significant at p = 0.009. The test failed, and read as
"the tie-breaking feature is broken."

The real measured deltas from `examples/mnist_low_data` differ [-0.70, -0.02,
+0.78, +1.12, -0.04] -- mean +0.23, p = 0.52, two seeds favouring the other arm.
Genuinely tied, and the feature was correct all along. The fixture, not the code,
carried the wrong belief: real arms are far less correlated across seeds than
intuition suggests, because seed noise is largely independent between them.

**Rule:** *for a test of a statistical decision rule, use real measured values
from a recorded run as the fixture, and cite the run.* Invented numbers smuggle
in an assumption about the correlation structure, which is usually the exact
quantity the rule is deciding on. If a fixture must be synthetic, assert its
statistics (sd, p, sign split) explicitly so the assumption is visible instead of
implicit.

**Corollary:** a failing test whose fixture you wrote by hand is ambiguous
evidence. Check whether the fixture is realistic before changing the code -- here,
"fixing" the code to make that test pass would have broken a correct feature.

