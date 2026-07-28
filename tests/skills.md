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

