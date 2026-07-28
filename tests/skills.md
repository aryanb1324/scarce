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

