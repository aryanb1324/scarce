"""
Unit tests for the kWTA module — validate shape and sparsity behavior on
a toy input before trusting it inside a full training run. See root
skills.md: "New modules must be tested standalone before being wired in."

Run with: python -m pytest tests/test_kwta.py -v
"""

import torch

from architecture.modules.kwta import KWinnersTakeAll


def test_output_shape_matches_input():
    x = torch.randn(4, 16)
    layer = KWinnersTakeAll(k=4)
    out = layer(x)
    assert out.shape == x.shape


def test_exactly_k_units_nonzero_per_sample():
    x = torch.randn(4, 16)
    layer = KWinnersTakeAll(k=4)
    out = layer(x)
    nonzero_per_sample = (out != 0).sum(dim=1)
    assert torch.all(nonzero_per_sample == 4)


def test_fractional_k():
    x = torch.randn(2, 10)
    layer = KWinnersTakeAll(k=0.3)  # 30% of 10 = 3
    out = layer(x)
    nonzero_per_sample = (out != 0).sum(dim=1)
    assert torch.all(nonzero_per_sample == 3)


def test_keeps_largest_values():
    x = torch.tensor([[1.0, 5.0, 2.0, 9.0]])
    layer = KWinnersTakeAll(k=2)
    out = layer(x)
    # top 2 values are 9.0 and 5.0 -> only those positions should survive
    expected = torch.tensor([[0.0, 5.0, 0.0, 9.0]])
    assert torch.allclose(out, expected)


def test_gradient_flows_through_surviving_units():
    x = torch.randn(2, 8, requires_grad=True)
    layer = KWinnersTakeAll(k=4)
    out = layer(x)
    out.sum().backward()
    assert x.grad is not None
    # at least some gradient should be nonzero (the surviving units)
    assert (x.grad != 0).any()
