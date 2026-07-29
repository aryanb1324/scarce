"""
k-Winner-Take-All (kWTA) sparse activation.

Biological motivation: in cortex, lateral inhibition means that when a
group of neurons receives input, only the most strongly-activated ones
actually fire; the rest get suppressed. This module does the same thing
to a layer's activations: keep only the top-k values per example, zero
out everything else.

Hypothesis being tested: this sparsity acts like a strong, structured
regularizer. Because only a few units are ever "on" for any given input,
different examples are less likely to interfere with each other's
representations, which should help the network generalize from fewer
training examples than a dense (all-units-active) network would need.

This is a drop-in replacement for a normal activation like ReLU. It does
not change the training loop, optimizer, or loss function at all.
"""

import torch
import torch.nn as nn
from typing import Union

class KWinnersTakeAll(nn.Module):
    """
    Keeps only the top-k activations per sample (across the feature/channel
    dimension) and zeros out the rest. Everything else about the layer
    (weights, gradients, optimizer) works exactly as normal — this module
    just masks the output.

    Args:
        k: number of units to keep active. Can be an int (exact count) or
            a float in (0, 1] interpreted as a fraction of the total units.
        boost_off: if True, applies no gradient-boosting to "losing" units
            (kept simple on purpose — this is the version to start with).
    """

    def __init__(self, k: Union[float, int]):
        super().__init__()
        self.k = k

    def _resolve_k(self, num_units: int) -> int:
        if isinstance(self.k, float):
            k = max(1, int(round(self.k * num_units)))
        else:
            k = min(self.k, num_units)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features) for a linear layer, or
        # (batch, channels, H, W) for a conv layer. We flatten everything
        # after the batch dim, apply top-k, then reshape back.
        orig_shape = x.shape
        batch = orig_shape[0]
        flat = x.view(batch, -1)
        num_units = flat.shape[1]
        k = self._resolve_k(num_units)

        # Find the top-k values per sample, build a mask, zero out the rest.
        topk_vals, topk_idx = torch.topk(flat, k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, topk_idx, 1.0)

        sparse = flat * mask
        return sparse.view(orig_shape)

    def extra_repr(self) -> str:
        return f"k={self.k}"
