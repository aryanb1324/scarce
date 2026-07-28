"""
Composes the fixed baseline architecture (baseline.py) with either a
standard activation or the experimental sparse activation (modules/kwta.py).

This is the only file that should need to change to add a new mechanism
to test — swap in a new activation factory here, keep baseline.py frozen.
"""

import torch.nn as nn

from architecture.baseline import SmallCNN
from architecture.modules.kwta import KWinnersTakeAll


def build_dense_model(num_classes: int = 10) -> nn.Module:
    """Standard baseline: ReLU activations, fully dense."""
    return SmallCNN(num_classes=num_classes, activation_fn=nn.ReLU)


def build_kwta_model(num_classes: int = 10, k: float = 0.2) -> nn.Module:
    """
    Experimental model: same architecture as the baseline, but each ReLU
    is followed by a kWTA sparsification step.

    k=0.2 means only the top 20% most active units per layer stay on for
    any given input; the rest are zeroed. Combining ReLU -> kWTA means
    kWTA only ever selects among already-nonnegative activations, which
    keeps the "winners" interpretation sensible.
    """

    def sparse_activation_factory():
        return nn.Sequential(nn.ReLU(), KWinnersTakeAll(k=k))

    return SmallCNN(num_classes=num_classes, activation_fn=sparse_activation_factory)
